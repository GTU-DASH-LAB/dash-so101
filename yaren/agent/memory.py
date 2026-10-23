"""Yaren's long-term memory: what she keeps between conversations.

The design follows Hermes Agent's (Nous Research) MEMORY.md / USER.md:
- two small plain-text stores with hard character limits: people.md (who she has met, what they like)
  and notes.md (her own notes about her work). A write that would overflow is refused, and the LLM
  must replace or forget something;
- a snapshot is taken once, when a conversation starts, and placed in the system prompt. Writes go to
  disk at once, but the prompt stays fixed for the conversation, so Ollama's prefix cache keeps working.

Unlike Hermes, runs are remembered automatically: every run that robot_server.py reports is appended
to runs.jsonl, and the snapshot lists the last few, so she can answer "how did it go yesterday?".

Everything lives in one directory (run/memory by default, git-ignored). It is plain text: read it,
edit it, or delete it to make her forget.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

SEP = "\n§\n"
STORES = {  # name: (file, character limit, heading in the prompt)
    "person": ("people.md", 1500, "About people you have met"),
    "yaren": ("notes.md", 1500, "Your own notes"),
}
MAX_FACT_CHARS = 200
RUNS_IN_PROMPT = 5
ENDED = {"done": "your camera check saw it finished", "timeout": "time limit reached, result unknown",
         "stopped": "stopped", "error": "failed with an error", "replaced": "replaced by a new task"}


def entry_id(text):
    """Short id from the text itself: stable whatever the order, and survives edits of other entries."""
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()[:4]


def _norm(s):
    return " ".join(s.lower().split())


class Memory:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- the two fact stores
    def entries(self, store):
        f = self.root / STORES[store][0]
        if not f.exists():
            return []
        return [e.strip() for e in f.read_text().split(SEP) if e.strip()]

    def _write(self, store, entries):
        f = self.root / STORES[store][0]
        tmp = f.with_suffix(".tmp")
        tmp.write_text(SEP.join(entries) + ("\n" if entries else ""))
        os.replace(tmp, f)

    def usage(self, store, entries=None):
        entries = self.entries(store) if entries is None else entries
        return len(SEP.join(entries)), STORES[store][1]

    def find(self, key):
        """(store, index) of the entry with this id, else of the only entry containing the text."""
        key = (key or "").strip()
        hits = []
        for store in STORES:
            for i, e in enumerate(self.entries(store)):
                if entry_id(e) == key.lower():
                    return store, i
                if len(key) >= 4 and _norm(key) in _norm(e):
                    hits.append((store, i))
        return hits[0] if len(hits) == 1 else None

    def listing(self):
        return [{"id": entry_id(e), "about": s, "fact": e} for s in STORES for e in self.entries(s)]

    def add(self, fact, about="person", replaces=None):
        """The result goes back to the LLM as the tool result, so errors say what to do next."""
        fact = " ".join(str(fact or "").split())
        if about not in STORES:
            about = "person"
        if not fact:
            return {"error": "empty fact"}
        if len(fact) > MAX_FACT_CHARS:
            return {"error": f"too long ({len(fact)} chars, max {MAX_FACT_CHARS}): say it in one short sentence"}
        with self.lock:
            old, moved_from = None, None
            entries = self.entries(about)
            if replaces:
                hit = self.find(replaces)
                if hit is None:
                    return {"error": f"no entry '{replaces}' to replace", "memory": self.listing()}
                store, i = hit
                old = self.entries(store)[i]
                if store != about:   # the fact moves to the other store
                    moved_from = (store, [e for j, e in enumerate(self.entries(store)) if j != i])
            if old is not None and old in entries:
                entries[entries.index(old)] = fact
            elif _norm(fact) in map(_norm, entries):
                return {"ok": True, "id": entry_id(fact), "note": "already remembered"}
            else:
                entries.append(fact)
            used, limit = self.usage(about, entries)
            if used > limit:
                return {"error": f"memory about '{about}' is full ({used}/{limit} chars). Replace or forget "
                                 "an older entry first.", "entries": [{"id": entry_id(e), "fact": e}
                                                                      for e in self.entries(about)]}
            self._write(about, entries)
            if moved_from:
                self._write(*moved_from)
        return {"ok": True, "id": entry_id(fact), "replaced": old, "usage": f"{used}/{limit} chars"}

    def forget(self, key):
        with self.lock:
            hit = self.find(key)
            if hit is None:
                return {"error": f"no single entry matches '{key}'; use an id", "memory": self.listing()}
            store, i = hit
            entries = self.entries(store)
            gone = entries.pop(i)
            self._write(store, entries)
        return {"ok": True, "forgotten": gone}

    # ---------------------------------------------------------------- runs (automatic)
    def last_run_at(self):
        runs = self.runs()
        return max((r.get("at", 0) for r in runs), default=0.0)

    def log_run(self, result, dry):
        rec = {"at": result.get("at") or time.time(), "task": result.get("task"), "ended": result.get("ended"),
               "seconds": result.get("seconds"), "dry": bool(dry)}
        with self.lock, open(self.root / "runs.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def runs(self, dry=None):
        f = self.root / "runs.jsonl"
        if not f.exists():
            return []
        out = []
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if dry is None or bool(r.get("dry")) == bool(dry):
                out.append(r)
        return out

    # ---------------------------------------------------------------- the prompt block
    def snapshot(self, dry, now=None):
        now = time.localtime(now or time.time())
        lines = ["What you remember (your long-term memory; it is kept between conversations, and you "
                 "write it yourself with the remember and forget tools).",
                 f"Now: {time.strftime('%A %Y-%m-%d %H:%M', now)}."]
        for store, (_, limit, heading) in STORES.items():
            entries = self.entries(store)
            used, _ = self.usage(store, entries)
            lines.append(f"{heading} [{used}/{limit} chars]:")
            lines += [f"- ({entry_id(e)}) {e}" for e in entries] or ["- (nothing yet)"]
        runs = self.runs(dry)
        label = "Your runs in dry tests (no real arm; replayed camera images)" if dry else "Your runs on the real arm"
        if runs:
            lines.append(f"{label}, newest last:")
            for r in runs[-RUNS_IN_PROMPT:]:
                when = time.strftime("%a %d %b %H:%M", time.localtime(r["at"]))
                lines.append(f"- {when}: \"{r['task']}\", {r['seconds']} s: {ENDED.get(r['ended'], r['ended'])}")
            counts = {}
            for r in runs:
                counts[r["ended"]] = counts.get(r["ended"], 0) + 1
            since = time.strftime("%Y-%m-%d", time.localtime(runs[0]["at"]))
            lines.append(f"Since {since}: {len(runs)} run(s): " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
                         + ". Only a person can tell you whether a run really succeeded.")
        else:
            lines.append(f"{label}: none yet.")
        return "\n".join(lines)

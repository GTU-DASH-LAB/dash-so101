"""Episodic memory: a persistent JSONL log of every attempt and its outcome.

Gives the system Reflexion-style learning without weight updates: failed attempts
(and the VLM's postmortem of *why* they failed) are recalled and injected into the
planner prompt on later attempts — within a session and across sessions.

Self-test: python so_brain/memory.py
"""

import json
import re
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "episodic_memory.jsonl"
STOPWORDS = {"a", "an", "the", "on", "in", "into", "to", "of", "put", "place", "move", "pick", "up"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}


class Memory:
    def __init__(self, path: "Path | str" = DEFAULT_PATH):
        self.path = Path(path)

    def log(self, **entry) -> None:
        entry.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def recall(self, phrase: str = "", k: int = 3) -> list[dict]:
        """Most recent non-successful attempts, preferring ones sharing words with `phrase`."""
        failures = [e for e in self._entries() if e.get("outcome") != "success"]
        query = _words(phrase)
        matching = [
            e
            for e in failures
            if query & _words(str(e.get("object", "")) + " " + str(e.get("destination", "")))
        ]
        return (matching or failures)[-k:]

    def lessons(self, phrase: str = "", k: int = 3) -> str:
        """Formatted for the planner prompt; empty string when there is nothing to learn."""
        lines = [
            f"- {e['ts']}: tried to pick {e.get('object')!r} -> place on {e.get('destination')!r}; "
            f"outcome: {e.get('outcome')}. {e.get('note', '')}".strip()
            for e in self.recall(phrase, k)
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
        m = Memory(f.name)
        m.log(object="a pen", destination="a cup", outcome="fail", note="grasped too close to the tip")
        m.log(object="an eraser", destination="a box", outcome="success")
        m.log(object="a pen", destination="a cup", outcome="fail", note="pen rolled away on contact")
        assert len(m.recall("pen")) == 2
        assert m.recall("never seen")  # falls back to any failures
        assert "rolled away" in m.lessons("pen")
        assert all(e["outcome"] != "success" for e in m.recall())
    print("memory self-test OK")

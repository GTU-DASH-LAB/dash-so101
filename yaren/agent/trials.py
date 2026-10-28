"""Test mode: the pen task again and again, scored - the "20 trials, 14 successes" of Fouad's results.

    each trial: film it -> run -> look from above (the camera's guess) -> the user's verdict -> pen back -> ...

The user's verdict is the score. The camera check is kept next to it for every trial: it was right on 19/19
and 18/18 dataset frames, but has never been measured on views after real runs, and the test measures that
too. A run that ends blocked, overheated or in an error was no fair try of the policy: it does not count and
is repeated. Results go to run/tests/*.json. The test outlives a reconnect of the browser tab that started it
(the new Session takes it over), and STOP ends it.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from .film import CLIPS, relabel

log = logging.getLogger("yaren.trials")
TESTS = Path(__file__).resolve().parent.parent / "run" / "tests"
MAX_TRIALS = 30


def _norm(words):
    return {w for x in words for w in re.findall(r"[^\W_]+", x.lower().replace("i̇", "i").replace("ı", "i"))}


# Short spoken answers that settle what the test waits for without the LLM; anything else goes to the LLM,
# which answers through the test tool. "tamam" (okay) is no verdict, but it is "ready". Her own questions
# never contain these words: an answer given while she still talks is checked for being her echo, and
# "oldu" right after "Sence oldu mu?" would be thrown away as one.
YES = _norm(["evet", "oldu", "olmuş", "başarılı", "yes", "yep", "yeah", "yup", "worked", "success", "successful"])
NO = _norm(["hayır", "olmadı", "olmamış", "başarısız", "no", "nope", "failed", "fail", "didn"])
READY = _norm(["hazır", "hazırım", "devam", "başla", "tamam", "tamamdır", "bitti", "ready", "go", "next",
               "continue", "okay", "ok", "done"])


def answer(text, kind):
    """A short, clear answer: verdict -> True / False, ready -> True; None when it is not that clear."""
    w = [x for x in re.findall(r"[^\W_]+", text.lower().replace("i̇", "i").replace("ı", "i"))]
    if not w or len(w) > 5:
        return None
    if kind == "verdict":
        yes, no = any(x in YES for x in w), any(x in NO for x in w)
        return True if yes and not no else False if no and not yes else None
    return True if kind == "ready" and any(x in READY for x in w) else None


GUESS = {   # what the camera check saw: for the LLM's note and her question (English, Turkish)
    "success": ("the pen looks like it is in the bowl", "bana kalem kasede gibi görünüyor"),
    "fail": ("the pen still looks like it is on the table", "bana kalem hâlâ masada gibi görünüyor"),
    "unsure": ("no pen was in view, on the table or in the bowl", "kalemi göremedim"),
    "unknown": ("the camera check did not work", "yukarıdan bakamadım"),
}
VERDICT_FILM_S = 12.0   # the clip keeps rolling this long after the question, for the answer; then it ends


class TestRun:
    def __init__(self, session, n, instruction, dry=False):
        self.session = session          # rebound when the browser tab reconnects
        self.n, self.instr, self.dry = n, instruction, dry
        self.results = []
        self.phase, self.trial = "starting", 1
        self.camera = None              # the camera's guess for the trial being asked about
        self.waiting = None             # (kind, future) while waiting for the user: "verdict" | "ready"
        self.started = time.time()
        self.ended = None
        self.file = None
        self.task = asyncio.create_task(self.run())

    @property
    def running(self):
        return not self.task.done()

    def summary(self):
        both = [r for r in self.results if r["camera"] in ("success", "fail") and r["user"] is not None]
        return {"done": len(self.results), "successes": sum(r["success"] for r in self.results),
                "camera_agreed": sum((r["camera"] == "success") == r["user"] for r in both),
                "camera_compared": len(both)}

    def state(self):
        return {"n": self.n, "trial": self.trial, "phase": self.phase, "camera": self.camera,
                "waiting": self.waiting[0] if self.waiting else None, "ended": self.ended, "file": self.file,
                "dry": self.dry, **self.summary(),
                "results": [{k: r.get(k) for k in ("trial", "success", "camera", "ended", "seconds", "clip")}
                            for r in self.results]}

    def send(self):
        self.session.send(type="test", **self.state())

    def say(self, en, tr):
        s = self.session
        s.say(tr if s.lang == "tr" else en)

    def give(self, kind, value=True):
        """The user's answer (a button, the spoken fast path or the LLM's test tool). False when nothing
        waits for that kind of answer now."""
        if not self.waiting or self.waiting[0] != kind or self.waiting[1].done():
            return False
        self.waiting[1].set_result(value)
        return True

    async def wait_for(self, kind):
        fut = asyncio.get_running_loop().create_future()
        self.waiting, self.phase = (kind, fut), kind
        self.send()
        try:
            return await fut
        finally:
            self.waiting = None

    # ---------------------------------------------------------------- the loop
    async def run(self):
        log.info("test: %d trials of %r", self.n, self.instr)
        try:
            while len(self.results) < self.n and not self.ended:
                self.trial = len(self.results) + 1
                await self.one_trial()
            self.ended = self.ended or "finished"
        except asyncio.CancelledError:
            self.ended = self.ended or "stopped"
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("test failed")
            self.ended = f"error: {type(e).__name__}: {e}"
        finally:
            self.phase, self.camera, self.waiting = "done", None, None
            self.save()
            asyncio.get_running_loop().create_task(self.wrap_up())   # not in this (maybe cancelled) task

    async def one_trial(self):
        s, i = self.session, self.trial
        body, cfg = s.hub.body, s.hub.cfg
        self.phase, self.camera = "running", None
        self.send()
        if i > 1 or self.results:
            self.say(f"Trial {i}.", f"{i}. deneme.")
        filming = s.settings.get("filmRuns", True) and await s.film_start("trial")
        r = await body.task(self.instr, cfg["task_timeout"])
        if r.get("error"):
            ended, secs = "error", None
            why = r["error"]
        else:
            ended = await s.follow_run(self.instr)
            secs = ((await body.status()).get("last_result") or {}).get("seconds")
            why = ended
        s = self.session
        meta = {"test_trial": i, "of": self.n, "instruction": self.instr, "run_ended": ended, "run_seconds": secs}
        if ended in ("stopped", "cancelled", "replaced"):      # STOP (or another run): the test is over
            if filming:
                await s.film_stop("trial", f"{i:02d}_stopped", meta)
            self.ended = "stopped"
            return
        if ended not in ("done", "timeout"):                   # blocked / overheat / error: no fair try
            if filming:
                await s.film_stop("trial", f"{i:02d}_{ended}", meta)
            s.body_note(f"Test trial {i} did not finish ({why}); it does not count and will be repeated. You asked "
                        "the user to clear the way, put the pen back and say ready: then call test with action ready.")
            self.say(f"Trial {i} didn't finish, so it doesn't count. Put the pen back, then tell me and we'll redo it.",
                     f"{i}. deneme yarıda kaldı, sayılmayacak. Kalemi yerine koyunca bana söyle, tekrar deneriz.")
            await self.wait_for("ready")
            return
        self.phase = "checking"
        self.send()
        cam = (await s.check_pen())["verdict"]                # looks from above, then goes home
        self.camera = cam
        s = self.session
        en, tr = GUESS.get(cam, GUESS["unknown"])
        s.body_note(f"Test trial {i} of {self.n} ended ({ended}); from above: {en}. You asked the user whether it "
                    "worked: when they answer, call test with action verdict.")
        self.say(f"Trial {i} is done: {en}. Did it work?", f"{i}. deneme bitti, {tr}. Sen ne dersin?")
        verdict = asyncio.ensure_future(self.wait_for("verdict"))
        clip = None
        if filming:      # the answer belongs in the clip, an empty table for minutes does not
            await asyncio.wait({verdict}, timeout=VERDICT_FILM_S)
            if not verdict.done():
                clip = await self.session.film_stop("trial", f"{i:02d}_pending", {**meta, "camera": cam}, announce=False)
                filming = False
        ok = bool(await verdict)
        s = self.session
        label = f"{i:02d}_{'success' if ok else 'fail'}"
        res = {"trial": i, "ended": ended, "seconds": secs, "camera": cam, "user": ok, "success": ok,
               "at": time.time()}
        self.results.append(res)
        if filming:
            res["clip"] = await s.film_stop("trial", label, {**meta, "success": ok, "camera": cam})
        elif clip:
            res["clip"] = relabel(clip, "success" if ok else "fail", CLIPS / "dry" if self.dry else CLIPS)
            s.send(type="clip", name=res["clip"], kind="trial", label=label, seconds=None)
        k = sum(r["success"] for r in self.results)
        log.info("test trial %d: %s (camera %s), %d/%d", i, "success" if ok else "fail", cam, k, len(self.results))
        self.send()
        if len(self.results) < self.n:
            s.body_note(f"Test trial {i}: {'it worked' if ok else 'it did not work'} (so far {k} of {i}). You asked "
                        "the user to put the pen back on the table and say ready: then call test with action ready.")
            self.say("Got it. Put the pen back on the table, then tell me.",
                     "Yazdım. Kalemi masaya geri koy, bitince bana söyle.")
            await self.wait_for("ready")

    def save(self):
        if not self.results and not self.ended:
            return
        TESTS.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.started)) + ("_dry" if self.dry else "") + ".json"
        data = {"instruction": self.instr, "trials_planned": self.n, "dry": self.dry, "ended": self.ended,
                "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)),
                **self.summary(), "results": self.results}
        (TESTS / name).write_text(json.dumps(data, indent=1, ensure_ascii=False))
        self.file = name
        log.info("test saved: %s (%s)", name, self.ended)

    async def wrap_up(self):
        s = self.session
        if s.film is not None and s.film["kind"] == "trial":     # stopped in the middle of a trial
            await s.film_stop("trial", f"{self.trial:02d}_stopped", {"test_trial": self.trial, "of": self.n})
        sm = self.summary()
        self.send()
        d, k = sm["done"], sm["successes"]
        how = "finished" if self.ended == "finished" else "was stopped" if self.ended == "stopped" else f"ended ({self.ended})"
        score = f"{k} of {d} trials worked ({round(100 * k / d)}%)" if d else "no trial was scored"
        if 0 < d <= 12:   # each one, so the summary never has to guess
            score += " - " + ", ".join(f"trial {r['trial']}: {'worked' if r['success'] else 'did not work'}"
                                       for r in self.results)
        cam = (f" Your camera check agreed with the user's verdict on {sm['camera_agreed']} of {sm['camera_compared']}."
               if sm["camera_compared"] else "")
        await s.report(f"[Robot body event] The test {how}: {score}.{cam} The results are saved and the test is "
                       "over (no more test calls). Tell the user the score in one or two short sentences.")

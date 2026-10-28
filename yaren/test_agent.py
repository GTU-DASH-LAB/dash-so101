#!/usr/bin/env python3
"""Offline checks of the voice agent's pure logic: no models, no robot.
    ~/envs/yaren/bin/python test_agent.py   (or pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.brain import (EXPRESS_RULES, TAG, health_report, posture_step, is_echo, is_parrot,  # noqa: E402
                         is_stop_command, parse_move, parse_step)
from agent.film import SR, Soundtrack  # noqa: E402
from agent.trials import answer  # noqa: E402

SAID = ["Selaaam kanki, hoş geldin! Bugün nasılsın?", "Tamam, kalemi kaseye koyuyorum.",
        "I wasn't trained for that, but I'll give it a shot!"]


def test_echo_of_her_own_words():
    assert is_echo("Selam kanki hoş geldin bugün nasılsın", SAID)
    assert is_echo("kalemi kaseye koyuyorum", SAID)
    assert is_echo("I'll give it a shot", SAID)
    assert is_echo("Bugün nasılsın?", SAID)
    assert is_echo("kalem kaseye", SAID)                 # a suffix Whisper dropped


def test_real_speech_gets_through():
    assert not is_echo("Hayır, başka bir şey yap", SAID)
    assert not is_echo("nasılsın bugün? ben iyiyim teşekkürler", SAID)   # half new: someone is talking
    assert not is_echo("Can you dance for me?", SAID)
    assert not is_echo("Merhaba", [])                    # she said nothing recently
    assert not is_echo("", SAID)


def test_stop_words_are_not_echo_checked():
    # on_utterance checks is_stop_command first: "dur" must always reach the robot
    assert is_stop_command("dur")
    assert is_stop_command("Yeter, dur!")
    assert not is_stop_command("dur bakalım kanki şimdi şunu bir anlat bana")



def test_copied_prompt_tags_are_dropped():
    # replies seen from qwen3.8 on 2026-10-28
    assert TAG.sub("", "[ear] Motorların nasıl, ısındın mı?") == "Motorların nasıl, ısındın mı?"
    assert TAG.sub("", "[tool] [tool use][tool] [tool use]").strip() == ""
    assert TAG.sub("", "[ear]\nTamam kanki, hazırım!") == "Tamam kanki, hazırım!"
    assert TAG.sub("", "Tamam kanki, başlıyorum!") == "Tamam kanki, başlıyorum!"


def test_parroting_the_user():
    assert is_parrot("Motorların nasıl, ısındın mı?", "Motorların nasıl, ısındın mı?")
    assert not is_parrot("Motorlarım gayet iyi kanki.", "Motorların nasıl, ısındın mı?")
    assert not is_parrot("Tamam!", "Tamam, dans et")          # too short to judge


def test_plan_steps():
    assert parse_step("task: Pick up the pen and put it in the bowl") == ("task", "Pick up the pen and put it in the bowl")
    assert parse_step("Gesture: dance") == ("gesture", "dance")
    assert parse_step("home") == ("home", "")
    assert parse_step("wait: 2") == ("wait", "2")
    assert parse_step("pose: survey") == ("pose", "survey")
    assert parse_step("check: Is there a pen on the table?") == ("check", "Is there a pen on the table?")
    for bad in ("fly: high", "gesture: backflip", "pose: yoga", "move: knee up", "wait: soon", "task:", "say"):
        try:
            parse_step(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_move_steps():
    assert parse_move("elbow up 20") == ("elbow", "up", 20.0)
    assert parse_move("wrist roll left 30 degrees") == ("wrist_roll", "left", 30.0)
    assert parse_move("gripper open") == ("gripper", "open", None)
    assert parse_move("hand close") == ("gripper", "close", None)


def express(text):
    return next((k for k, rx in EXPRESS_RULES if rx.search(text.strip())), None)


def test_body_language_rules():
    assert express("Evet, tabii ki yapabilirim!") == "nod"
    assert express("Sure, let's do it.") == "nod"
    assert express("Hayır, onu yapamam.") == "shake"
    assert express("Haha, çok komiksin kanki.") == "wiggle"
    assert express("Üzgünüm, bunu bilmiyorum.") == "droop"
    assert express("Harika bir fikir bu!") == "bounce"
    assert express("Bugün nasılsın?") == "tilt"
    assert express("Ben Yaren, bir robot kolum.") is None
    assert express("Bunu yokladım.") is None            # "yok" only as a word at the start


def test_posture():
    rest = {"state": "idle", "at_home": True, "near_pose": None}
    standing = {"state": "idle", "at_home": False, "near_pose": "portrait"}
    assert posture_step(rest, 2, wake=True, up=False, allowed=True) == "up"
    assert posture_step(rest, 2, wake=True, up=False, allowed=False) is None      # just after go home / a task
    assert posture_step(rest, 12, wake=True, up=False, allowed=True) is None      # that was a while ago
    assert posture_step({**rest, "state": "moving"}, 2, wake=True, up=False, allowed=True) is None
    assert posture_step({**rest, "at_home": False}, 2, wake=True, up=False, allowed=True) is None
    assert posture_step(standing, 61, wake=False, up=True, allowed=True) == "down"
    assert posture_step(standing, 30, wake=False, up=True, allowed=True) is None
    assert posture_step({**standing, "near_pose": None}, 61, wake=False, up=True, allowed=True) is None  # moved since


def test_health_report():
    assert "error" in health_report({"dry": True})
    h = {"hot": False, "warm": True, "hottest": "elbow_flex", "max_temp_c": 58, "limit_c": 70,
         "temp_c": {"elbow_flex": 58, "gripper": 40}, "min_volt_v": 11.9}
    r = health_report({"health": h})
    assert r["verdict"] == "warm" and r["hottest_motor"] == "elbow_flex" and r["cuts_out_at_c"] == 70


def test_test_mode_answers():
    for yes in ("Evet", "oldu", "Oldu oldu!", "başarılı", "Yes", "it worked", "yeah"):
        assert answer(yes, "verdict") is True, yes
    for no in ("Hayır", "olmadı", "Olmamış", "başarısız", "No", "nope", "it failed", "it didn't work"):
        assert answer(no, "verdict") is False, no
    for unclear in ("evet ama hayır", "bence kalem biraz dışarıda kaldı ama yine de sayalım", "tamam", "", "hmm"):
        assert answer(unclear, "verdict") is None, unclear
    for ready in ("hazır", "Hazırım", "tamam", "devam", "ready", "okay go", "bitti"):
        assert answer(ready, "ready") is True, ready
    for other in ("bir dakika", "neden düştü kalem", "evet"):
        assert answer(other, "ready") is None, other
    # her own test questions never contain the answers they wait for (an answer said over her is
    # echo-checked: "oldu" right after "Sence oldu mu?" would be dropped as her echo)
    asks = {"verdict": ("3. deneme bitti, bana kalem kasede gibi görünüyor. Sen ne dersin?",
                        "Trial 3 is done: the pen looks like it is in the bowl. Did it work?"),
            "ready": ("Yazdım. Kalemi masaya geri koy, bitince bana söyle.",
                      "3. deneme yarıda kaldı, sayılmayacak. Kalemi yerine koyunca bana söyle, tekrar deneriz.",
                      "Got it. Put the pen back on the table, then tell me.",
                      "Trial 3 didn't finish, so it doesn't count. Put the pen back, then tell me and we'll redo it.")}
    words = {"verdict": ("oldu", "olmadı", "evet", "hayır", "yes", "no", "worked", "failed"),
             "ready": ("hazır", "tamam", "devam", "bitti", "ready", "done", "okay", "go")}
    for kind, lines in asks.items():
        for line in lines:
            for word in words[kind]:
                assert not is_echo(word, [line]), (word, line)


def test_soundtrack_timing():
    import numpy as np
    tr = Soundtrack()
    t0 = 1000.0
    one = np.ones(SR, np.float32) * 0.5               # 1 s of "speech" at the soundtrack's rate
    tr.add_speech(one, SR, t0 + 1.0)                  # plays from ~1.03 s
    tr.add_speech(one, SR, t0 + 1.2)                  # queued: right after the first, ~2.03 s
    out = tr.render(t0, 4.0)
    nz = np.flatnonzero(out)
    assert abs(nz[0] / SR - 1.03) < 0.01 and abs(nz[-1] / SR - 3.03) < 0.01
    tr.cut(t0 + 2.5)                                  # barge-in: the second sentence stops there
    out = tr.render(t0, 4.0)
    assert abs(np.flatnonzero(out)[-1] / SR - 2.5) < 0.01
    # the microphone: frames arrive with jitter but play back to back; a long silence (mic off) is kept
    tr = Soundtrack()
    frame = (np.ones(512, np.int16) * 8000).tobytes()
    at = t0 + 0.032
    for k in range(10):
        tr.add_mic(frame, at + (0.01 if k % 2 else 0.0))
        at += 0.032
    tr.add_mic(frame, t0 + 3.0)
    out = tr.render(t0, 4.0)
    nz = np.flatnonzero(out)
    assert nz[0] / SR < 0.002 and abs(np.flatnonzero(out[:SR * 2])[-1] / SR - 0.32) < 0.01
    assert abs(nz[-1] / SR - 3.0) < 0.01 and out.max() <= 0.99


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

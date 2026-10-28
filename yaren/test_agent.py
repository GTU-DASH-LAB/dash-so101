#!/usr/bin/env python3
"""Offline checks of the voice agent's pure logic: no models, no robot.
    ~/envs/yaren/bin/python test_agent.py   (or pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.brain import (EXPRESS_RULES, TAG, health_report, posture_step, is_echo, is_parrot,  # noqa: E402
                         is_stop_command, parse_move, parse_step)

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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

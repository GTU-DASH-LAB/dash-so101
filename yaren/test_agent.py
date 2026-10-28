#!/usr/bin/env python3
"""Offline checks of the voice agent's pure logic: no models, no robot.
    ~/envs/yaren/bin/python test_agent.py   (or pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.brain import is_echo, is_stop_command  # noqa: E402

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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

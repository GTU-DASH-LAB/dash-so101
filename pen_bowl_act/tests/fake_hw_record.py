#!/usr/bin/env python3
"""Run record_session.py against FAKE arms + a synthetic camera, with scripted key
presses and injected failures. Everything below the robot (dataset writer, PNG
writer, h264 encoding, finalize, resume) is the real lerobot code path.

    python tests/fake_hw_record.py <scenario> <session_dir> <episodes>

Scenarios: normal, rerecord, short, crash_record, crash_reset, esc_record,
           esc_reset, ctrl_c, hard_kill
Used by tests/run_recording_tests.sh; no robot or camera is touched.
"""

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

scenario, session_dir, episodes = sys.argv[1], sys.argv[2], int(sys.argv[3])
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FAIL = {"now": None}  # exception instance to raise from the next hardware call


def hw_call():
    if FAIL["now"] is not None:
        e, FAIL["now"] = FAIL["now"], None
        raise e


class FakeCamera:
    is_connected = True

    def disconnect(self):
        self.is_connected = False


class FakeRobot:
    name = "so_follower"
    robot_type = "so_follower"

    def __init__(self):
        self.cameras = {"wrist": FakeCamera()}
        self._connected = False
        self.t = 0
        self.sent = []

    @property
    def action_features(self):
        return {f"{m}.pos": float for m in MOTORS}

    @property
    def observation_features(self):
        return {**{f"{m}.pos": float for m in MOTORS}, "wrist": (480, 640, 3)}

    @property
    def is_connected(self):
        return self._connected

    def connect(self, calibrate=True):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def get_observation(self):
        hw_call()
        self.t += 1
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        x = (self.t * 7) % 640
        img[:, max(0, x - 20):x + 20] = (200, 80, 40)  # a moving bar: cheap to encode
        img[:40, :, 1] = (self.t * 3) % 255
        return {**{f"{m}.pos": float(np.sin(self.t / 30 + i)) for i, m in enumerate(MOTORS)}, "wrist": img}

    def send_action(self, action):
        hw_call()
        self.sent.append(action)
        return action


from lerobot.teleoperators import Teleoperator  # noqa: E402


class FakeLeader(Teleoperator):
    name = "so_leader"
    config_class = None

    def __init__(self):  # skip the base __init__ (it wants a config + calibration files)
        self._connected = False
        self.t = 0

    action_features = property(lambda self: {f"{m}.pos": float for m in MOTORS})
    feedback_features = property(lambda self: {})
    is_connected = property(lambda self: self._connected)
    is_calibrated = property(lambda self: True)

    def connect(self, calibrate=True):
        self._connected = True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_action(self):
        hw_call()
        self.t += 1
        return {f"{m}.pos": float(np.cos(self.t / 30 + i)) for i, m in enumerate(MOTORS)}

    def send_feedback(self, feedback):
        pass

    def disconnect(self):
        self._connected = False

    def __del__(self):
        pass


import lerobot.robots as robots_mod  # noqa: E402
import lerobot.teleoperators as teleops_mod  # noqa: E402
import lerobot.utils.keyboard_input as kb  # noqa: E402

robots_mod.make_robot_from_config = lambda cfg: FakeRobot()
teleops_mod.make_teleoperator_from_config = lambda cfg: FakeLeader()

EVENTS = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


class FakeListener:
    def stop(self):
        pass


kb.init_keyboard_listener = lambda: (FakeListener(), EVENTS)

import record_session  # noqa: E402

PHASES: list[tuple[float, str]] = []
_orig_banner = record_session.banner


def spy_banner(color, text):
    PHASES.append((time.time(), text.split()[0]))
    _orig_banner(color, text)


record_session.banner = spy_banner


def wait_phase(kind: str, nth: int, timeout: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if sum(1 for _, p in PHASES if p == kind) >= nth:
            return
        time.sleep(0.02)
    raise TimeoutError(f"phase {kind} #{nth} never came")


def key(name: str) -> None:
    from lerobot.utils.keyboard_input import apply_recording_control

    apply_recording_control(name, EVENTS)


def director():
    """Scripted operator: decides when to press keys / pull cables."""
    if scenario == "rerecord":
        wait_phase("RECORDING", 2); time.sleep(1.0); key("left")
    elif scenario == "short":
        wait_phase("RECORDING", 2); time.sleep(0.3); key("right")
    elif scenario == "crash_record":
        wait_phase("RECORDING", 2); time.sleep(1.0); FAIL["now"] = ConnectionError("Failed to sync read 'Present_Position' ... There is no status packet!")
    elif scenario == "crash_reset":
        wait_phase("RESET", 2); time.sleep(0.4); FAIL["now"] = TimeoutError("Timed out waiting for frame from OpenCVCamera")
    elif scenario == "esc_record":
        wait_phase("RECORDING", 2); time.sleep(1.0); key("esc")
    elif scenario == "esc_reset":
        wait_phase("RESET", 2); time.sleep(0.3); key("esc")
    elif scenario == "ctrl_c":
        wait_phase("RECORDING", 2); time.sleep(1.0); FAIL["now"] = KeyboardInterrupt()
    elif scenario == "hard_kill":
        wait_phase("RECORDING", 2); time.sleep(1.0); os._exit(137)  # like kill -9: no finally, no finalize


threading.Thread(target=director, daemon=True).start()

sys.argv = [
    "record_session.py", "--session-dir", session_dir, "--repo-id", "test/fake_session",
    "--episodes", str(episodes), "--task", "Pick up the pen and put it in the bowl",
    "--fps", "30", "--episode-time-s", "2", "--reset-time-s", "1", "--min-episode-s", "1",
    "--follower-port", "/dev/null", "--follower-id", "x", "--leader-port", "/dev/null", "--leader-id", "x",
    "--cameras", '{"wrist": {"type": "opencv", "index_or_path": "/dev/null", "width": 640, "height": 480, "fps": 30}}',
    "--vcodec", "h264", "--crf", "23",
]
sys.exit(record_session.main())

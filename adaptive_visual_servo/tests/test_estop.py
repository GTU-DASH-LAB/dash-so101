"""E-Stop contract: once stop_event is set, every motor-facing rig call
raises EStopped BEFORE any bus IO, so a running episode releases the serial
port for the E-Stop thread's torque-off writes (which otherwise fail with
'Port is in use!' -- observed live)."""

import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_real import EStopped, SO101Rig


def _stopped_rig():
    rig = object.__new__(SO101Rig)  # no hardware: __init__ skipped on purpose
    rig.stop_event = threading.Event()
    rig.stop_event.set()
    rig._g = 0.0  # gripper_contact's early-out is for open grippers
    return rig


def test_all_motor_calls_raise_before_bus_io():
    rig = _stopped_rig()
    # no .robot/.cfg attached: if any call reached bus IO instead of raising
    # EStopped first, it would die with AttributeError and fail the test
    q = np.zeros(5)
    for call in (rig.get_q, lambda: rig.set_q(q), lambda: rig.set_q_raw(q),
                 lambda: rig.set_gripper(1.0), rig.gripper_contact):
        with pytest.raises(EStopped):
            call()


def test_estop_not_a_runtime_error():
    """control.py catches RuntimeError for recoverable failures (babble);
    EStopped must NOT be swallowed by those handlers."""
    assert not issubclass(EStopped, RuntimeError)


if __name__ == "__main__":
    test_all_motor_calls_raise_before_bus_io()
    test_estop_not_a_runtime_error()
    print("ALL OK")

"""Null-space secondary objectives: dq = J+(lam*e) + (I - J+J) dq_null."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RealConfig, ServoConfig
from control import damped_pinv, make_null_fn

RC = RealConfig()
Q_LO = tuple(np.radians(RC.q_min_deg))
Q_HI = tuple(np.radians(RC.q_max_deg))


def test_projection_preserves_image_task():
    """(I - J+J) dq_null must produce ~zero first-order pixel motion for any
    dq_null, so the secondary objectives can never fight the servo."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        J = rng.normal(0, 2.0, (2, 5))
        Jp = damped_pinv(J, 1e-4)  # tiny damping: projector near-exact
        N = np.eye(5) - Jp @ J
        v = rng.normal(0, 1.0, 5)
        leak = np.linalg.norm(J @ (N @ v))
        direct = np.linalg.norm(J @ v) + 1e-12
        assert leak < 0.02 * direct, f"null-space leak {leak:.4f} vs direct {direct:.2f}"


def test_limit_repulsion_direction_and_deadzone():
    scfg = ServoConfig(q_lo=Q_LO, q_hi=Q_HI, null_k_vis=0.0)
    fn = make_null_fn(scfg, {"q": None})
    lo, hi = np.asarray(Q_LO), np.asarray(Q_HI)
    mid = 0.5 * (lo + hi)
    # mid-range: exactly zero (deadzone -- never bias normal motion)
    assert np.allclose(fn(mid.copy()), 0.0)
    # joint 1 at 95% toward its upper limit: pushed back DOWN, others untouched
    q = mid.copy()
    q[1] = mid[1] + 0.95 * (hi[1] - mid[1])
    dq = fn(q)
    assert dq[1] < 0, f"expected repulsion away from q_hi, got {dq[1]:+.4f}"
    assert np.allclose(np.delete(dq, 1), 0.0)
    # same joint at 95% toward its lower limit: pushed back UP
    q[1] = mid[1] - 0.95 * (mid[1] - lo[1])
    assert fn(q)[1] > 0
    # repulsion grows toward the limit, capped by k_lim at the limit itself
    q[1] = hi[1]
    assert abs(fn(q)[1]) <= scfg.null_k_lim + 1e-12


def test_wrist_posture_attraction():
    """With a marker-visible reference set, only the wrist joints are pulled
    toward it, proportionally to their error."""
    scfg = ServoConfig(q_lo=None, null_k_lim=0.0)  # isolate the visibility term
    q_ref = {"q": None}
    fn = make_null_fn(scfg, q_ref)
    q = np.zeros(5)
    assert np.allclose(fn(q), 0.0)  # no reference yet -> inert
    q_ref["q"] = np.array([0.3, 0.1, -0.2, 0.4, -0.5])
    dq = fn(q)
    assert dq[3] > 0 and dq[4] < 0, "wrist pulled toward the visible pose"
    assert np.allclose(dq[:3], 0.0), "non-wrist joints must not be touched"
    assert np.isclose(dq[3], scfg.null_k_vis * 0.4)
    # short arm (toy 3-DOF sim): wrist indices out of range -> term skipped
    assert np.allclose(fn(np.zeros(3)), 0.0)


def test_disabled_returns_none():
    scfg = ServoConfig(q_lo=None, null_k_lim=0.0, null_k_vis=0.0)
    assert make_null_fn(scfg, {"q": None}) is None


if __name__ == "__main__":
    for fn in [test_projection_preserves_image_task,
               test_limit_repulsion_direction_and_deadzone,
               test_wrist_posture_attraction, test_disabled_returns_none]:
        fn()
        print(f"ok {fn.__name__}")
    print("ALL OK")

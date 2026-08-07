"""Babbled Jacobian vs. ground truth (finite differences of the true
render-free projection pipeline) — the sim equivalent of checking against
lerobot-kinematics' analytic Jacobian."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ServoConfig
from control import babble, damped_pinv
from sim import SimWorld

SCFG = ServoConfig()


def true_jacobian(world, q, eps=1e-5):
    J = np.zeros((2, 3))
    for i in range(3):
        d = np.zeros(3); d[i] = eps
        a = world.cam.project(world.grip_center(q + d))
        b = world.cam.project(world.grip_center(q - d))
        J[:, i] = (a - b) / (2 * eps)
    return J


def check_pose(seed, dq_pose):
    w = SimWorld(seed=seed)
    if dq_pose is not None:
        w.set_q(np.array(w.cfg.home_q) + dq_pose)
    rng = np.random.default_rng(seed)
    J, s = babble(w, SCFG, rng)
    q = w.get_q()
    J_gt = true_jacobian(w, q)
    # blink estimate consistent with ground truth EE pixel
    assert np.linalg.norm(s - w.grip_px()) < 8
    # direction + magnitude agreement on random probe directions
    rng2 = np.random.default_rng(seed + 100)
    for _ in range(10):
        dq = rng2.uniform(-1, 1, 3) * 0.05
        pred, act = J @ dq, J_gt @ dq
        if np.linalg.norm(act) < 2:  # nearly-null direction: angle meaningless
            continue
        cos = pred @ act / (np.linalg.norm(pred) * np.linalg.norm(act) + 1e-9)
        assert cos > 0.9, f"seed {seed}: prediction {np.degrees(np.arccos(cos)):.0f}deg off"
        ratio = np.linalg.norm(pred) / np.linalg.norm(act)
        assert 0.6 < ratio < 1.5, f"seed {seed}: magnitude ratio {ratio:.2f}"
    rel = np.linalg.norm(J - J_gt) / np.linalg.norm(J_gt)
    print(f"  seed {seed}: relative Frobenius error {rel:.2%}")
    assert rel < 0.35


def test_babble_at_home():
    check_pose(20, None)


def test_babble_other_poses():
    check_pose(21, np.array([0.6, -0.15, 0.25]))
    check_pose(22, np.array([1.0, 0.1, 0.0]))


def test_damped_pinv_sane():
    J = np.array([[100.0, 50.0, 20.0], [0.0, 80.0, 60.0]])
    Ji = damped_pinv(J, 1e-3)
    e = np.array([10.0, -5.0])
    ds = J @ (Ji @ e)
    assert np.linalg.norm(ds - e) < 0.5, "low damping should nearly reproduce e"
    # singular J must not blow up
    Js = np.array([[100.0, 100.0, 0.0], [100.0, 100.0, 0.0]])
    dq = damped_pinv(Js, 1e-2) @ e
    assert np.all(np.isfinite(dq)) and np.linalg.norm(dq) < 1.0


if __name__ == "__main__":
    for fn in [test_babble_at_home, test_babble_other_poses, test_damped_pinv_sane]:
        fn()
        print(f"ok {fn.__name__}")
    print("ALL OK")

"""Sim world sanity: rendering, spawning, kinematics, grasp physics, blink."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SimConfig
from sim import SimWorld, solve_ik_true


def test_render_and_spawn():
    w = SimWorld(seed=1)
    w.spawn_random(3)
    img = w.read()
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    # deterministic background, noisy reads
    a, b = w.render(noise=False), w.render(noise=False)
    assert np.array_equal(a, b)
    assert not np.array_equal(w.read(), w.read())
    # objects + pad + home EE inside the frame with margin
    for o in w.objects:
        u, v = w.cam.project(o.pos)
        assert 20 < u < 620 and 20 < v < 460, f"object off-frame: {u},{v}"
    pu, pv = w.cam.project(w.pad)
    assert 20 < pu < 620 and 20 < pv < 460
    gu, gv = w.grip_px()
    assert 0 < gu < 640 and 0 < gv < 480, f"home EE off-frame: {gu},{gv}"


def test_kinematics_and_clamp():
    w = SimWorld(seed=2)
    p0 = w.grip_px()
    w.set_q(w.get_q() + np.array([0.15, 0.0, 0.0]))
    assert np.linalg.norm(w.grip_px() - p0) > 5, "pan step should move EE pixels"
    w.set_q(np.array([9.0, 9.0, 9.0]))
    assert np.all(w.get_q() <= np.array(w.cfg.q_max) + 1e-9)


def test_grasp_physics():
    w = SimWorld(seed=3)
    (o,) = w.spawn_random(1)
    tgt = np.array([o.pos[0], o.pos[1], w.cfg.grasp_ee_z])
    q = solve_ik_true(w, tgt)
    w.set_q(q)
    assert np.linalg.norm(w.ee() - tgt) < 0.005, "IK helper should reach the object"
    # far-away close must not grab
    w2 = SimWorld(seed=3)
    (o2,) = w2.spawn_random(1)
    w2.set_gripper(0.1)
    assert not w2.gripper_contact()
    # close at the object => attached, follows EE, releases on open
    w.set_gripper(0.1)
    assert w.gripper_contact(), "gripper at object should capture it"
    w.set_q(q + np.array([0.3, 0.1, 0.1]))
    ee = w.ee()
    assert np.hypot(ee[0] - o.pos[0], ee[1] - o.pos[1]) < 0.001, "carried object follows EE"
    w.set_gripper(1.0)
    assert not w.gripper_contact()
    assert abs(o.pos[2] - w.cfg.obj_height / 2) < 1e-9, "released object rests on table"


def test_blink_visible():
    w = SimWorld(seed=4)
    w.set_gripper(1.0)
    a = w.render(noise=False).astype(np.int16)
    w.set_gripper(0.45)
    b = w.render(noise=False).astype(np.int16)
    diff = np.abs(a - b).max(axis=2)
    ys, xs = np.nonzero(diff > 15)
    assert len(xs) > 20, "gripper blink must change pixels"
    cx, cy = xs.mean(), ys.mean()
    gu, gv = w.grip_px()
    assert np.hypot(cx - gu, cy - gv) < 25, f"blink centroid {cx},{cy} far from EE {gu},{gv}"


if __name__ == "__main__":
    for fn in [test_render_and_spawn, test_kinematics_and_clamp,
               test_grasp_physics, test_blink_visible]:
        fn()
        print(f"ok {fn.__name__}")
    # visual artifact for eyeballing (scratchpad-friendly path via env or arg)
    out = sys.argv[1] if len(sys.argv) > 1 else None
    if out:
        import cv2
        w = SimWorld(seed=1)
        w.spawn_random(3)
        cv2.imwrite(out, w.render(noise=True))
        print("wrote", out)
    print("ALL OK")

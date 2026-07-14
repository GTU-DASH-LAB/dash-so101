"""PyBulletWorld sanity: URDF loads, rig interface, real-contact grasp physics.

Mirrors test_sim.py's structure for the toy sim, but this backend is the real
one -- actual URDF kinematics, physics stepping, rendered camera."""

import os
import sys

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pb_sim import ARM_JOINTS, PAD_CENTER, PyBulletWorld


def solve_ik_true(world, target_xyz):
    """pybullet's own IK -- test-only setup helper, mirrors sim.py's
    solve_ik_true; never used by control.py."""
    q_full = p.calculateInverseKinematics(world.robot, world.ee_link, list(target_xyz),
                                          maxNumIterations=200, residualThreshold=1e-4)
    return np.array(q_full[:len(ARM_JOINTS)])


def test_render_and_spawn():
    w = PyBulletWorld(gui=False, seed=1)
    try:
        w.spawn_random(2)
        img = w.read()
        assert img.shape == (480, 640, 3) and img.dtype == np.uint8
        for o in w.objects:
            u, v = w._project(np.array([*w.object_xy(o), 0.02]))
            assert 0 < u < 640 and 0 < v < 480, f"object off-frame: {u},{v}"
        pu, pv = w._project(np.array([*PAD_CENTER, 0.0]))
        assert 0 < pu < 640 and 0 < pv < 480
    finally:
        w.close()


def test_kinematics_and_clamp():
    w = PyBulletWorld(gui=False, seed=2)
    try:
        p0 = w.grip_px()
        w.set_q(w.get_q() + np.array([0.3, 0.0, 0.0, 0.0, 0.0]))
        assert np.linalg.norm(w.grip_px() - p0) > 5, "pan step should move EE pixels"
        w.set_q(np.full(5, 99.0))
        assert np.all(w.get_q() <= w.q_max + 1e-6)
        w.set_q(np.full(5, -99.0))
        assert np.all(w.get_q() >= w.q_min - 1e-6)
    finally:
        w.close()


def test_grasp_physics():
    w = PyBulletWorld(gui=False, seed=3)
    try:
        (o,) = w.spawn_random(1)
        xy = w.object_xy(o)
        q = solve_ik_true(w, [xy[0], xy[1], 0.035])
        w.set_q(q)
        assert np.hypot(*(w.ee_world()[:2] - xy)) < 0.02, "IK helper should reach the object"
        w.set_gripper(0.0)
        assert w.gripper_contact(), "closing near the object should trigger a proximity grasp"
        # carried object follows the gripper through motion
        w.set_q(q + np.array([0.3, 0.1, -0.1, 0.0, 0.0]))
        new_xy = w.object_xy(o)
        assert np.hypot(*(w.ee_world()[:2] - new_xy)) < 0.02, "carried object should follow EE"
        w.set_gripper(1.0)
        assert not w.gripper_contact(), "opening should release the object"
    finally:
        w.close()


def test_grasp_requires_proximity():
    w = PyBulletWorld(gui=False, seed=4)
    try:
        w.spawn_random(1)
        w.set_gripper(0.0)  # close on empty air, far from any object
        assert not w.gripper_contact()
    finally:
        w.close()


if __name__ == "__main__":
    for fn in [test_render_and_spawn, test_kinematics_and_clamp,
               test_grasp_physics, test_grasp_requires_proximity]:
        fn()
        print(f"ok {fn.__name__}")
    print("ALL OK")

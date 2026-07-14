"""PlacoModel (lerobot's placo-based SO-101 IK/FK) validated against
pybullet's own URDF ground truth, and control.py's nominal_z_to (unchanged,
duck-typed) driven end-to-end with the real 5-DOF pb_sim backend."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control import nominal_z_to
from config import ServoConfig
from lerobot_ik import PlacoModel
from pb_sim import PyBulletWorld

SCFG = ServoConfig()


def test_fk_matches_pybullet_ground_truth():
    model = PlacoModel()
    w = PyBulletWorld(gui=False, seed=1)
    try:
        rng = np.random.default_rng(0)
        for _ in range(5):
            q = w.q_min + rng.uniform(0.2, 0.8, 5) * (w.q_max - w.q_min)
            w.set_q(q)
            err = np.linalg.norm(w.ee_world() - model.ee(w.get_q()))
            assert err < 0.01, f"placo FK vs pybullet ground truth off by {err:.4f}m"
    finally:
        w.close()


def test_nominal_z_to_converges_on_real_kinematics():
    model = PlacoModel()
    w = PyBulletWorld(gui=False, seed=2)
    try:
        z_target = 0.08
        ok = nominal_z_to(w, model, z_target, SCFG)
        assert ok, "nominal_z_to should converge"
        true_z = w.ee_world()[2]
        assert abs(true_z - z_target) < 0.02, \
            f"true EE z={true_z:.4f} far from target {z_target}"
    finally:
        w.close()


if __name__ == "__main__":
    for fn in [test_fk_matches_pybullet_ground_truth,
               test_nominal_z_to_converges_on_real_kinematics]:
        fn()
        print(f"ok {fn.__name__}")
    print("ALL OK")

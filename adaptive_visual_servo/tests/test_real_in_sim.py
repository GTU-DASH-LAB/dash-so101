"""Exercises the exact code path run_real.py uses for its nominal model
(NominalModel.from_lengths + RealConfig's declared link lengths) against the
simulator standing in for hardware -- so the real-robot control logic gets
validated without needing the arm. control.py/perception.py are already
robot-agnostic (see tests/test_e2e.py); this only covers the one thing that
differs between sim and real: how the nominal model is constructed (perturbed
from ground truth in sim's own tests vs. a bare hand-measured guess here)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RealConfig, ServoConfig, SimConfig
from control import run_episode
from sim import NominalModel, SimWorld

SCFG = ServoConfig()


def test_from_lengths_matches_real_config_defaults():
    rc = RealConfig()
    m = NominalModel.from_lengths(rc.link_base, rc.link1, rc.link2)
    assert (m.lb, m.l1, m.l2) == (rc.link_base, rc.link1, rc.link2)
    q = np.array(SimConfig().home_q)
    assert np.linalg.norm(m.jac(q)) > 0, "usable Jacobian, not degenerate"


def test_run_real_style_model_survives_hand_measurement_error():
    """A user with a ruler won't measure exact link lengths -- perturb
    RealConfig's defaults the way a few-mm tape-measure error would, and
    confirm the episode still completes (XY stays visually closed regardless
    of this model's error, same tolerance as sim's own model_error)."""
    seed = 80
    world = SimWorld(SimConfig(), seed=seed)
    rng = np.random.default_rng(seed)
    rc = RealConfig()
    measured = (rc.link_base + 0.006, rc.link1 - 0.008, rc.link2 + 0.007)
    model = NominalModel.from_lengths(*measured)
    background = world.capture_background()
    world.spawn_random(1)
    res = run_episode(world, model, SCFG, background, rng)
    assert res["ok"], f"episode failed with hand-measured model: {res['reason']}"


if __name__ == "__main__":
    for fn in [test_from_lengths_matches_real_config_defaults,
               test_run_real_style_model_survives_hand_measurement_error]:
        fn()
        print(f"ok {fn.__name__}")
    print("ALL OK")

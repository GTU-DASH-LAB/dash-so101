"""End-to-end pick-and-place in randomized worlds, scored by sim ground truth."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ServoConfig, SimConfig
from control import run_episode
from sim import NominalModel, SimWorld

SCFG = ServoConfig()


def _delivered(world):
    return sum(np.hypot(o.pos[0] - world.pad[0], o.pos[1] - world.pad[1])
               < world.cfg.pad_radius + 0.02 and not o.attached
               for o in world.objects)


def _run_world(seed, n_objects, target_hue=None):
    world = SimWorld(SimConfig(), seed=seed)
    rng = np.random.default_rng(seed)
    model = NominalModel(world.cfg, rng)
    background = world.capture_background()
    world.spawn_random(n_objects)
    J = None
    runs = 1 if target_hue is not None else n_objects
    for _ in range(runs):
        res = run_episode(world, model, SCFG, background, rng,
                          target_hue=target_hue, J=J)
        J = res["J"]
    return world, res


def test_single_object():
    world, res = _run_world(seed=40, n_objects=1)
    assert res["ok"], f"episode failed: {res['reason']}"
    assert _delivered(world) == 1, "object not on the pad (ground truth)"


def test_randomized_batch():
    total = got = 0
    fails = []
    for seed in range(50, 62):
        n = 1 + seed % 2  # alternate 1 and 2 objects
        world, _ = _run_world(seed=seed, n_objects=n)
        d = _delivered(world)
        got += d
        total += n
        if d < n:
            fails.append((seed, d, n))
    rate = got / total
    print(f"  batch: {got}/{total} delivered ({rate:.0%}); misses: {fails}")
    assert rate >= 0.85, f"success rate {rate:.0%} below 85%: {fails}"


def test_pick_by_color():
    world = SimWorld(SimConfig(), seed=70)
    rng = np.random.default_rng(70)
    model = NominalModel(world.cfg, rng)
    background = world.capture_background()
    red = world.spawn_object((0.17, -0.02), 0.013, "circle", (30, 30, 210))
    blue = world.spawn_object((0.15, 0.07), 0.013, "square", (210, 60, 30))
    res = run_episode(world, model, SCFG, background, rng, target_hue=0)
    assert res["ok"], f"hue episode failed: {res['reason']}"
    assert np.hypot(red.pos[0] - world.pad[0], red.pos[1] - world.pad[1]) \
        < world.cfg.pad_radius + 0.02, "red object should be on the pad"
    assert np.hypot(blue.pos[0] - 0.15, blue.pos[1] - 0.07) < 0.01, \
        "blue object should be untouched"


def test_manual_target_and_pad():
    """A human-clicked pixel is exactly as valid a target as a detected
    one -- run_episode never needs to know an object was there, only where
    to aim. Skips detection entirely (manual_target_px/manual_pad_px), still
    exercises the full adaptive control loop (babble, Broyden, visual
    servo)."""
    world = SimWorld(SimConfig(), seed=80)
    rng = np.random.default_rng(80)
    model = NominalModel(world.cfg, rng)
    background = world.capture_background()
    (obj,) = world.spawn_random(1)
    target_px = world.cam.project(obj.pos)
    drop_world = np.array([0.10, 0.12, 0.0])  # off the default pad, a "clicked" spot
    drop_px = world.cam.project(drop_world)
    res = run_episode(world, model, SCFG, background, rng,
                      manual_target_px=target_px, manual_pad_px=drop_px)
    assert res["ok"], f"manual-target episode failed: {res['reason']}"
    assert np.hypot(obj.pos[0] - drop_world[0], obj.pos[1] - drop_world[1]) < 0.03, \
        "object should land at the manually clicked drop point, not the color pad"


if __name__ == "__main__":
    for fn in [test_single_object, test_randomized_batch, test_pick_by_color,
               test_manual_target_and_pad]:
        fn()
        print(f"ok {fn.__name__}")
    print("ALL OK")

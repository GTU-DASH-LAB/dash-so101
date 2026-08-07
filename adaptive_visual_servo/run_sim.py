"""Run adaptive visual-servo pick-and-place episodes in simulation.

    python adaptive_visual_servo/run_sim.py --episodes 5 --seed 0
    python adaptive_visual_servo/run_sim.py --episodes 1 --save-frames /tmp/avs

Each episode: fresh randomized world (object count/shape/color/position),
capture background, then deliver every object to the pad, reusing the babbled
Jacobian across objects. Success is scored with sim ground truth the
controller never sees. Exit code 0 iff every object was delivered.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ServoConfig, SimConfig
from control import run_episode
from sim import NominalModel, SimWorld


class Recorder:
    """Wraps a rig; dumps every frame the controller sees as JPEG."""

    def __init__(self, rig, out_dir):
        self._rig, self._dir, self._i = rig, out_dir, 0
        os.makedirs(out_dir, exist_ok=True)

    def read(self):
        import cv2
        frame = self._rig.read()
        cv2.imwrite(os.path.join(self._dir, f"{self._i:05d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        self._i += 1
        return frame

    def __getattr__(self, name):
        return getattr(self._rig, name)


def run(seed, n_objects, scfg, target_hue=None, save_frames=None, verbose=True):
    world = SimWorld(SimConfig(), seed=seed)
    rng = np.random.default_rng(seed)
    model = NominalModel(world.cfg, rng)      # the controller's wrong model
    background = world.capture_background()   # photo before objects are placed
    world.spawn_random(n_objects)
    rig = Recorder(world, save_frames) if save_frames else world

    J = None
    delivered = 0
    for i in range(n_objects):
        res = run_episode(rig, model, scfg, background, rng,
                          target_hue=target_hue, J=J)
        J = res["J"]
        # ground-truth scoring (test-side only)
        on_pad = sum(
            np.hypot(o.pos[0] - world.pad[0], o.pos[1] - world.pad[1])
            < world.cfg.pad_radius + 0.02 and not o.attached
            for o in world.objects)
        newly = on_pad - delivered
        delivered = on_pad
        if verbose:
            print(f"  object {i + 1}/{n_objects}: controller={res['reason']!r} "
                  f"ground_truth={'DELIVERED' if newly else 'NOT delivered'}")
        if target_hue is not None:
            break  # hue mode: one requested object
    return delivered, n_objects if target_hue is None else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--objects", type=int, default=0,
                    help="objects per episode (0 = random 1-3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hue", type=int, default=None,
                    help="only pick the object with this OpenCV hue (0-179)")
    ap.add_argument("--save-frames", default=None,
                    help="dump every controller frame as JPEG into this dir")
    args = ap.parse_args()

    scfg = ServoConfig()
    rng = np.random.default_rng(args.seed)
    total = got = 0
    for ep in range(args.episodes):
        n = args.objects or int(rng.integers(1, 4))
        print(f"episode {ep + 1}/{args.episodes} (seed {args.seed + ep}, {n} objects)")
        d, t = run(args.seed + ep, n, scfg, target_hue=args.hue,
                   save_frames=args.save_frames)
        got += d
        total += t
    print(f"\ndelivered {got}/{total} objects "
          f"({100.0 * got / max(total, 1):.0f}% ground-truth success)")
    sys.exit(0 if got == total else 1)


if __name__ == "__main__":
    main()

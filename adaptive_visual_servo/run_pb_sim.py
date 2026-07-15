"""Run adaptive visual-servo pick-and-place on the pybullet 6-DOF backend,
over the real downloaded SO-101 URDF (assets/SO101/).

    python adaptive_visual_servo/run_pb_sim.py --gui --episodes 2 --seed 9

`--gui` opens a live 3D window so you can watch the arm actually do the task
(pybullet's own OpenGL viewer). Without it, runs headless (much faster) --
useful for CI-style checks, same spirit as run_sim.py but on the real mesh.

Slower and currently less reliable than run_sim.py's toy world (see
README's "Known limitations" -- blink_locate's centroid bias on the real
gripper mesh is larger than on the toy sim's simple rendering), because this
one drives the actual URDF kinematics/physics/camera rather than a fast
analytic approximation.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ServoConfig
from control import run_episode
from lerobot_ik import PlacoModel
from pb_sim import PAD_CENTER, PyBulletWorld

# Tuned for the real gripper mesh's larger blink noise and to bound
# worst-case retry-cascade cost -- see README "Known limitations".
PB_SCFG = ServoConfig(tol_coarse_px=18.0, tol_fine_px=12.0, reject_px=45.0,
                      max_steps=35, retries=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--objects", type=int, default=1, help="objects per episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gui", action="store_true", help="open a live 3D view")
    ap.add_argument("--hue", type=int, default=None,
                    help="only pick the object with this OpenCV hue (0-179)")
    args = ap.parse_args()

    model = PlacoModel()
    delivered = total = 0
    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"episode {ep + 1}/{args.episodes} (seed {seed})")
        w = PyBulletWorld(gui=args.gui, seed=seed)
        try:
            bg = w.capture_background()
            objs = w.spawn_random(args.objects)
            rng = np.random.default_rng(seed)
            J = None
            for i, o in enumerate(objs):
                res = run_episode(w, model, PB_SCFG, bg, rng, target_hue=args.hue, J=J)
                J = res["J"]
                dist = np.hypot(*(w.object_xy(o) - np.array(PAD_CENTER)))
                delivered_this = dist < 0.06 and not o["attached"]
                print(f"  object {i + 1}/{len(objs)}: controller={res['reason']!r} "
                      f"ground_truth={'DELIVERED' if delivered_this else 'NOT delivered'}")
                delivered += int(delivered_this)
                total += 1
        finally:
            w.close()
    print(f"\ndelivered {delivered}/{total} objects "
          f"({100.0 * delivered / max(total, 1):.0f}% ground-truth success)")


if __name__ == "__main__":
    main()

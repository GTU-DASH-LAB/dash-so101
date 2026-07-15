"""Run adaptive visual-servo pick-and-place on the pybullet 6-DOF backend,
over the real downloaded SO-101 URDF (assets/SO101/).

    python adaptive_visual_servo/run_pb_sim.py --gui --episodes 2 --seed 9

`--gui` opens a live 3D window (pybullet's OpenGL viewer) AND a "controller
view" cv2 window showing exactly what the vision stack sees: detected objects
(green boxes), the chosen target (red), the drop pad (magenta), and the
current EE estimate (yellow cross). `--detector nanodet` runs the NanoDet
detector on the sim frames instead of background subtraction so you can see
its detections -- expect weak results on the sim's abstract shapes (they
don't resemble COCO classes; nanodet is meant for the real camera).

Note on the gripper opening/closing during motion: that IS the perception --
the EE is localized markerlessly by wiggling the gripper between two frames
and diffing them ("blink"). With measure_every=3 the servo dead-reckons on
the estimated Jacobian between blinks, so it happens 1/3 as often as it
used to.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ServoConfig
from control import run_episode
from lerobot_ik import PlacoModel
from pb_sim import PAD_CENTER, PyBulletWorld

# Tuned for the real gripper mesh's larger blink noise and to bound
# worst-case retry-cascade cost -- see README "Known limitations".
PB_SCFG = ServoConfig(tol_coarse_px=18.0, tol_fine_px=12.0, reject_px=45.0,
                      max_steps=35, retries=1, measure_every=3)


class ControllerView:
    """Wraps the rig: every frame the controller reads is also shown in a cv2
    window, annotated with the latest detection/target/EE-estimate events
    from run_episode's debug hook."""

    def __init__(self, rig):
        self._rig = rig
        self._ann = {}
        self._last = None

    def __getattr__(self, name):
        return getattr(self._rig, name)

    def read(self):
        frame = self._rig.read()
        self._last = frame
        self._show()
        return frame

    def debug(self, ev):
        self._ann.update({k: v for k, v in ev.items() if v is not None})
        self._show()

    def _show(self):
        if self._last is None:
            return
        img = self._last.copy()
        for b in self._ann.get("blobs", []):
            x, y, w, h = b.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        pad = self._ann.get("pad_px")
        if pad is not None:
            cv2.circle(img, tuple(np.int32(pad)), 8, (200, 0, 200), 2)
        tgt = self._ann.get("target")
        if tgt is not None:
            cv2.circle(img, tuple(np.int32(tgt)), 6, (0, 0, 255), 2)
            cv2.circle(img, tuple(np.int32(tgt)), 1, (0, 0, 255), -1)
        s = self._ann.get("s")
        if s is not None:
            u, v = np.int32(s)
            cv2.drawMarker(img, (u, v), (0, 220, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(img, "green=detections red=target magenta=pad yellow=EE",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        cv2.imshow("controller view", img)
        cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--objects", type=int, default=1, help="objects per episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gui", action="store_true",
                    help="live 3D view + annotated controller-view window")
    ap.add_argument("--hue", type=int, default=None,
                    help="only pick the object with this OpenCV hue (0-179)")
    ap.add_argument("--detector", choices=["bgsub", "nanodet"], default="bgsub",
                    help="nanodet mostly won't recognize the sim's abstract "
                         "shapes (not COCO objects) -- offered so you can SEE "
                         "what it detects; bgsub is the sim default")
    args = ap.parse_args()

    model = PlacoModel()
    detector = None
    if args.detector == "nanodet":
        from nanodet_detector import NanodetDetector
        detector = NanodetDetector(class_names=None, score_thresh=0.3)

    delivered = total = 0
    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"episode {ep + 1}/{args.episodes} (seed {seed})")
        w = PyBulletWorld(gui=args.gui, seed=seed)
        rig = ControllerView(w) if args.gui else w
        try:
            bg = rig.capture_background()
            objs = w.spawn_random(args.objects)
            rng = np.random.default_rng(seed)
            J = None
            for i, o in enumerate(objs):
                res = run_episode(rig, model, PB_SCFG, bg, rng, target_hue=args.hue,
                                  detector=detector, J=J,
                                  debug=rig.debug if args.gui else None)
                J = res["J"]
                dist = np.hypot(*(w.object_xy(o) - np.array(PAD_CENTER)))
                delivered_this = dist < 0.06 and not o["attached"]
                print(f"  object {i + 1}/{len(objs)}: controller={res['reason']!r} "
                      f"ground_truth={'DELIVERED' if delivered_this else 'NOT delivered'}")
                delivered += int(delivered_this)
                total += 1
        finally:
            w.close()
    if args.gui:
        cv2.destroyAllWindows()
    print(f"\ndelivered {delivered}/{total} objects "
          f"({100.0 * delivered / max(total, 1):.0f}% ground-truth success)")


if __name__ == "__main__":
    main()

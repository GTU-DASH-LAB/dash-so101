"""Real SO-101 hardware runner for adaptive_visual_servo.

Wraps lerobot's SOFollower (SO-100/101 driver) + an OpenCV camera behind the
same duck-typed rig interface control.py already exercises against SimWorld
and pb_sim.PyBulletWorld: get_q/set_q/set_gripper/gripper_contact/read (all
in radians / [0,1] / BGR frames). control.py and perception.py are otherwise
unchanged from sim to real -- only this file talks to lerobot. Drives all 6
motors: 5 arm joints (shoulder_pan/lift, elbow_flex, wrist_flex, wrist_roll)
directly through the visual servo's Jacobian, gripper separately.

Object detection defaults to NanoDet-Plus (nanodet_detector.py) -- a real
trained COCO detector suits the real camera and everyday objects (ball,
bottle, cup, fruit...). `--detector bgsub` falls back to background
subtraction for arbitrary non-COCO objects; `--classes` tunes the allowlist.

*** UNTESTED against real hardware. *** First run checklist:
  - Arm powered, workspace clear, YOUR hand near the power switch.
  - Camera fixed overhead/eye-to-hand, matching the sim's assumption.
  - Position the arm somewhere safe before connecting -- home is wherever
    it's sitting when this script starts (same pattern as ui/server.py's
    Reset), not a hardcoded pose.
  - `--calibrate-gripper` FIRST: config.py's gripper_open_pos/closed_pos and
    load_threshold are placeholders. Run it, watch the printed load values,
    and set real numbers in config.py before trusting grasp detection.
  - Z-descend uses lerobot_ik.PlacoModel against the real downloaded URDF
    (assets/SO101/so101_new_calib.urdf) -- no per-arm length calibration
    needed there; XY stays visually closed regardless of any residual error.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RealConfig, ServoConfig
from control import run_episode
from lerobot_ik import PlacoModel
from nanodet_detector import TABLETOP_CLASSES, NanodetDetector


class SO101Rig:
    """rig interface in radians / [0,1] / BGR, converted to the bus's native
    degrees/0-100/RGB at this one boundary so ServoConfig's numbers mean the
    same thing in sim and on hardware."""

    def __init__(self, cfg: RealConfig):
        from lerobot.cameras.configs import ColorMode
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower

        self.cfg = cfg
        cam = OpenCVCameraConfig(index_or_path=cfg.camera_index, fps=30,
                                 width=640, height=480, color_mode=ColorMode.BGR)
        self.robot = SOFollower(SOFollowerRobotConfig(
            port=cfg.port, id=cfg.robot_id, cameras={"cam": cam},
            max_relative_target=None,  # we apply our own smaller per-step clamp
            use_degrees=True))
        self.robot.connect(calibrate=True)
        self._g = 1.0

    def close(self):
        self.robot.disconnect()

    # ---- rig interface ----
    def get_q(self):
        pos = self.robot.bus.sync_read("Present_Position", list(self.cfg.joints))
        return np.radians([pos[j] for j in self.cfg.joints])

    def set_q(self, q):
        c = self.cfg
        q = np.clip(q, np.radians(c.q_min_deg), np.radians(c.q_max_deg))
        deg_before = np.degrees(self.get_q())
        step = np.clip(np.degrees(q) - deg_before, -c.max_step_deg, c.max_step_deg)
        deg = deg_before + step
        action = {f"{j}.pos": float(v) for j, v in zip(c.joints, deg)}
        self.robot.send_action(action)
        time.sleep(c.settle_s)

    def set_gripper(self, g):
        self._g = float(np.clip(g, 0.0, 1.0))
        c = self.cfg
        pos = c.gripper_closed_pos + self._g * (c.gripper_open_pos - c.gripper_closed_pos)
        self.robot.send_action({"gripper.pos": float(pos)})
        time.sleep(c.settle_s)

    def gripper_contact(self):
        load = abs(self.robot.bus.read("Present_Load", "gripper"))
        return self._g < 0.5 and load > self.cfg.load_threshold

    def read(self):
        return self.robot.cameras["cam"].read_latest()

    def capture_background(self):
        return self.read()


def calibrate_gripper(rig: SO101Rig):
    """Cycle the gripper and print Present_Load so you can pick real
    gripper_open_pos/gripper_closed_pos/load_threshold values for config.py."""
    print("Watch these values: open/closed positions and the load spike when "
          "fingers meet resistance (press an object, or just close on nothing "
          "to see the free-swing baseline). Ctrl-C to stop.")
    g = 1.0
    try:
        while True:
            rig.set_gripper(g)
            load = rig.robot.bus.read("Present_Load", "gripper")
            print(f"g={g:.2f} load={load}")
            time.sleep(0.5)
            g = 0.0 if g > 0.5 else 1.0
    except KeyboardInterrupt:
        rig.set_gripper(1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cfg = RealConfig()
    ap.add_argument("--port", default=cfg.port)
    ap.add_argument("--camera", type=int, default=cfg.camera_index)
    ap.add_argument("--id", default=cfg.robot_id)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--hue", type=int, default=None,
                    help="only pick the object with this OpenCV hue (0-179)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate-gripper", action="store_true")
    ap.add_argument("--detector", choices=["nanodet", "bgsub"], default="nanodet",
                    help="nanodet (default): real trained COCO detector, best for "
                         "everyday objects on the real camera. bgsub: background "
                         "subtraction, any object but needs a clean background photo")
    ap.add_argument("--classes", default=None,
                    help="comma-separated COCO class allowlist for nanodet "
                         f"(default: {','.join(TABLETOP_CLASSES)}; 'all' = all 80)")
    args = ap.parse_args()

    cfg.port, cfg.camera_index, cfg.robot_id = args.port, args.camera, args.id
    rig = SO101Rig(cfg)
    try:
        if args.calibrate_gripper:
            calibrate_gripper(rig)
            return
        model = PlacoModel()
        # pb_sim testing (same URDF gripper mesh) found the toy sim's tight
        # tol_coarse_px=7/tol_fine_px=4 just chatters forever on real-mesh
        # blink_locate noise -- start looser here too, tune further once you
        # can watch the real camera's actual blink noise on hardware.
        scfg = ServoConfig(tol_coarse_px=18.0, tol_fine_px=12.0, reject_px=45.0,
                           measure_every=3)  # blink 1/3 as often; J dead-reckons between
        rng = np.random.default_rng(args.seed)
        detector = None
        if args.detector == "nanodet":
            classes = (None if args.classes == "all" else
                       tuple(args.classes.split(",")) if args.classes else
                       TABLETOP_CLASSES)
            detector = NanodetDetector(class_names=classes)
        # background photo still needed even with nanodet: locate_by_diff
        # (carry-phase tracking) diffs against it regardless of the detector
        input("Workspace clear of objects for the background photo -- "
              "press ENTER when ready...")
        background = rig.capture_background()
        J = None
        for ep in range(args.episodes):
            print(f"episode {ep + 1}/{args.episodes}")
            res = run_episode(rig, model, scfg, background, rng,
                              target_hue=args.hue, detector=detector, J=J)
            J = res["J"]
            print(f"  {res['reason']!r}")
            if res["reason"] in ("no objects detected", "drop pad not found"):
                break
    finally:
        rig.set_gripper(1.0)
        rig.close()


if __name__ == "__main__":
    main()

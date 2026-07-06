"""Classical pick-and-place: demo-calibrated waypoints + continuous visual servoing.

No learned policy. The camera-to-joints mapping comes from so_brain/pid_calib.json
(fit from the teleop demos by mine_calibration.py). In IK mode (the default, when
so_brain/kinematics.py has been run), the descent onto the object AND onto the
destination re-detect their target every iteration and re-solve the arm pose via
URDF forward/inverse kinematics — real per-frame image feedback, not a single
open-loop trajectory. The gripper's own position comes from joint encoders + FK,
so no probing motion is needed to find it (unlike wiggle-based servoing).

    ./so_brain/pid.sh                        # pick the calib phrases
    ./so_brain/pid.sh --pick "a pen" --place "a black mouse pad"
    ./so_brain/pid.sh --dry-run              # detect + print poses, no robot motion

Sequence: home -> hover above object -> servo-descend (continuous feedback) ->
close -> lift -> carry -> servo-descend onto destination -> open -> home.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from so_brain import grounding
from so_brain.kinematics import apply_homography  # placo-free: safe without --no-ik
from so_brain.mine_calibration import JOINTS, features

FPS = 30.0
SERVO_MAX_ITERS = 25
SERVO_Z_RATE = 0.012  # meters descended per iteration cap — bounds descent speed
SERVO_XY_TOL = 0.004  # meters; stop refining horizontal position below this
SERVO_Z_TOL = 0.003  # meters


class Calib:
    def __init__(self, path: str):
        self.raw = json.load(open(path))
        self.lo = np.array(self.raw["joint_min"])
        self.hi = np.array(self.raw["joint_max"])

    def predict(self, kind: str, uv) -> np.ndarray:
        W = np.array(self.raw[kind]["W"])
        pose = features(np.asarray(uv, dtype=float)) @ W
        return np.clip(pose, self.lo, self.hi)

    def in_workspace(self, uv) -> bool:
        (ulo, vlo), (uhi, vhi) = self.raw["pen_uv_range"]
        margin = 0.05
        return ulo - margin <= uv[0] <= uhi + margin and vlo - margin <= uv[1] <= vhi + margin


class Camera:
    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open camera {index}")

    def frame(self) -> np.ndarray:
        for _ in range(4):  # flush stale buffered frames
            self.cap.grab()
        ok, bgr = self.cap.read()
        if not ok:
            raise SystemExit("camera read failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()


class Arm:
    """Thin wrapper over SOFollower: read the 6-dof state, glide to a target pose."""

    def __init__(self, port: str, robot_id: str):
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        self.robot = SOFollower(SOFollowerRobotConfig(port=port, id=robot_id, cameras={}))
        self.robot.connect()

    def state(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[f"{j}.pos"] for j in JOINTS])

    def goto(self, pose5, gripper: float, duration: float):
        """Cosine-eased joint interpolation from the current state."""
        start = self.state()
        target = np.array([*pose5, gripper])
        steps = max(int(duration * FPS), 1)
        for i in range(1, steps + 1):
            a = (1 - np.cos(np.pi * i / steps)) / 2
            q = start + a * (target - start)
            self.robot.send_action({f"{j}.pos": float(q[k]) for k, j in enumerate(JOINTS)})
            time.sleep(1 / FPS)

    def step(self, pose5, gripper: float):
        """Command a pose directly, for one control tick (no interpolation)."""
        q = np.append(pose5, gripper)
        self.robot.send_action({f"{j}.pos": float(q[k]) for k, j in enumerate(JOINTS)})

    def disconnect(self):
        self.robot.disconnect()


def servo_descend(arm: Arm, cam: Camera, calib: Calib, kin, phrase: str,
                  z_target: float, gripper: float) -> np.ndarray:
    """Continuously re-detect `phrase` and descend the gripper onto it.

    Every iteration: grab a fresh frame, detect the target, convert its pixel to a
    table (x, y) via the demo-fitted homography, read the arm's actual position from
    joint encoders (FK — no wiggle-probing needed), and refine the pose toward the
    live target. Descent is rate-limited (SERVO_Z_RATE/iteration) so horizontal
    corrections keep being applied throughout the whole way down, not just once at
    hover height. Returns the final commanded pose.
    """
    table = calib.raw["table"]
    H = np.array(table["H"])
    last_xy = None
    sol = arm.state()[:5]
    for it in range(1, SERVO_MAX_ITERS + 1):
        cur_xyz = kin.fk(sol)[:3, 3]
        try:
            u, v, _ = grounding.locate(cam.frame(), phrase)
            last_xy = apply_homography(H, np.array([u, v]))[0]
        except LookupError:
            if last_xy is None:
                raise SystemExit(f"{phrase!r} not visible; cannot servo onto it")
            print(f"servo[{it}]: {phrase!r} lost from view; holding last known position")
        z_now = cur_xyz[2]
        z_step = z_now + np.clip(z_target - z_now, -SERVO_Z_RATE, SERVO_Z_RATE)
        target_xyz = np.array([last_xy[0], last_xy[1], z_step])
        try:
            sol = np.clip(kin.refine_position(sol, target_xyz), calib.lo, calib.hi)
        except RuntimeError as e:
            print(f"servo[{it}]: {e}; holding position")
            continue
        arm.step(sol, gripper=gripper)
        time.sleep(1 / FPS)
        err_xy = float(np.hypot(last_xy[0] - cur_xyz[0], last_xy[1] - cur_xyz[1]))
        err_z = abs(z_target - z_now)
        print(f"servo[{it}]: xy_err={err_xy*100:.2f}cm z_err={err_z*100:.2f}cm z_now={z_now*100:.1f}cm")
        if err_xy < SERVO_XY_TOL and err_z < SERVO_Z_TOL:
            break
    return sol


def main():
    env = os.environ
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="so_brain/pid_calib.json")
    ap.add_argument("--pick", default=None, help="detector phrase (default: from calib)")
    ap.add_argument("--place", default=None)
    ap.add_argument("--port", default=env.get("FOLLOWER_PORT", "/dev/ttyACM1"))
    ap.add_argument("--id", default=env.get("FOLLOWER_ID", "my_follower"))
    ap.add_argument("--camera", type=int, default=int(env.get("CAMERA_INDEX", 0)))
    ap.add_argument("--no-servo", action="store_true",
                    help="skip continuous visual servoing; use the single regression pose")
    ap.add_argument("--no-ik", action="store_true",
                    help="use the pixel->pose regression instead of URDF IK "
                    "(also disables continuous servoing, which needs FK)")
    ap.add_argument("--hover-height", type=float, default=0.06,
                    help="approach height above the grasp plane, meters (IK mode)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    calib = Calib(args.calib)
    pick_phrase = args.pick or calib.raw["pick_phrase"]
    place_phrase = args.place or calib.raw["place_phrase"]
    use_ik = "table" in calib.raw and not args.no_ik
    use_servo = use_ik and not args.no_servo  # continuous per-frame feedback needs FK

    cam = Camera(args.camera, int(env.get("CAMERA_W", 640)), int(env.get("CAMERA_H", 480)))
    frame = cam.frame()
    pen = np.array(grounding.locate(frame, pick_phrase)[:2])
    pad = np.array(grounding.locate(frame, place_phrase)[:2])
    print(f"{pick_phrase!r} at ({pen[0]:.3f}, {pen[1]:.3f}) | {place_phrase!r} at ({pad[0]:.3f}, {pad[1]:.3f})")
    if not calib.in_workspace(pen):
        raise SystemExit(f"object at ({pen[0]:.2f}, {pen[1]:.2f}) is outside the demo workspace "
                         f"{calib.raw['pen_uv_range']} — move it toward the table center")

    hover = calib.predict("approach", pen)
    grasp = calib.predict("grasp", pen)
    kin = None
    if use_ik:
        # Geometric polish: pixel -> table (x, y) via the demo-fitted homography, then
        # a local damped-least-squares correction of the regression pose so the
        # gripper lands exactly there (URDF FK). The regression seed keeps the arm in
        # the demonstrated configuration; the refinement only fixes position
        # (validated offline: median 0.3cm across the 50 demos).
        from so_brain.kinematics import SO101Kinematics

        kin = SO101Kinematics()
        table = calib.raw["table"]
        xy = apply_homography(np.array(table["H"]), pen)[0]
        print(f"IK mode: object on the table at x={xy[0]*100:.1f}cm y={xy[1]*100:.1f}cm")
        try:
            grasp = np.clip(
                kin.refine_position(grasp, [xy[0], xy[1], table["z_grasp"]]),
                calib.lo, calib.hi)
            hover = np.clip(
                kin.refine_position(hover, [xy[0], xy[1], table["z_grasp"] + args.hover_height]),
                calib.lo, calib.hi)
        except RuntimeError as e:
            print(f"IK refinement failed ({e}); using the regression poses as-is")
    # Median demonstrated release pose — the pad was fixed across demos, so the
    # pixel->pose regression has no signal there. Its FK height is the servo's
    # descent target when tracking the destination live below.
    place = np.array(calib.raw["place_median"])
    place_z = kin.fk(place)[2, 3] if use_ik else None
    lift_i = JOINTS.index("shoulder_lift")
    carry = place.copy()
    carry[lift_i] = hover[lift_i]  # cross the table at hover height, not dragging
    g_open, g_closed = calib.raw["gripper_open"], calib.raw["gripper_closed"]
    for name, pose in (("hover", hover), ("grasp", grasp), ("carry", carry), ("place", place)):
        print(f"{name:6s}: " + "  ".join(f"{j}={p:6.1f}" for j, p in zip(JOINTS, pose)))
    print(f"gripper: open={g_open:.1f} closed={g_closed:.1f}")
    if args.dry_run:
        cam.release()
        return

    arm = Arm(args.port, args.id)
    try:
        print("-> home")
        arm.goto(calib.raw["home"], gripper=g_open, duration=2.5)
        print("-> hover above the object")
        arm.goto(hover, gripper=g_open, duration=2.5)
        if use_servo:
            print("-> servo-descend onto the object (continuous per-frame feedback)")
            grasp = servo_descend(arm, cam, calib, kin, pick_phrase, table["z_grasp"], g_open)
            # Lift straight up from wherever the servo actually grasped, rather than
            # jumping sideways to the pre-servo (coarser) hover estimate.
            grasp_xy = kin.fk(grasp)[:2, 3]
            lift_target = [grasp_xy[0], grasp_xy[1], table["z_grasp"] + args.hover_height]
            hover = np.clip(kin.refine_position(grasp, lift_target), calib.lo, calib.hi)
        else:
            print("-> descend")
            arm.goto(grasp, gripper=g_open, duration=1.5)
        print("-> close")
        arm.goto(grasp, gripper=g_closed, duration=0.8)
        print("-> lift")
        arm.goto(hover, gripper=g_closed, duration=1.5)
        print("-> carry")
        arm.goto(carry, gripper=g_closed, duration=2.5)
        if use_servo:
            print("-> servo-descend onto the destination (continuous per-frame feedback)")
            place = servo_descend(arm, cam, calib, kin, place_phrase, place_z, g_closed)
        else:
            print("-> place")
            arm.goto(place, gripper=g_closed, duration=1.5)
        print("-> release")
        arm.goto(place, gripper=g_open, duration=0.8)
        print("-> home")
        arm.goto(calib.raw["home"], gripper=g_open, duration=2.5)

        after = cam.frame()
        try:
            u, v, _ = grounding.locate(after, pick_phrase)
            dist = float(np.hypot(u - pad[0], v - pad[1]))
            print(f"outcome: object at ({u:.3f}, {v:.3f}), {dist:.3f} from the destination "
                  + ("— SUCCESS" if dist < 0.12 else "— MISS"))
        except LookupError:
            print("outcome: object not visible after the attempt")
    finally:
        arm.disconnect()
        cam.release()


if __name__ == "__main__":
    main()

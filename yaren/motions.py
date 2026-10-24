"""Yaren's own movements: short gestures (hello, nod, ...) and moving one part of the arm on request.

Everything is planned here, in joint space, before the arm moves, and refused or cut short when it is
not safe:
- joint limits: the envelope of the 100 teleoperated demonstrations of so101_pen_in_bowl_wrist
  (65,550 frames), except wrist_flex, which may go down to -30 so the gripper can point straight up;
- speed: per joint, no faster than the 99th percentile of those demonstrations; lively parts of a
  gesture (a head shake, a nod) up to the 99.9th, a part moved on request at half the 99th. Cosine
  easing, so every segment starts and ends at rest;
- the table: forward kinematics of the SO-101 (so101_new_calib.urdf, whose joint angles are lerobot's
  degrees) keeps the elbow, the wrist and the gripper tip at least 4 cm above the table, or, when a
  move starts lower than that, never lower than where it started. Only a return to the pose a gesture
  started from may go back down.

robot_server.py runs a Plan in its control thread like homing: one interpolated command per tick, and a
STOP ends it at once (the servos hold the last command). Pure numpy: test_motions.py checks every
gesture and part move offline, no robot needed.
"""

import math

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Demonstration envelope (action min..max over all frames), degrees; gripper 0..100 = closed..open.
LIMITS = {
    "shoulder_pan": (-80.0, 60.0),     # demos -81.5 .. 61.0
    "shoulder_lift": (-104.0, 83.0),   # demos -107.8 .. 83.1 (calibrated range +-104.4)
    "elbow_flex": (-95.0, 97.0),       # demos -97.0 .. 97.1
    "wrist_flex": (-30.0, 102.0),      # demos 18.0 .. 104.6; the URDF allows -95
    "wrist_roll": (12.0, 146.0),       # demos 11.4 .. 146.4
    "gripper": (0.0, 45.0),            # demos 0.0 .. 45.3
}
# Per-joint speed caps (deg/s, gripper %/s): percentiles of |action velocity| in the demonstrations.
SPEED = {"shoulder_pan": 53, "shoulder_lift": 105, "elbow_flex": 98, "wrist_flex": 74, "wrist_roll": 53,
         "gripper": 85}           # p99
LIVELY = {"shoulder_pan": 79, "shoulder_lift": 142, "elbow_flex": 140, "wrist_flex": 111, "wrist_roll": 82,
          "gripper": 124}         # p99.9
GENTLE = {j: v / 2 for j, v in SPEED.items()}   # a single part moved on request: calm and easy to follow
MIN_SEGMENT_S = 0.25
MIN_MOVE_S = 0.6
CLEARANCE_M = 0.04                # elbow, wrist and gripper tip above the table
TOLERANCE_M = 0.002               # numerical slack only

# Standing tall: upper arm and forearm vertical, gripper pointing up (tip ~52 cm above the table).
STAND = {"shoulder_lift": -30.0, "elbow_flex": -60.0, "wrist_flex": 0.0}

# so101_new_calib.urdf (TheRobotStudio SO-ARM100; copy in adaptive_visual_servo/assets/SO101/):
# (joint, origin xyz, origin rpy); every revolute joint turns about its z axis. base_link z = 0 is the
# table: the recorded rest pose puts the tip at 0.5 cm, the moment of grasping at 2 cm.
CHAIN = [
    ("shoulder_pan", (0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),
    ("shoulder_lift", (-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),
    ("elbow_flex", (-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),
    ("wrist_flex", (-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),
    ("wrist_roll", (5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),
    ("gripper_frame_joint", (-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0)),
]

# What people call the parts -> (joint, {direction: sign}). None: decided by the kinematics (up/down).
PARTS = {
    "base": ("shoulder_pan", {"left": -1, "right": +1}),        # the robot's own left (+y) / right
    "shoulder": ("shoulder_lift", None),
    "elbow": ("elbow_flex", None),
    "wrist": ("wrist_flex", None),
    "wrist_roll": ("wrist_roll", {"left": -1, "right": +1}),    # right = clockwise seen from above
    "gripper": ("gripper", {"open": +1, "close": -1}),
}
UP_DOWN_DEFAULT = -1            # where the kinematics cannot tell (arm vertical): minus raises the arm
DEFAULT_DEG = 20.0
MAX_DEG = 45.0
MIN_MOVE_DEG = 3.0              # a move the table would cut shorter than this is refused (invisible)


class Refused(Exception):
    """A request that cannot be done safely; the message goes back to the LLM."""


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


_ORIGINS = []
for _name, _xyz, _r in CHAIN:
    _A = np.eye(4)
    _A[:3, :3], _A[:3, 3] = _rpy(*_r), _xyz
    _ORIGINS.append(_A)


def points(pose):
    """Elbow, wrist and gripper-tip positions (m, base frame: x forward, y left, z up) of a pose."""
    T, out = np.eye(4), {}
    for (name, _, _), A in zip(CHAIN, _ORIGINS):
        q = math.radians(pose.get(name, 0.0)) if name in JOINTS else 0.0
        R = np.eye(4)
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = math.cos(q), -math.sin(q), math.sin(q), math.cos(q)
        T = T @ A @ R
        out[name] = T[:3, 3].copy()
    return {"elbow": out["elbow_flex"], "wrist": out["wrist_flex"], "tip": out["gripper_frame_joint"]}


def clamp(pose):
    return {j: min(max(v, LIMITS[j][0]), LIMITS[j][1]) if j in LIMITS else v for j, v in pose.items()}


def _duration(a, b, speeds):
    # cosine easing: peak speed = pi/2 * distance / duration
    return max([MIN_SEGMENT_S] + [math.pi / 2 * abs(b[j] - a[j]) / speeds[j] for j in JOINTS if j in a and j in b])


class Plan:
    """Keyframes joined by cosine-eased segments. at(t) is the command for time t (s) since the start."""

    def __init__(self, name, start, keyframes, summary=None):
        self.name, self.start, self.segs = name, dict(start), []
        t, prev = 0.0, dict(start)
        for pose, speeds, hold, home in keyframes:
            pose = {**prev, **pose}
            T = _duration(prev, pose, speeds) if speeds is not GENTLE else max(MIN_MOVE_S, _duration(prev, pose, speeds))
            self.segs.append((t, T, prev, pose, home))
            t += T
            if hold:
                self.segs.append((t, hold, pose, pose, home))
                t += hold
            prev = pose
        self.seconds, self.end = t, prev
        self.summary = {"name": name, "seconds": round(t, 2), **(summary or {})}

    def at(self, t):
        for t0, T, a, b, _ in self.segs:
            if t < t0 + T:
                s = 0.5 - 0.5 * math.cos(math.pi * max(0.0, t - t0) / T)
                return {j: a[j] + (b[j] - a[j]) * s for j in a}
        return dict(self.end)

    def lowest(self, dt=1 / 30):
        """Worst clearance violation along the plan: (metres below the allowed floor, time, point) or None."""
        worst = None
        for t0, T, a, b, home in self.segs:
            za, zb = points(a), points(b)
            for k in range(math.ceil(T / dt) + 1):      # always includes the segment's end
                s = 0.5 - 0.5 * math.cos(math.pi * min(1.0, k * dt / T))
                p = points({j: a[j] + (b[j] - a[j]) * s for j in a})
                for name in p:
                    floor = min(CLEARANCE_M, za[name][2], zb[name][2] if home else CLEARANCE_M) - TOLERANCE_M
                    below = floor - p[name][2]
                    if below > 0 and (worst is None or below > worst[0]):
                        worst = (below, t0 + k * dt, name)
        return worst


def _kf(pose, lively=False, hold=0.0, home=False, speeds=None):
    return (pose, speeds or (LIVELY if lively else SPEED), hold, home)


def _shake(U, n, amp_pan=15.0, amp_roll=20.0, talk=False):
    """Head shake with the arm upright: base and wrist roll turn about the same vertical line."""
    p, r = U["shoulder_pan"], U["wrist_roll"]
    out = []
    for _ in range(n):
        for side in (+1, -1):
            pose = {"shoulder_pan": p + side * amp_pan, "wrist_roll": r + side * amp_roll}
            if talk:
                pose["gripper"] = 35.0 if side > 0 else 3.0
            out.append(_kf(clamp(pose), lively=True))
    return out


def _g_hello(S, U):
    return [_kf(U, hold=0.2)] + _shake(U, 2, talk=True) + [_kf(U, lively=True, hold=0.2)]


def _g_nod(S, U):
    w = U["wrist_flex"]
    return [_kf(U, hold=0.2)] + [_kf(clamp({"wrist_flex": w + d}), lively=True) for _ in range(3) for d in (25, -10)] \
        + [_kf(U, lively=True, hold=0.2)]


def _g_shake_head(S, U):
    return [_kf(U, hold=0.2)] + _shake(U, 3) + [_kf(U, lively=True, hold=0.2)]


def _g_dance(S, U):
    p, r, w = U["shoulder_pan"], U["wrist_roll"], U["wrist_flex"]
    steps = []
    for _ in range(3):
        for side in (+1, -1):
            steps.append(_kf(clamp({"shoulder_pan": p + 20 * side, "wrist_roll": r - 20 * side,
                                    "wrist_flex": w + (20 if side > 0 else -15)}), lively=True))
    return [_kf(U, hold=0.2)] + steps + [_kf(U, lively=True, hold=0.2)]


def _g_bow(S, U):
    bow = clamp({"shoulder_lift": U["shoulder_lift"] + 30, "wrist_flex": U["wrist_flex"] + 50})
    return [_kf(U, hold=0.2), _kf(bow, hold=0.8), _kf(U, hold=0.2)]


def _g_clap(S, U):
    return [_kf(U, hold=0.2)] + [_kf({"gripper": g}, lively=True) for _ in range(3) for g in (40.0, 0.0)] \
        + [_kf({"gripper": U["gripper"]}, lively=True, hold=0.2)]


def _g_look_around(S, U):
    p = U["shoulder_pan"]
    look = clamp({"wrist_flex": 60.0})
    return [_kf(U), _kf(look, hold=0.3), _kf(clamp({"shoulder_pan": p + 35}), hold=0.6),
            _kf(clamp({"shoulder_pan": p - 35}), hold=0.6), _kf({"shoulder_pan": p}, hold=0.3), _kf(U, hold=0.2)]


def _g_stand_up(S, U):
    return [_kf(U)]


GESTURES = {  # name: (keyframes, returns to the start pose afterwards)
    "hello": (_g_hello, True), "nod": (_g_nod, True), "shake_head": (_g_shake_head, True),
    "dance": (_g_dance, True), "bow": (_g_bow, True), "clap": (_g_clap, True),
    "look_around": (_g_look_around, True), "stand_up": (_g_stand_up, False),
}


def _check(plan, what):
    bad = plan.lowest()
    if bad:
        raise Refused(f"{what} would bring my {bad[2]} within {CLEARANCE_M * 100:.0f} cm of the table from "
                      "this position; ask me to go home first")
    return plan


def plan_gesture(name, start):
    """start: the current joint positions {joint: degrees} (all six)."""
    if name not in GESTURES:
        raise Refused(f"unknown gesture '{name}'; known: {', '.join(GESTURES)}")
    fn, back = GESTURES[name]
    S = {j: float(start[j]) for j in JOINTS}
    U = clamp({**S, **STAND})
    frames = fn(S, U)
    if back:
        frames.append(_kf(S, home=True))
    return _check(Plan(name, S, frames, {"returns": back}), f"the {name} gesture")


def _tip_z(pose):
    return points(pose)["tip"][2]


def plan_move(start, part, direction, degrees=None):
    """Move one part by `degrees` (gripper: percentage points of its opening) in a direction people use."""
    if part not in PARTS:
        raise Refused(f"unknown part '{part}'; known: {', '.join(PARTS)}")
    joint, signs = PARTS[part]
    allowed = signs.keys() if signs else ("up", "down")
    if direction not in allowed:
        raise Refused(f"the {part} moves {' or '.join(allowed)}, not '{direction}'")
    S = {j: float(start[j]) for j in JOINTS}
    if joint == "gripper" and degrees is None:
        target = LIMITS["gripper"][1] if direction == "open" else LIMITS["gripper"][0]
        amount = abs(target - S["gripper"])
    else:
        amount = min(max(abs(float(degrees if degrees is not None else DEFAULT_DEG)), 1.0), MAX_DEG)
    if signs:
        sign = signs[direction]
    else:   # up/down: the sign that raises/lowers the gripper tip from here
        z0, want = _tip_z(S), (1 if direction == "up" else -1)
        dz = {sg: _tip_z(clamp({**S, joint: S[joint] + sg * amount})) - z0 for sg in (+1, -1)}
        sign = max(dz, key=lambda sg: want * dz[sg])
        if want * dz[sign] < 0.01:
            sign = UP_DOWN_DEFAULT * want
    target = min(max(S[joint] + sign * amount, LIMITS[joint][0]), LIMITS[joint][1])
    limited = "joint limit" if abs(target - S[joint]) < amount - 0.5 else None
    if (target - S[joint]) * sign < 0.5:
        raise Refused(f"my {part} is already as far {direction} as I let it go from this position; "
                      "standing up first (stand_up gesture) or moving another part may help")
    # cut the move short where it would come too close to the table
    full = target - S[joint]
    lo, hi = 0.0, 1.0
    if Plan("move", S, [_kf({joint: target})]).lowest():
        limited = "table"
        for _ in range(12):
            mid = (lo + hi) / 2
            if Plan("move", S, [_kf({joint: S[joint] + mid * full})]).lowest():
                hi = mid
            else:
                lo = mid
        if abs(lo * full) < MIN_MOVE_DEG:
            raise Refused(f"moving my {part} {direction} would push my gripper into the table; "
                          "stand up first (stand_up gesture) or lift the elbow")
        target = S[joint] + lo * full
    end = {**S, joint: target}
    zs, ze = _tip_z(S), _tip_z(end)
    summary = {"part": part, "direction": direction, "joint": joint, "moved": round(target - S[joint], 1),
               "unit": "%" if joint == "gripper" else "deg", "limited_by": limited,
               "tip_height_cm": [round(zs * 100), round(ze * 100)]}
    return Plan(f"{part} {direction}", S, [_kf({joint: target}, speeds=GENTLE)], summary)


def to_action(pose):
    return {f"{j}.pos": float(v) for j, v in pose.items()}


def from_obs(joints):
    return {k.removesuffix(".pos"): float(v) for k, v in joints.items() if k.removesuffix(".pos") in JOINTS}

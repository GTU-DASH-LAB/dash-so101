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
        for kf in keyframes:
            pose, speeds, hold, home = kf[:4]
            pose = {**prev, **pose}
            T = _duration(prev, pose, speeds) if speeds is not GENTLE else max(MIN_MOVE_S, _duration(prev, pose, speeds))
            if len(kf) > 4 and kf[4]:           # a fixed length (on the beat), never faster than allowed
                T = max(T, kf[4])
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


def _kf(pose, lively=False, hold=0.0, home=False, speeds=None, dur=None):
    return (pose, speeds or (LIVELY if lively else SPEED), hold, home, dur)


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


# ---------------------------------------------------------------- named poses
# survey: the pose the demonstrations look around from before reaching (so101_pen_in_bowl_wrist episode 7,
# frame 169): tip 16 cm above the table, the wrist camera looking down at the pen and the bowl. portrait:
# standing, the gripper (and its camera) pointing forward and ~15 deg up, tip ~36 cm high: faces, photos.
POSES = {
    "survey": {"shoulder_lift": -48.7, "elbow_flex": 8.3, "wrist_flex": 102.0},
    "portrait": {"shoulder_lift": -30.0, "elbow_flex": -60.0, "wrist_flex": 75.0},
}


def plan_pose(name, start, gentle=False):
    """Go to a named pose and stay (base, wrist roll and gripper keep their angles)."""
    if name not in POSES:
        raise Refused(f"unknown pose '{name}'; known: {', '.join(POSES)}")
    S = {j: float(start[j]) for j in JOINTS}
    return _check(Plan(name, S, [_kf(clamp({**S, **POSES[name]}), speeds=GENTLE if gentle else None)]),
                  f"going to the {name} pose")


def near_pose(name, pose, tol=4.0):
    return all(abs(pose[j] - v) <= tol for j, v in POSES[name].items())


# ---------------------------------------------------------------- body language
# Small moves around wherever the arm is (only when it is up: tip >= 10 cm), back to that pose after.
EXPRESS_MIN_TIP_M = 0.10


def _x_nod(S):
    w = S["wrist_flex"]
    return [_kf(clamp({"wrist_flex": w + d})) for d in (10, -3, 10)]


def _x_tilt(S):        # curious, a question
    return [_kf(clamp({"wrist_roll": S["wrist_roll"] + 18}), hold=0.5)]


def _x_shake(S):       # no, sorry
    p = S["shoulder_pan"]
    return [_kf(clamp({"shoulder_pan": p + d}), lively=True) for d in (8, -8, 8, -8)]


def _x_bounce(S):      # excited
    lift, w = S["shoulder_lift"], S["wrist_flex"]
    return [_kf(clamp({"shoulder_lift": lift + d, "wrist_flex": w - d}), lively=True) for d in (-6, 2, -6, 2)]


def _x_droop(S):       # sad
    return [_kf(clamp({"wrist_flex": S["wrist_flex"] + 18}), speeds=GENTLE, hold=0.8)]


def _x_wiggle(S):      # laughing
    r = S["wrist_roll"]
    return [_kf(clamp({"wrist_roll": r + d}), lively=True) for d in (12, -12, 12, -12)]


EXPRESSIONS = {"nod": _x_nod, "tilt": _x_tilt, "shake": _x_shake, "bounce": _x_bounce, "droop": _x_droop,
               "wiggle": _x_wiggle}


def plan_express(kind, start):
    if kind not in EXPRESSIONS:
        raise Refused(f"unknown expression '{kind}'; known: {', '.join(EXPRESSIONS)}")
    S = {j: float(start[j]) for j in JOINTS}
    if _tip_z(S) < EXPRESS_MIN_TIP_M:
        raise Refused("I am folded down; body language only when I am up")
    return _check(Plan(f"express {kind}", S, EXPRESSIONS[kind](S) + [_kf(S, home=True)]), f"the {kind} move")


# ---------------------------------------------------------------- dancing to a beat
# Every move lasts a whole number of beats (claps half a beat): a move too big to fit its beats at the
# lively speeds is scaled down to fit, so the arm stays on the music instead of lagging behind it.
DANCE_BPM = (80, 130)
DANCE_POSE = {**STAND, "wrist_flex": 20.0}
DANCE_MOVES = {  # name: [(offsets from the dance pose, beats), ...]
    "sway": [({"shoulder_pan": 15, "wrist_roll": -20}, 2), ({"shoulder_pan": -15, "wrist_roll": 20}, 2)],
    "bob": [({"wrist_flex": 20}, 1), ({"wrist_flex": -8}, 1)] * 2,
    "twist": [({"wrist_roll": 35}, 1), ({"wrist_roll": -35}, 1)],
    "clap": [({"gripper": 35}, 0.5), ({"gripper": 0}, 0.5)] * 2,
    "lean": [({"shoulder_lift": 12, "wrist_flex": 15}, 2), ({"shoulder_lift": -8, "wrist_flex": -10}, 2)],
    "groove": [({"shoulder_pan": 12, "wrist_flex": 15}, 1), ({"shoulder_pan": -12, "wrist_flex": -5}, 1)],
}
DANCE_ORDER = ["sway", "bob", "groove", "clap", "twist", "lean", "bob", "sway", "groove", "twist", "clap", "lean"]


def _fit(prev, pose, T, speeds=LIVELY):
    need = _duration(prev, pose, speeds)
    f = min(1.0, T / need) if need > 0 else 1.0
    return {j: prev[j] + (pose[j] - prev[j]) * f for j in prev}


def plan_dance(start, bpm, beats):
    """Stand up (a whole number of beats: the music's intro), dance `beats` beats, then go back to the
    start pose (the music's outro). summary: intro_beats, beats, outro_s for the music."""
    bpm = min(max(float(bpm), DANCE_BPM[0]), DANCE_BPM[1])
    beat = 60.0 / bpm
    S = {j: float(start[j]) for j in JOINTS}
    U = clamp({**S, **DANCE_POSE})
    intro = max(2, math.ceil(_duration(S, U, SPEED) / beat))
    frames, prev, done, k = [_kf(U, dur=intro * beat)], U, 0.0, 0
    while done < beats - 1e-9:
        for off, n in DANCE_MOVES[DANCE_ORDER[k % len(DANCE_ORDER)]]:
            n = math.ceil(max(n, MIN_SEGMENT_S / beat) * 2 - 1e-9) / 2   # half-beat claps: whole beats when fast
            if done + n > beats + 1e-9:
                break
            pose = _fit(prev, clamp({**U, **{j: U[j] + d for j, d in off.items()}}), n * beat)
            frames.append(_kf(pose, lively=True, dur=n * beat))
            prev, done = pose, done + n
        else:
            k += 1
            continue
        break
    frames.append(_kf(U, lively=True, dur=max(1.0, math.ceil(_duration(prev, U, LIVELY) / beat)) * beat))
    up_s = sum(seg[1] for seg in Plan("x", S, frames).segs)
    frames.append(_kf(S, home=True))
    plan = Plan("dance", S, frames, {"bpm": round(bpm, 1), "intro_beats": intro, "beats": round(done, 1)})
    plan.summary["outro_s"] = round(plan.seconds - up_s, 2)
    return _check(plan, "dancing")


def to_action(pose):
    return {f"{j}.pos": float(v) for j, v in pose.items()}


def from_obs(joints):
    return {k.removesuffix(".pos"): float(v) for k, v in joints.items() if k.removesuffix(".pos") in JOINTS}


# ---------------------------------------------------------------- hand-guiding
# "Move me by hand and I won't fall": the servos stay powered (they are position servos, so the arm keeps
# holding its weight). A joint pushed further than its deadband from its goal gets a new goal half a
# deadband behind the hand, so it follows the push against a light, steady resistance; let go and it settles
# there and holds. The deadbands sit above the largest steady error the servos showed while holding still
# in the demonstrations (16,980 still frames: pan 1.5, lift 3.7, elbow 4.2, wrist flex 2.3, roll 1.2 deg,
# gripper 7.1 % while squeezing a pen), so gravity alone should never start a follow. A gravity-torque model
# from the URDF's link masses did not explain those errors (R^2 <= 0.08 per joint: the 1:345 gearboxes hold
# the load), so none is used. Servo gains are left as lerobot configures them.
GUIDE_DEADBAND = {"shoulder_pan": 3.0, "shoulder_lift": 5.0, "elbow_flex": 5.5, "wrist_flex": 3.5,
                  "wrist_roll": 3.0, "gripper": 9.0}


class Guide:
    def __init__(self, start):
        self.goal = {j: float(start[j]) for j in JOINTS}

    def step(self, present):
        """New goal for the measured positions of this tick."""
        for j, db in GUIDE_DEADBAND.items():
            d = present[j] - self.goal[j]
            if abs(d) > db:
                self.goal[j] = present[j] - math.copysign(db / 2, d)
        return dict(self.goal)


# ---------------------------------------------------------------- taught motions (teach / replay)
# Taught by pushing the arm (hand-guiding) or by moving the leader arm (mirroring): the follower's
# measured joints at the control rate. Replayed as recorded after a calm lead-in from wherever the arm
# is, inside the demonstrations' joint envelope (the gripper may open fully) and no faster than the
# lively speeds: a motion taught faster is slowed down as a whole. It may go as low as the recording
# itself did - that height was shown by hand - but never lower; the lead-in keeps the usual clearance.
TAUGHT_MAX_S = 60.0
TAUGHT_MIN_S = 0.5
TAUGHT_LIMITS = {**LIMITS, "gripper": (0.0, 100.0)}
SPEED_FACTORS = (0.5, 1.5)


def _taught_clamp(pose):
    return {j: min(max(v, TAUGHT_LIMITS[j][0]), TAUGHT_LIMITS[j][1]) for j, v in pose.items()}


class Trajectory:
    """A lead-in Plan, then the recorded frames every dt seconds (linear in between). Same interface as
    Plan where robot_server.py and the tests use it: name, seconds, end, summary, at(t), lowest()."""

    def __init__(self, name, lead, frames, dt, floor, summary):
        self.name, self.lead, self.frames, self.dt, self.floor = name, lead, frames, dt, floor
        self.seconds = lead.seconds + (len(frames) - 1) * dt
        self.end = dict(frames[-1])
        self.segs = lead.segs
        self.summary = {"name": name, "seconds": round(self.seconds, 2), **summary}

    def at(self, t):
        if t < self.lead.seconds:
            return self.lead.at(t)
        x = (t - self.lead.seconds) / self.dt
        i = int(x)
        if i >= len(self.frames) - 1:
            return dict(self.end)
        a, b, f = self.frames[i], self.frames[i + 1], x - i
        return {j: a[j] + (b[j] - a[j]) * f for j in a}

    def lowest(self, dt=1 / 30):
        bad = self.lead.lowest(dt)
        if bad:
            return bad
        for k, pose in enumerate(self.frames):
            for name, p in points(pose).items():
                below = self.floor[name] - TOLERANCE_M - p[2]
                if below > 0:
                    return (below, self.lead.seconds + k * self.dt, name)
        return None


def recording_frames(rec, rate=30.0):
    """The recording resampled to `rate` Hz and lightly smoothed (5 frames, servo jitter): poses."""
    ts = np.array([f[0] for f in rec["frames"]], float)
    qs = np.array([f[1] for f in rec["frames"]], float)
    if len(ts) < 2:
        raise Refused("that recording is empty")
    grid = np.arange(0.0, ts[-1] - ts[0] + 1e-9, 1.0 / rate) + ts[0]
    q = np.stack([np.interp(grid, ts, qs[:, i]) for i in range(len(JOINTS))], 1)
    if len(q) >= 5:
        k = np.ones(5) / 5
        pad = np.pad(q, ((2, 2), (0, 0)), mode="edge")
        q = np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(len(JOINTS))], 1)
    return [dict(zip(JOINTS, map(float, row))) for row in q]


def plan_replay(start, rec, speed=1.0):
    """rec: {"name", "frames": [[t, [6 joints]], ...]} as robot_server.py saves it."""
    frames = recording_frames(rec)
    S = {j: float(start[j]) for j in JOINTS}
    floor = {n: min(CLEARANCE_M, min(points(f)[n][2] for f in frames)) for n in ("elbow", "wrist", "tip")}
    clamped = [_taught_clamp(f) for f in frames]
    was_clamped = any(abs(a[j] - b[j]) > 0.5 for a, b in zip(frames, clamped) for j in JOINTS)
    speed = min(max(float(speed or 1.0), SPEED_FACTORS[0]), SPEED_FACTORS[1])
    q = np.array([[f[j] for j in JOINTS] for f in clamped])
    vmax = np.abs(np.diff(q, axis=0)).max(0) * 30.0 if len(q) > 1 else np.zeros(len(JOINTS))
    allowed = min([speed] + [LIVELY[j] / v for j, v in zip(JOINTS, vmax) if v > 1e-6])
    dt = 1.0 / 30.0 / allowed
    lead = _check(Plan("lead-in", S, [_kf(clamped[0], speeds=GENTLE)]), "getting to where that motion starts")
    traj = Trajectory(rec.get("name", "taught"), lead, clamped, dt, floor,
                      {"speed": round(float(allowed), 2), "slowed_down": bool(allowed < speed - 1e-6),
                       "clamped_to_limits": bool(was_clamped), "taught_s": round((len(clamped) - 1) / 30.0, 1)})
    bad = traj.lowest()
    if bad:
        raise Refused(f"replaying that from here would bring my {bad[2]} closer to the table than when "
                      "you taught it")
    return traj


class Follow:
    """Mirroring: the command chases a moving target (the leader arm), each joint no faster than
    `speeds`, inside the taught-motion limits."""

    def __init__(self, start, speeds=LIVELY):
        self.cmd = {j: float(start[j]) for j in JOINTS}
        self.speeds = speeds

    def step(self, target, dt):
        for j in JOINTS:
            goal = _taught_clamp({j: target[j]})[j]
            v = self.speeds[j] * dt
            self.cmd[j] += max(-v, min(v, goal - self.cmd[j]))
        return dict(self.cmd)

    def gap(self, target):
        return max(abs(_taught_clamp({j: target[j]})[j] - self.cmd[j]) for j in JOINTS)


# ---------------------------------------------------------------- following a face
# From the portrait pose: base pan turns toward the face, wrist flex tilts toward it. ex, ey: the face
# centre's offset from the image centre in the upright image, -1..1 (+x right, +y down). Turning right
# is +pan (PARTS), tilting down is +wrist_flex (the gripper points from up, through forward, to down as
# wrist_flex grows). robot_server.py flips a sign if the error keeps growing anyway (a mirrored camera).
TRACK_GAIN = 30.0         # deg/s per unit of image error
TRACK_DEADBAND = 0.08
TRACK_WRIST = (40.0, 102.0)


def track_step(pose, ex, ey, dt, sx=1, sy=1):
    p = dict(pose)
    for j, e, sg in (("shoulder_pan", ex, sx), ("wrist_flex", ey, sy)):
        if abs(e) > TRACK_DEADBAND:
            v = GENTLE[j] * dt
            p[j] += max(-v, min(v, sg * TRACK_GAIN * e * dt))
    p = clamp(p)
    p["wrist_flex"] = min(max(p["wrist_flex"], TRACK_WRIST[0]), TRACK_WRIST[1])
    return p

#!/usr/bin/env python3
"""Offline checks of motions.py: no robot, no GPU.  python3 test_motions.py  (or pytest)

Every gesture and every part move is planned from poses the arm really was in (recorded rest, the
moment of grasping, standing) and sampled at the control rate (30 Hz): joint limits, speed caps,
clearance above the table, the return to the start pose, and what up/down/left/right mean.
"""

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import motions as M  # noqa: E402

URDF = HERE.parent / "adaptive_visual_servo" / "assets" / "SO101" / "so101_new_calib.urdf"
DT = 1 / 30
# Means over the 100 episodes of so101_pen_in_bowl_wrist (observation.state): first frame (rest), the
# frame where the gripper is widest open (reaching for the pen), last frame (back at rest).
REST = dict(zip(M.JOINTS, [-6.6, -99.0, 91.1, 73.4, 88.6, 1.7]))
REACH = dict(zip(M.JOINTS, [-7.1, 13.4, 6.4, 68.2, 83.5, 27.7]))
END = dict(zip(M.JOINTS, [-5.6, -103.4, 93.9, 78.0, 89.2, 1.8]))
STANDING = {**REST, **M.STAND}
STARTS = {"rest": REST, "end": END, "standing": STANDING, "reach": REACH}


def samples(plan):
    n = int(plan.seconds / DT) + 2
    return [plan.at(k * DT) for k in range(n)]


def test_chain_matches_urdf():
    if not URDF.exists():
        return
    joints = {j.get("name"): j for j in ET.parse(URDF).getroot().findall("joint")}
    for name, xyz, rpy in M.CHAIN:
        o = joints[name].find("origin")
        assert tuple(map(float, o.get("xyz").split())) == xyz, name
        assert tuple(map(float, o.get("rpy").split())) == rpy, name


def test_kinematics_match_the_recordings():
    assert abs(M.points(REST)["tip"][2] - 0.005) < 0.01     # resting on the table
    assert abs(M.points(REACH)["tip"][2] - 0.02) < 0.01     # grasping a pen
    assert M.points(STANDING)["tip"][2] > 0.5                # standing tall


def check_plan(plan, start, returns):
    seq = samples(plan)
    for a, b in zip(seq, seq[1:]):
        for j in M.JOINTS:
            assert abs(b[j] - a[j]) / DT <= M.LIVELY[j] * 1.01, (plan.name, j, abs(b[j] - a[j]) / DT)
    for t0, T, a, b, home in plan.segs:
        if not home:
            for j, (lo, hi) in M.LIMITS.items():
                assert lo - 1e-6 <= b[j] <= hi + 1e-6 or b[j] == start[j], (plan.name, j, b[j])
    assert plan.lowest() is None
    if returns:
        assert all(abs(plan.end[j] - start[j]) < 1e-9 for j in M.JOINTS), plan.name
        assert all(abs(seq[-1][j] - start[j]) < 1e-6 for j in M.JOINTS), plan.name


def test_gestures():
    for sname, start in STARTS.items():
        for name, (_, returns) in M.GESTURES.items():
            plan = M.plan_gesture(name, start)
            check_plan(plan, start, returns)
            assert M.MIN_SEGMENT_S <= plan.seconds < 20.0, (name, sname, plan.seconds)


def test_unknown_gesture_is_refused():
    try:
        M.plan_gesture("backflip", REST)
    except M.Refused as e:
        assert "hello" in str(e)
    else:
        raise AssertionError("backflip planned")


def roll_angle_from_above(pose):
    """Heading of the gripper's x axis seen from above (degrees, counter-clockwise positive)."""
    T = __import__("numpy").eye(4)
    for (name, _, _), A in zip(M.CHAIN, M._ORIGINS):
        q = math.radians(pose.get(name, 0.0)) if name in M.JOINTS else 0.0
        R = __import__("numpy").eye(4)
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = math.cos(q), -math.sin(q), math.sin(q), math.cos(q)
        T = T @ A @ R
    return math.degrees(math.atan2(T[1, 0], T[0, 0]))


def test_directions():
    for sname in ("rest", "reach", "standing"):
        start = STARTS[sname]
        for part, (joint, signs) in M.PARTS.items():
            for d in (signs or {"up": 0, "down": 0}):
                try:
                    plan = M.plan_move(start, part, d, 20)
                except M.Refused:
                    assert (sname, part, d) in REFUSALS_OK, (sname, part, d)
                    continue
                check_plan(plan, start, False)
                end, s = plan.end, plan.summary
                assert abs(s["moved"]) <= 20.0 + 1e-6
                tip0, tip1 = M.points(start)["tip"], M.points(end)["tip"]
                if d in ("up", "down"):
                    want = 1 if d == "up" else -1
                    # the tip goes the asked way, unless neither way of turning this joint can do that by
                    # 1 cm (gripper already pointing straight down/up): then the default bend is used
                    best = max(want * (M.points(M.clamp({**start, joint: start[joint] + sg * 20}))["tip"][2]
                                       - tip0[2]) for sg in (1, -1))
                    assert want * (tip1[2] - tip0[2]) > 0 or best < 0.01, (sname, part, d)
                if part == "base" and sname == "reach":
                    assert (tip1[1] - tip0[1]) * (1 if d == "left" else -1) > 0, (sname, d)
                if part == "wrist_roll" and sname == "standing":   # right = clockwise seen from above
                    turn = (roll_angle_from_above(end) - roll_angle_from_above(start) + 180) % 360 - 180
                    assert turn * (1 if d == "left" else -1) > 0, (d, turn)
                if part == "gripper":
                    assert (end["gripper"] - start["gripper"]) * (1 if d == "open" else -1) > 0


# Where a refusal is the right answer: already at a joint limit, or the table is in the way.
REFUSALS_OK = {("rest", "shoulder", "down"), ("rest", "elbow", "down"), ("rest", "wrist", "down"),
               ("rest", "gripper", "close"), ("reach", "shoulder", "down"), ("reach", "elbow", "down"),
               ("reach", "wrist", "down"), ("standing", "gripper", "close")}


def test_table_and_limits():
    # reaching pose: the tip is 2 cm above the table, so nothing may lower it
    for part in ("shoulder", "elbow", "wrist"):
        try:
            plan = M.plan_move(REACH, part, "down", 30)
        except M.Refused as e:
            assert "table" in str(e) or "far" in str(e)
        else:
            assert M.points(plan.end)["tip"][2] >= M.points(REACH)["tip"][2] - M.TOLERANCE_M
    # a big request is capped, and the gripper never opens past the demonstrations
    assert abs(M.plan_move(STANDING, "base", "left", 400).summary["moved"]) <= M.MAX_DEG
    assert M.plan_move(STANDING, "gripper", "open").end["gripper"] == M.LIMITS["gripper"][1]
    for bad in (("knee", "up"), ("base", "up"), ("gripper", "left")):
        try:
            M.plan_move(REST, *bad)
        except M.Refused:
            pass
        else:
            raise AssertionError(bad)


def test_guide():
    """A joint model: the servo pulls toward its goal (stiffness), gravity pulls with at most the largest
    steady error seen in the demos, a hand pushes. Pushed: it follows; let go: it holds; never creeps."""
    g = M.Guide(REST)
    pos = dict(REST)
    k = 0.3                                   # fraction of the error closed per tick (any 0 < k < 1)
    sag = {"shoulder_lift": -3.74, "elbow_flex": 4.18}   # worst measured steady error, signed both ways
    def tick(hand=None):
        goal = g.step(pos)
        for j in M.JOINTS:
            target = goal[j] + sag.get(j, 0.0)
            pos[j] = hand[j] if hand and j in hand else pos[j] + k * (target - pos[j])
        return goal
    for _ in range(300):                      # 10 s alone: gravity must not drag the goal anywhere
        tick()
    assert all(abs(g.goal[j] - REST[j]) < 1e-9 for j in M.JOINTS), g.goal
    hand = dict(pos)
    for _ in range(90):                       # a hand raises the elbow by 40 deg over 3 s
        hand["elbow_flex"] -= 40 / 90
        tick(hand)
    lag = g.goal["elbow_flex"] - hand["elbow_flex"]          # the goal trails the hand by half to one deadband
    assert M.GUIDE_DEADBAND["elbow_flex"] / 2 - 1e-6 <= lag <= M.GUIDE_DEADBAND["elbow_flex"] + 1e-6, lag
    held = g.goal["elbow_flex"]
    for _ in range(300):                      # let go: it settles near the hand's last position and stays
        tick()
    assert g.goal["elbow_flex"] == held
    # springback on release: the trailing distance plus gravity's sag, at most one deadband + 4.18 deg
    assert abs(pos["elbow_flex"] - hand["elbow_flex"]) <= M.GUIDE_DEADBAND["elbow_flex"] + 4.18 + 1e-6


def test_poses_and_expressions():
    for sname, start in STARTS.items():
        for name in M.POSES:
            plan = M.plan_pose(name, start)
            check_plan(plan, start, False)
            assert M.near_pose(name, plan.end, 0.01)
    up = M.plan_pose("portrait", REST).end
    assert M.points(up)["tip"][2] > 0.3
    for kind in M.EXPRESSIONS:
        check_plan(M.plan_express(kind, up), up, True)
        try:
            M.plan_express(kind, REST)          # folded down: no body language
        except M.Refused:
            pass
        else:
            raise AssertionError(kind)


def test_dance_stays_on_the_beat():
    for bpm in (80, 100, 118, 130):
        for sname, start in STARTS.items():
            plan = M.plan_dance(start, bpm, 32)
            check_plan(plan, start, True)
            beat = 60.0 / bpm
            s = plan.summary
            assert s["beats"] == 32
            t = s["intro_beats"] * beat
            assert abs(plan.segs[0][1] - t) < 1e-6           # standing up fills the intro exactly
            for t0, T, a, b, home in plan.segs[1:-1]:        # every dance move: whole (or half) beats
                assert abs(T / beat * 2 - round(T / beat * 2)) < 1e-6, (bpm, T / beat)


def wave_recording(start, seconds=3.0, amp=15.0, rate=30, fast=1.0):   # peak 63 deg/s
    import random
    rnd = random.Random(0)
    frames = []
    for k in range(int(seconds * rate)):
        t = k / rate
        pose = dict(start)
        pose["shoulder_pan"] += amp * math.sin(2 * math.pi * fast * t / seconds * 2)
        frames.append([t, [pose[j] + rnd.uniform(-0.4, 0.4) for j in M.JOINTS]])
    return {"name": "wave", "frames": frames}


def test_replay():
    rec = wave_recording(STANDING)
    for sname, start in STARTS.items():
        traj = M.plan_replay(start, rec)
        check_plan(traj, start, False)
        assert not traj.summary["slowed_down"] and traj.summary["speed"] == 1.0
        seq = [traj.at(k * DT) for k in range(int(traj.seconds / DT) + 2)]
        for a, b in zip(seq, seq[1:]):
            for j in M.JOINTS:
                assert abs(b[j] - a[j]) / DT <= M.LIVELY[j] * 1.01, (sname, j)
        assert abs(traj.at(traj.seconds)["shoulder_pan"] - traj.end["shoulder_pan"]) < 1e-9
    slow = M.plan_replay(REST, rec, speed=0.5)
    assert abs(slow.summary["speed"] - 0.5) < 1e-9 and slow.seconds > M.plan_replay(REST, rec).seconds
    fast = M.plan_replay(REST, wave_recording(STANDING, amp=40, fast=6.0))   # ~500 deg/s: slowed down
    assert fast.summary["slowed_down"] and fast.summary["speed"] < 1.0
    json.dumps(fast.summary)            # the HTTP answer: numpy bools broke /replay (500) on 2026-10-28
    low = {"name": "touch", "frames": [[k / 30, [REACH[j] for j in M.JOINTS]] for k in range(30)]}
    M.plan_replay(REST, low)            # as low as it was taught: fine
    lower = dict(REACH, shoulder_lift=REACH["shoulder_lift"] + 10)   # tip 2.0 -> -1.1 cm
    try:                                # lower than it was taught (a modified file): refused
        traj = M.plan_replay(REST, low)
        traj.frames[10] = lower
        assert traj.lowest() is not None
    except M.Refused:
        pass


def test_follow_and_track():
    f = M.Follow(REST)
    target = dict(REST, shoulder_pan=REST["shoulder_pan"] + 90, gripper=200)
    for _ in range(300):
        cmd = f.step(target, DT)
    assert cmd["gripper"] == 100.0 and cmd["shoulder_pan"] == M.TAUGHT_LIMITS["shoulder_pan"][1]
    f = M.Follow(REST)
    a = f.step(target, DT)
    assert abs(a["shoulder_pan"] - REST["shoulder_pan"]) <= M.LIVELY["shoulder_pan"] * DT + 1e-9
    up = M.plan_pose("portrait", REST).end
    p = M.track_step(up, 0.5, 0.0, DT)
    assert p["shoulder_pan"] > up["shoulder_pan"]               # face on the right: turn right
    p = M.track_step(up, 0.0, 0.5, DT)
    assert p["wrist_flex"] > up["wrist_flex"]                   # face low: tilt down
    p = M.track_step(up, 0.05, -0.05, DT)
    assert p == M.clamp(dict(up)) or all(abs(p[j] - up[j]) < 1e-9 for j in M.JOINTS)   # deadband


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    for sname, start in STARTS.items():
        print(sname, {g: round(M.plan_gesture(g, start).seconds, 1) for g in M.GESTURES})

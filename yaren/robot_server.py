#!/usr/bin/env python3
"""Yaren's body: the MolmoAct2 policy (TensorRT) and the SO-101 arm, kept loaded behind a small HTTP API.

    robot_server.py [--port 8791] [--task-timeout 30] [--no-auto-home]
                    [--dry --dry-root DIR --dry-repo-id ID --dry-episode N]
                    <the lerobot-rollout arguments of ../pen_bowl_molmoact2/3_run_autonomous.sh>

Run it through ./start.sh (GPU0, lelab venv), which builds those arguments the same way the run script
does. Everything on the robot side is lerobot-rollout's own code: build_rollout_context (with
tools/trt/rollout_trt.py's TensorRT patches) loads the policy, connects the arm with the same
max_relative_target, and builds the processors; each control tick is lerobot's send_next_action, as in
--strategy.type=base. What this adds: the task text can change between runs, runs start and stop on
request, and the arm returns to the pose it was in at startup after a run (as the run script does
between --trials attempts). Between runs it can also play gestures and move single parts (motions.py:
planned with the SO-101's kinematics, within the demonstrations' joint range and speeds).

API (127.0.0.1 only - the voice server is the only client):
  GET  /status                   state, current task, timings, last result
  POST /task {task, max_seconds} start a run with this instruction (English, as MolmoAct2 was trained)
  POST /stop {home: false}       stop now; the arm holds where it is (home=true: then go home)
  POST /home                     move to the start pose (rate limited, ~2-4 s)
  POST /gesture {name}           play a gesture of motions.py (hello, nod, ...); not during a run
  POST /move {part, direction, degrees}   move one part of the arm (motions.py); not during a run
  POST /guide {on, level}        hand-guiding: push the arm and it follows, let go and it holds (motions.Guide);
                                 level = stiffness, the fraction of each motor's torque limit (0.1-1, default 0.4)
  POST /teach {action, source, name}   record a motion (start: source hand = hand-guiding, leader = mirroring
                                 the leader arm; stop: save it under name; cancel; list; delete)
  POST /replay {name, speed}     replay a taught motion (motions.plan_replay)
  POST /mirror {on}              the arm copies the leader arm live (motions.Follow)
  POST /follow {on}              face following with the wrist camera from the portrait pose
  POST /pose {name}              go to a named pose (motions.POSES: survey, portrait) and stay
  POST /dance {bpm, beats, plan_only}  a dance on the beat (motions.plan_dance); plan_only: just the timing
  POST /express {kind}           a small body-language move (motions.EXPRESSIONS), only when idle and up
  GET  /camera/<name>.jpg|.mjpg  latest camera frame / MJPEG stream (?upright=1: turned so a face is upright)

--dry: no arm, no camera. The robot is replaced by a stand-in that replays a recorded episode of
so101_pen_in_bowl_wrist and swallows actions, so the policy, the API and the UI can be tested
without moving anything. The leader arm is replaced by that episode's actions (what the leader did).

A task always starts from the start pose: requested anywhere else, the arm goes home first (MolmoAct2
only ever started from there). Motor temperatures are read once a second: new moves are refused, and a
run is stopped and sent home, before a motor gets near the limit the servo itself would cut torque at.
"""

import asyncio
import collections
import concurrent.futures
import json
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools" / "trt"))

import motions  # noqa: E402
import rollout_trt  # noqa: E402

log = logging.getLogger("yaren.robot")
JPEG_QUALITY = 80
HOME_SPEED_DEG_S = 45.0       # homing: straight-line interpolation at this speed (lerobot's own
HOME_MIN_S = 1.5              # return-to-start uses 3 s whatever the distance)
HOME_EPS_DEG = 1.0
AT_HOME_DEG = 3.0             # status "at_home": every joint this close to the start pose
GUIDE_LEVEL = 0.4             # hand-guiding: torque limit as a fraction of the motors' own (RAM register,
GUIDE_LEVEL_MIN = 0.1         # power-on value restored when guiding ends). At 1.0 a hand cannot push the arm.
TASK_HOME_DEG = 10.0          # a task requested further than this from the start pose goes home first
# Margins below each servo's own Max_Temperature_Limit (read from the motor; past it the servo cuts torque
# and the arm falls): warn, refuse new moves, stop a run and go home.
TEMP_WARN, TEMP_BLOCK, TEMP_STOP = 15, 8, 5
MIRROR_ALIGNED_DEG = 3.0      # mirroring starts calmly (GENTLE) until the arm is this close to the leader
FOLLOW_MAX_S = 300.0          # face following ends by itself after 5 minutes,
FOLLOW_LOST_S = 30.0          # or after this long without seeing a face
FACE_EVERY = 3                # face detection every 3rd tick (10 Hz)


def pop_opt(name, default=None, flag=False):
    for i, arg in enumerate(sys.argv):
        if arg == name:
            if flag:
                del sys.argv[i]
                return True
            value = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            return value
        if not flag and arg.startswith(name + "="):
            del sys.argv[i]
            return arg.split("=", 1)[1]
    return default


class DatasetRobot:
    """Stand-in for the SO-101 in --dry mode: same feature schema (taken from an unconnected
    SOFollower built from the same config), observations replayed from a recorded episode."""

    def __init__(self, robot_cfg, repo_id, root, episode, max_frames=450):
        from lerobot.datasets import LeRobotDataset
        from lerobot.robots import make_robot_from_config

        real = make_robot_from_config(robot_cfg)  # never connected: schema only
        self.observation_features = real.observation_features
        self.action_features = real.action_features
        self.name = real.name
        self.robot_type = getattr(real, "robot_type", real.name)
        self.config = robot_cfg
        self.cameras = {}
        ds = LeRobotDataset(repo_id, root=root, episodes=[episode])
        names = ds.meta.features["observation.state"]["names"]
        cams = [k for k, v in self.observation_features.items() if isinstance(v, tuple)]
        self.frames, self.actions = [], []
        for i in range(min(len(ds), max_frames)):
            item = ds[i]
            obs = {n if n.endswith(".pos") else f"{n}.pos": float(v) for n, v in zip(names, item["observation.state"])}
            for cam in cams:
                img = item[f"observation.images.{cam}"]
                obs[cam] = (img * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            self.frames.append(obs)
            self.actions.append({n if n.endswith(".pos") else f"{n}.pos": float(v) for n, v in zip(names, item["action"])})
        self.i = 0
        self.is_connected = False
        self.actions_sent = 0
        self.last_action = None
        log.info("dry robot: %d frames of %s episode %d, cameras %s", len(self.frames), repo_id, episode, cams)

    @property
    def is_calibrated(self):
        return True

    def connect(self, calibrate=True):
        self.is_connected = True

    def get_observation(self):
        time.sleep(0.004)  # roughly the bus read of the real arm
        obs = dict(self.frames[self.i % len(self.frames)])
        self.i += 1
        return obs

    def send_action(self, action):
        self.actions_sent += 1
        self.last_action = {k: float(v) for k, v in action.items()}
        return action

    def disconnect(self):
        self.is_connected = False


class DatasetLeader:
    """Stand-in for the leader arm in --dry mode: the recorded episode's actions, one per call."""

    def __init__(self, robot):
        self.actions = robot.actions
        self.i = 0
        self.is_connected = True

    def get_action(self):
        a = self.actions[self.i % len(self.actions)]
        self.i += 1
        return dict(a)

    def disconnect(self):
        self.is_connected = False


def connect_leader(port, leader_id):
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    leader = SOLeader(SOLeaderTeleopConfig(port=port, id=leader_id))
    leader.connect(calibrate=False)          # never the interactive calibration prompt
    if not leader.is_calibrated:
        leader.disconnect()
        raise motions.Refused("the leader arm's calibration does not match its file; calibrate it with lerobot first")
    return leader


def slug(name):
    out = "".join(c if c.isalnum() else "_" for c in name.strip().lower()).strip("_")
    return out[:40]


class Body:
    """Owns the robot. One control thread does every robot call; the API only sets requests."""

    def __init__(self, ctx, task_timeout, auto_home, dry=False, leader_port=None, leader_id="dash_so101",
                 motions_dir=HERE / "run" / "motions"):
        from lerobot.rollout.strategies.core import send_next_action
        from lerobot.utils.action_interpolator import ActionInterpolator

        self.ctx = ctx
        self.send_next_action = send_next_action
        self.engine = ctx.policy.inference
        self.policy = ctx.policy.policy
        self.robot = ctx.hardware.robot_wrapper
        self.home = dict(ctx.hardware.initial_position)
        self.fps = ctx.runtime.cfg.fps
        self.interp = ActionInterpolator(multiplier=ctx.runtime.cfg.interpolation_multiplier)
        self.task_timeout = task_timeout
        self.auto_home = auto_home
        self.cams = [k for k, v in self.robot.observation_features.items() if isinstance(v, tuple)]
        self.lock = threading.Lock()
        self.state = "idle"            # idle | running | homing | moving (a planned motion) | guiding |
                                       # mirroring | following
        self.request = None            # ("task", text, max_s) | ("stop", home, reason) | ("home",) |
                                       # ("gesture", name, future) | ("move", part, direction, degrees, future)
                                       # | ("guide", on, level, future) | ("teach", action, source, name, future)
                                       # | ("replay", name, speed, future) | ("mirror", on, future) |
                                       # ("follow", on, future) | ("pose", name, gentle, future) |
                                       # ("dance", bpm, beats, plan_only, future) | ("express", kind, future)
        self.task = None
        self.task_started = 0.0
        self.task_max_s = task_timeout
        self.last_result = None
        self.recent_results = collections.deque(maxlen=10)   # so a poller that blinks misses no run
        self.error = None
        self.frames = {}
        self.joints = {}
        self.tick_ms = []
        self.chunk_ms = []
        self.runs = 0
        self.alive = True
        self._home_plan = None
        self._motion = None            # (motions.Plan, perf_counter at its start)
        self.last_motion = None
        self._guide = None             # motions.Guide while hand-guiding
        self.guide_level = GUIDE_LEVEL
        self.guide_load = None         # Present_Load (0.1 % of max torque) while guiding, for tuning
        self._saved_torque = None      # Torque_Limit per motor before guiding lowered it
        inner = getattr(self.robot, "inner", None)
        self.bus = getattr(inner, "bus", None)    # None in --dry
        self.dry = dry
        self.leader, self.leader_port, self.leader_id = None, leader_port, leader_id
        self.motions_dir = Path(motions_dir)
        self._pending_task = None      # (text, max_s): a task waiting for the arm to get home first
        self._then = None              # what starts when the current motion is done: "follow"
        self._rec = None               # the motion being taught: {name, source, t0, frames, armed}
        self.last_recording = None     # kept when teaching ends unsaved (a STOP), so it can still be saved
        self._follow = None            # motions.Follow while mirroring
        self._mirror_aligned = False
        self._mirror_fail = 0
        self._face = None              # face following: {t0, seen_at, rot, ex, ey, sx, sy, grow_x, grow_y, ...}
        self._cascade = None
        self.face_rot = None           # the last rotation that showed an upright face (photos use it)
        self.health = None             # motor temperatures / voltages, once a second
        self.temp_limit = None
        self.health_note = None        # the last warning, for the voice side
        self.engine.reset()
        self.engine.start()

    # ---- requests (API threads) ----
    def submit(self, req):
        with self.lock:
            if req[0] == "express" and self.request is not None:
                # body language never pushes out a real request (a tool call made in the same breath)
                old, req = req, self.request
            else:
                old, self.request = self.request, req
        if old and isinstance(old[-1], concurrent.futures.Future) and not old[-1].done():
            old[-1].set_result({"error": "busy" if old[0] == "express" else "replaced by a newer request before it started"})

    def ask(self, req, timeout=2.0):
        """Submit a request the control thread answers (it plans from the joints of that very tick)."""
        fut = concurrent.futures.Future()
        self.submit(req + (fut,))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {"error": "the arm did not take the request (control loop busy)"}

    def status(self):
        with self.lock:
            el = time.time() - self.task_started if self.state == "running" else None
            tick = np.asarray(self.tick_ms[-90:]) if self.tick_ms else np.zeros(1)
            rec = self._rec
            return {
                "state": self.state, "task": self.task, "elapsed_s": None if el is None else round(el, 1),
                "max_seconds": self.task_max_s, "last_result": self.last_result, "error": self.error,
                "pending_task": self._pending_task[0] if self._pending_task else None,
                "recent_results": list(self.recent_results),
                "motion": self._motion_info(), "last_motion": self.last_motion,
                "guide_level": self.guide_level, "guide_load": self.guide_load,
                "recording": None if rec is None else {
                    "source": rec["source"], "name": rec["name"], "armed": rec["armed"],
                    "seconds": round(rec["frames"][-1][0], 1) if rec["frames"] else 0.0},
                "unsaved_recording_s": self._rec_seconds(self.last_recording),
                "taught": self.taught(), "leader": self.leader is not None,
                "face": None if self._face is None else {k: self._face.get(k) for k in ("seen", "ex", "ey", "rot")},
                "health": self.health_summary(), **self._where(),
                "runs": self.runs, "joints": {k: round(v, 1) for k, v in self.joints.items()},
                "tick_ms_p50": round(float(np.percentile(tick, 50)), 2),
                "tick_ms_max": round(float(tick.max()), 1),
                "chunk_ms_last": round(self.chunk_ms[-1], 1) if self.chunk_ms else None,
                "cameras": self.cams, "fps": self.fps,
            }

    def _where(self):
        """at_home: resting at the start pose. near_pose: the named pose (motions.POSES) it stands in."""
        j = self.joints
        if not j:
            return {"at_home": False, "near_pose": None}
        pose = motions.from_obs(j)
        return {"at_home": all(abs(j[k] - v) <= AT_HOME_DEG for k, v in self.home.items() if k in j),
                "near_pose": next((n for n in motions.POSES if len(pose) == len(motions.JOINTS)
                                   and motions.near_pose(n, pose)), None)}

    def _motion_info(self):
        if self.state != "moving" or not self._motion:
            return None
        plan, t0 = self._motion
        return {**plan.summary, "elapsed_s": round(time.perf_counter() - t0, 1)}

    @staticmethod
    def _rec_seconds(rec):
        return round(rec["frames"][-1][0], 1) if rec and rec["frames"] else None

    def taught(self):
        """Names and lengths of the saved motions (cached on the folder's mtime: status runs at 2 Hz)."""
        try:
            mt = self.motions_dir.stat().st_mtime
        except FileNotFoundError:
            return []
        if getattr(self, "_taught_cache", (None,))[0] != mt:
            out = []
            for f in sorted(self.motions_dir.glob("*.json")):
                try:
                    d = json.loads(f.read_text())
                    out.append({"name": d["name"], "seconds": round(d["frames"][-1][0], 1), "source": d.get("source")})
                except Exception:  # noqa: BLE001 - a broken file is just not listed
                    log.warning("unreadable taught motion %s", f)
            self._taught_cache = (mt, out)
        return self._taught_cache[1]

    def health_summary(self):
        h = self.health
        if not h:
            return None
        hottest = max(h["temp_c"], key=h["temp_c"].get)
        t, lim = h["temp_c"][hottest], h["limit_c"][hottest]
        return {"max_temp_c": t, "hottest": hottest, "limit_c": lim, "temp_c": h["temp_c"],
                "min_volt_v": min(h["volt_v"].values()), "warm": t >= lim - TEMP_WARN,
                "hot": t >= lim - TEMP_BLOCK, "age_s": round(time.time() - h["at"], 1)}

    def _read_health(self):
        if self.bus is None:
            return
        if self.temp_limit is None:
            self.temp_limit = {m: int(self.bus.read("Max_Temperature_Limit", m, normalize=False)) for m in self.bus.motors}
            log.info("servo temperature limits: %s", self.temp_limit)
        temp = {m: int(v) for m, v in self.bus.sync_read("Present_Temperature", normalize=False).items()}
        volt = {m: int(v) / 10 for m, v in self.bus.sync_read("Present_Voltage", normalize=False).items()}
        with self.lock:
            was = self.health_summary()
            self.health = {"temp_c": temp, "volt_v": volt, "limit_c": self.temp_limit, "at": time.time()}
            now = self.health_summary()
            if now["warm"] and not (was and was["warm"]):
                self.health_note = f"{now['hottest']} is warm: {now['max_temp_c']} C (it cuts out at {now['limit_c']} C)"
                log.warning("motor temperature: %s", self.health_note)
            if self.state == "running" and now["max_temp_c"] >= now["limit_c"] - TEMP_STOP:
                log.warning("run stopped: %s at %s C", now["hottest"], now["max_temp_c"])
                self._finish("overheat")
                self._start_home(self.joints)

    def _too_hot(self):
        h = self.health_summary()
        if h and h["hot"]:
            return (f"my {h['hottest']} motor is at {h['max_temp_c']} C (it cuts out at {h['limit_c']} C); "
                    "let me rest and cool down for a few minutes")
        return None

    # ---- control thread ----
    def _end_motion(self, how):
        plan, t0 = self._motion
        then, self._then = self._then, None
        self.last_motion = {"name": plan.name, "ended": how, "seconds": round(time.perf_counter() - t0, 1),
                            "at": time.time()}
        log.info("motion %r ended (%s) after %.1f s", plan.name, how, self.last_motion["seconds"])
        self._motion = None
        self.state = "idle"
        if how == "done" and then == "follow":
            self._start_follow()

    def _plan_for(self, req, pose):
        kind = req[0]
        if kind == "gesture":
            return motions.plan_gesture(req[1], pose)
        if kind == "move":
            return motions.plan_move(pose, *req[1:4])
        if kind == "pose":
            return motions.plan_pose(req[1], pose, gentle=req[2])
        if kind == "dance":
            return motions.plan_dance(pose, req[1], req[2])
        if kind == "express":
            return motions.plan_express(req[1], pose)
        if kind == "replay":
            f = self.motions_dir / f"{slug(req[1])}.json"
            if not f.exists():
                names = ", ".join(m["name"] for m in self.taught()) or "none yet"
                raise motions.Refused(f"I don't know a motion called '{req[1]}' (taught motions: {names})")
            return motions.plan_replay(pose, json.loads(f.read_text()), req[2])
        raise motions.Refused(f"unknown motion request {kind}")

    def _start_motion(self, req, joints, then=None):
        kind, fut = req[0], req[-1]
        if self.state == "running":
            fut.set_result({"error": f"a task run is in progress ('{self.task}'); stop it first"})
            return
        if kind == "express" and self.state != "idle":
            fut.set_result({"error": f"busy ({self.state})"})
            return
        hot = self._too_hot()
        if hot:
            fut.set_result({"error": hot})
            return
        try:
            plan = self._plan_for(req, motions.from_obs(joints))
        except motions.Refused as e:
            fut.set_result({"error": str(e)})
            return
        if kind == "dance" and req[3]:          # plan_only: the music needs the timing first
            fut.set_result({"planned": True, **plan.summary})
            return
        self._end_modes(joints)
        self._pending_task = None
        if self.state == "moving":
            self._end_motion("replaced")
        self._motion = (plan, time.perf_counter())
        self._then = then
        self.state = "moving"
        log.info("motion %r: %.1f s %s", plan.name, plan.seconds, plan.summary)
        fut.set_result({"started": True, **plan.summary})

    # ---- modes: hand-guiding, mirroring, face following; teaching records the first two ----
    def _end_modes(self, joints):
        """End guiding / mirroring / following: the arm holds where it is."""
        if self.state == "guiding":
            self._set_guide(False, joints)
        elif self.state == "mirroring":
            self._stop_mirror(joints)
        elif self.state == "following":
            self._stop_follow("stopped")

    def _end_recording(self):
        rec, self._rec = self._rec, None
        if rec and rec["frames"] and rec["frames"][-1][0] >= motions.TAUGHT_MIN_S:
            self.last_recording = rec
            log.info("recording ended: %.1f s from %s", rec["frames"][-1][0], rec["source"])
        return rec

    def _leader_arm(self):
        if self.leader is None:
            if self.dry:
                self.leader = DatasetLeader(self.robot.inner)
            elif not self.leader_port:
                raise motions.Refused("no leader arm here (LEADER_SERIAL not found when I started)")
            else:
                t = time.perf_counter()
                self.leader = connect_leader(self.leader_port, self.leader_id)
                log.info("leader arm connected on %s in %.1f s", self.leader_port, time.perf_counter() - t)
        return self.leader

    def _start_mirror(self, joints):
        leader = self._leader_arm()
        target = motions.from_obs(leader.get_action())
        pose = motions.from_obs(joints)
        self._follow = motions.Follow(pose, motions.GENTLE)
        self._mirror_aligned = self._follow.gap(target) < MIRROR_ALIGNED_DEG
        if self._mirror_aligned:
            self._follow.speeds = motions.LIVELY
        self._mirror_fail = 0
        self.state = "mirroring"
        log.info("mirroring the leader arm (%.0f deg away)", self._follow.gap(target))

    def _stop_mirror(self, joints):
        self._end_recording()
        self._follow = None
        self.state = "idle"
        self.robot.send_action(motions.to_action(motions.from_obs(joints)))   # hold right here
        log.info("mirroring off")

    def _mirror_tick(self, joints, dt):
        try:
            target = motions.from_obs(self.leader.get_action())
            self._mirror_fail = 0
        except Exception as e:  # noqa: BLE001 - a missed read: hold this tick, give up after 5
            self._mirror_fail += 1
            if self._mirror_fail >= 5:
                log.error("leader arm not answering (%s): mirroring off", e)
                self.error = f"leader arm not answering: {e}"
                self._stop_mirror(joints)
            return
        cmd = self._follow.step(target, dt)
        if not self._mirror_aligned and self._follow.gap(target) < MIRROR_ALIGNED_DEG:
            self._mirror_aligned = True
            self._follow.speeds = motions.LIVELY
            log.info("mirroring: caught up with the leader arm")
        self.robot.send_action(motions.to_action(cmd))

    def _teach(self, req, joints):
        _, action, source, name, fut = req
        res = {}
        try:
            if action == "start":
                if self.state == "running":
                    raise motions.Refused(f"a task run is in progress ('{self.task}'); stop it first")
                hot = self._too_hot()
                if hot:
                    raise motions.Refused(hot)
                if self.state == "moving":
                    self._end_motion("replaced")
                self._pending_task = None
                if source == "leader":
                    if self.state != "mirroring":
                        self._end_modes(joints)
                        self._start_mirror(joints)
                else:
                    source = "hand"
                    if self.state != "guiding":
                        self._end_modes(joints)
                        self._set_guide(True, joints)
                        if self.state != "guiding":
                            raise motions.Refused("hand-guiding did not start")
                self._rec = {"name": (name or "").strip(), "source": source, "t0": None, "frames": [],
                             "armed": source == "hand" or self._mirror_aligned}
                res = {"recording": True, "source": source, "max_seconds": motions.TAUGHT_MAX_S}
            elif action in ("stop", "cancel"):
                rec = self._end_recording() if self._rec else None
                if action == "cancel":
                    self.last_recording = None
                    self._end_modes(joints)
                    res = {"cancelled": True}
                else:
                    rec = rec if rec and rec["frames"] and rec["frames"][-1][0] >= motions.TAUGHT_MIN_S else self.last_recording
                    self._end_modes(joints)
                    if not rec:
                        raise motions.Refused("nothing was recorded (or it was shorter than half a second)")
                    nm = (name or rec["name"] or "").strip() or f"motion {len(self.taught()) + 1}"
                    if not slug(nm):
                        raise motions.Refused(f"'{nm}' is not a usable name")
                    self.motions_dir.mkdir(parents=True, exist_ok=True)
                    data = {"name": nm, "source": rec["source"], "recorded_at": time.time(), "fps": self.fps,
                            "frames": rec["frames"]}
                    (self.motions_dir / f"{slug(nm)}.json").write_text(json.dumps(data))
                    self.last_recording = None
                    res = {"saved": nm, "seconds": round(rec["frames"][-1][0], 1), "source": rec["source"]}
                    log.info("taught motion saved: %s (%.1f s, %s)", nm, rec["frames"][-1][0], rec["source"])
            elif action == "list":
                res = {"motions": self.taught()}
            elif action == "delete":
                f = self.motions_dir / f"{slug(name or '')}.json"
                if not name or not f.exists():
                    raise motions.Refused(f"no taught motion called '{name}'")
                f.unlink()
                res = {"deleted": name}
            else:
                raise motions.Refused(f"unknown teach action '{action}'")
        except motions.Refused as e:
            res = {"error": str(e)}
        except Exception as e:  # noqa: BLE001 - e.g. the leader arm's port
            log.exception("teach %s failed", action)
            res = {"error": f"{type(e).__name__}: {e}"}
        fut.set_result(res)

    def _record_tick(self, joints):
        rec = self._rec
        if rec is None or not rec["armed"]:
            if rec is not None and self.state == "mirroring" and self._mirror_aligned:
                rec["armed"] = True
            return
        now = time.perf_counter()
        if rec["t0"] is None:
            rec["t0"] = now
        rec["frames"].append([round(now - rec["t0"], 4), [round(joints[f"{j}.pos"], 2) for j in motions.JOINTS]])
        if rec["frames"][-1][0] >= motions.TAUGHT_MAX_S:
            log.info("recording reached %.0f s: stopped (the arm stays in %s)", motions.TAUGHT_MAX_S, self.state)
            self._end_recording()

    # face following
    def _start_follow(self):
        if self._cascade is None:
            self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        now = time.perf_counter()
        self._face = {"t0": now, "seen_at": now, "seen": False, "rot": self.face_rot, "ex": 0.0, "ey": 0.0,
                      "sx": 1, "sy": 1, "grow_x": 0, "grow_y": 0, "last": None, "try": 0}
        self.state = "following"
        log.info("face following on")

    def _stop_follow(self, how):
        self._face = None
        self.state = "idle"
        log.info("face following off (%s)", how)

    def detect_face(self, rgb, rots, cascade=None):
        """Largest frontal face in any of these rotations (Haar: upright faces only): (rot, ex, ey) or None."""
        cascade = cascade or self._cascade
        small = cv2.cvtColor(cv2.resize(rgb, (320, 240)), cv2.COLOR_RGB2GRAY)
        for rot in rots:
            img = small if rot == 0 else cv2.rotate(small, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                                              270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot])
            faces = cascade.detectMultiScale(img, scaleFactor=1.15, minNeighbors=5, minSize=(24, 24))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                H, W = img.shape
                return rot, (x + w / 2) / (W / 2) - 1, (y + h / 2) / (H / 2) - 1
        return None

    def _follow_tick(self, obs, joints, n, dt):
        f = self._face
        now = time.perf_counter()
        if now - f["t0"] > FOLLOW_MAX_S or now - f["seen_at"] > FOLLOW_LOST_S:
            self._stop_follow("time limit" if now - f["t0"] > FOLLOW_MAX_S else "no face for a while")
            return
        cam = self.cams[0] if self.cams else None
        if cam and cam in obs and n % FACE_EVERY == 0:
            rots = [f["rot"]] if f["rot"] is not None else [(0, 180, 90, 270)[f["try"] % 4]]
            f["try"] += 1
            hit = self.detect_face(obs[cam], rots)
            if hit:
                rot, ex, ey = hit
                if f["rot"] is None:
                    log.info("face found (image turned %d deg)", rot)
                f["rot"], self.face_rot = rot, rot
                # a sign that makes the error grow, update after update, is wrong (a mirrored image)
                if f["last"] is not None:
                    for k, e, last in (("x", ex, f["last"][0]), ("y", ey, f["last"][1])):
                        if abs(e) > 0.25 and abs(e) > abs(last) + 0.02:
                            f["grow_" + k] += 1
                            if f["grow_" + k] >= 8:
                                f["s" + k] *= -1
                                f["grow_" + k] = 0
                                log.info("face following: flipped the %s direction", k)
                        else:
                            f["grow_" + k] = 0
                f.update(seen=True, ex=ex, ey=ey, seen_at=now, last=(ex, ey))
            elif now - f["seen_at"] > 1.0:
                f.update(seen=False, ex=0.0, ey=0.0, last=None)
                if now - f["seen_at"] > 5.0:
                    f["rot"] = None          # look in every orientation again
        pose = motions.track_step(f.get("cmd") or motions.from_obs(joints), f["ex"], f["ey"], dt, f["sx"], f["sy"])
        f["cmd"] = pose
        self.robot.send_action(motions.to_action(pose))

    def _follow_request(self, req, joints):
        on, fut = req[1], req[-1]
        if not on:
            if self.state == "following":
                self._stop_follow("asked")
            elif self._then == "follow":
                self._then = None
            fut.set_result({"following": False})
            return
        if self.state == "following":
            fut.set_result({"following": True})
            return
        if not self.cams:
            fut.set_result({"error": "no camera"})
            return
        if motions.near_pose("portrait", motions.from_obs(joints)):
            if self.state == "running":
                fut.set_result({"error": f"a task run is in progress ('{self.task}'); stop it first"})
                return
            if self.state == "moving":
                self._end_motion("replaced")
            self._end_modes(joints)
            self._pending_task = None
            self._start_follow()
            fut.set_result({"following": True})
        else:                                   # stand up and face forward first, then follow
            inner = concurrent.futures.Future()
            self._start_motion(("pose", "portrait", False, inner), joints, then="follow")
            r = inner.result()
            fut.set_result({"following": True, "getting_up_s": r.get("seconds")} if r.get("started") else r)

    def _soften(self, level):
        """Lower every motor's torque limit to `level` of its own value (Torque_Limit is a RAM register)."""
        if self.bus is None:
            return
        if self._saved_torque is None:
            self._saved_torque = {m: int(self.bus.read("Torque_Limit", m, normalize=False)) for m in self.bus.motors}
            log.info("torque limits before guiding: %s", self._saved_torque)
        for m, full in self._saved_torque.items():
            self.bus.write("Torque_Limit", m, max(1, int(round(full * level))), normalize=False)

    def _restore_torque(self):
        if self.bus is None or self._saved_torque is None:
            return
        for m, full in self._saved_torque.items():
            self.bus.write("Torque_Limit", m, full, normalize=False)
        log.info("torque limits restored: %s", self._saved_torque)
        self._saved_torque = None

    def _set_guide(self, on, joints, fut=None, level=None):
        res = {"guiding": bool(on)}
        try:
            if on and self.state == "running":
                res = {"error": f"a task run is in progress ('{self.task}'); stop it first"}
            elif on:
                if level is not None:
                    self.guide_level = min(1.0, max(GUIDE_LEVEL_MIN, float(level)))
                if self.state == "moving":
                    self._end_motion("replaced")
                if self.state != "guiding":
                    self._guide = motions.Guide(motions.from_obs(joints))
                    log.info("hand-guiding on, stiffness %.0f %%", self.guide_level * 100)
                self._soften(self.guide_level)
                self.state = "guiding"
                res["level"] = self.guide_level
            elif self.state == "guiding":
                self._end_recording()
                self._guide = None
                self.guide_load = None
                self.state = "idle"
                self.robot.send_action(motions.to_action(motions.from_obs(joints)))   # hold right here, so
                self._restore_torque()                                                # full torque snaps nothing
                log.info("hand-guiding off")
        except Exception as e:  # noqa: BLE001 - a failed register write: full torque, not guiding
            log.exception("hand-guiding switch failed")
            self._guide, self.state = None, "idle"
            try:
                self._restore_torque()
            except Exception:  # noqa: BLE001
                log.exception("could not restore torque limits (they come back at power-off)")
            res = {"error": f"{type(e).__name__}: {e}"}
        if fut:
            fut.set_result(res)

    def _finish(self, how):
        dur = round(time.time() - self.task_started, 1)
        self.last_result = {"task": self.task, "ended": how, "seconds": dur, "at": time.time()}
        self.recent_results.append(self.last_result)
        log.info("run ended (%s) after %.1f s: %s", how, dur, self.task)
        self.engine.reset()
        self.interp.reset()
        self.state = "idle"

    def _start_home(self, joints):
        start = {k: joints[k] for k in self.home if k in joints}
        dist = max((abs(self.home[k] - start[k]) for k in start), default=0.0)
        self._home_plan = (start, time.perf_counter(), max(HOME_MIN_S, dist / HOME_SPEED_DEG_S))
        self.state = "homing"

    def _begin_task(self, text, max_s):
        self.engine._task = text          # the sync engine passes this to the policy every tick
        self.engine.reset()
        self.interp.reset()
        self.task, self.task_max_s, self.task_started = text, max_s, time.time()
        self.error = None
        self.runs += 1
        self.state = "running"
        log.info("run %d: %r (max %.0f s)", self.runs, text, max_s)

    def _mirror_request(self, req, joints):
        on, fut = req[1], req[-1]
        try:
            if on:
                if self.state == "running":
                    raise motions.Refused(f"a task run is in progress ('{self.task}'); stop it first")
                hot = self._too_hot()
                if hot:
                    raise motions.Refused(hot)
                if self.state != "mirroring":
                    if self.state == "moving":
                        self._end_motion("replaced")
                    self._end_modes(joints)
                    self._pending_task = None
                    self._start_mirror(joints)
                fut.set_result({"mirroring": True, "catching_up": not self._mirror_aligned})
            else:
                if self.state == "mirroring":
                    self._stop_mirror(joints)
                fut.set_result({"mirroring": False})
        except motions.Refused as e:
            fut.set_result({"error": str(e)})
        except Exception as e:  # noqa: BLE001 - e.g. the leader arm's port
            log.exception("mirroring failed to start")
            fut.set_result({"error": f"{type(e).__name__}: {e}"})

    def _apply_request(self, req, joints):
        kind = req[0]
        if kind == "guide":
            self._set_guide(req[1], joints, req[3], req[2])
            return
        if kind == "teach":
            self._teach(req, joints)
            return
        if kind == "mirror":
            self._mirror_request(req, joints)
            return
        if kind == "follow":
            self._follow_request(req, joints)
            return
        if kind in ("gesture", "move", "pose", "dance", "express", "replay"):
            self._start_motion(req, joints)
            return
        # task / stop / home end every mode; the arm holds where it is first
        self._end_modes(joints)
        if self.state == "moving":
            self._end_motion("stopped" if kind == "stop" else "replaced")
        self._pending_task = None
        if kind == "task":
            _, text, max_s, fut = req
            if self.state == "running":
                self._finish("replaced")
            hot = self._too_hot()
            if hot:
                fut.set_result({"error": hot})
                return
            far = max((abs(joints[k] - self.home[k]) for k in self.home if k in joints), default=0.0)
            if far > TASK_HOME_DEG:     # MolmoAct2 only ever started from the start pose
                self._pending_task = (text, max_s)
                self._start_home(joints)
                log.info("task %r waits: going home first (%.0f deg away)", text, far)
                fut.set_result({"started": True, "task": text, "going_home_first": True})
            else:
                self._begin_task(text, max_s)
                fut.set_result({"started": True, "task": text})
        elif kind == "stop":
            if self.state == "running":
                self._finish(req[2] or "stopped")
            self.state = "idle"
            if req[1]:
                self._start_home(joints)
        elif kind == "home":
            if self.state == "running":
                self._finish("stopped")
            self._start_home(joints)

    def warmup(self):
        """Predict one chunk on the live observation and throw it away (nothing is sent), so the
        first real run does not pay the ~300 ms first-call cost at its first tick."""
        from lerobot.utils.constants import OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame

        t = time.perf_counter()
        obs_p = self.ctx.processors.robot_observation_processor(self.robot.get_observation())
        for _ in range(2):
            self.engine.get_action(build_dataset_frame(self.ctx.data.dataset_features, obs_p, prefix=OBS_STR))
            self.engine.reset()
        log.info("policy warm-up: %.0f ms (no action sent)", (time.perf_counter() - t) * 1000)

    def loop(self):
        period = 1.0 / self.fps
        n = 0
        try:
            self.warmup()
        except Exception:  # noqa: BLE001
            log.exception("warm-up failed (the first run will just be slower)")
        while self.alive:
            t0 = time.perf_counter()
            try:
                obs = self.robot.get_observation()
                joints = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
                with self.lock:
                    req, self.request = self.request, None
                    if req:
                        self._apply_request(req, joints)
                    state = self.state
                if state == "running":
                    if time.time() - self.task_started >= self.task_max_s:
                        with self.lock:
                            self._finish("timeout")
                            if self.auto_home:
                                self._start_home(joints)
                    else:
                        before = len(self.policy._action_queue)
                        t_a = time.perf_counter()
                        obs_p = self.ctx.processors.robot_observation_processor(obs)
                        self.engine.notify_observation(obs_p)
                        self.send_next_action(obs_p, obs, self.ctx, self.interp)
                        if before == 0:  # this tick predicted a new chunk
                            self.chunk_ms.append((time.perf_counter() - t_a) * 1000)
                            self.chunk_ms = self.chunk_ms[-50:]
                elif state == "homing":
                    start, t_h, dur = self._home_plan
                    a = min(1.0, (time.perf_counter() - t_h) / dur)
                    self.robot.send_action({k: start[k] * (1 - a) + self.home[k] * a for k in start})
                    if a >= 1.0 and all(abs(joints[k] - self.home[k]) < HOME_EPS_DEG for k in start) \
                            or time.perf_counter() - t_h > dur + 2.0:
                        with self.lock:
                            if self.state == "homing":
                                self.state = "idle"
                                if self._pending_task:
                                    text, max_s = self._pending_task
                                    self._pending_task = None
                                    self._begin_task(text, max_s)
                elif state == "moving":
                    plan, t_m = self._motion
                    t = time.perf_counter() - t_m
                    self.robot.send_action(motions.to_action(plan.at(t)))
                    if t >= plan.seconds:
                        with self.lock:
                            if self.state == "moving":
                                self._end_motion("done")
                elif state == "mirroring":
                    with self.lock:
                        self._mirror_tick(joints, period)
                        self._record_tick(joints)
                elif state == "following":
                    with self.lock:
                        if self._face is not None:
                            self._follow_tick(obs, joints, n, period)
                elif state == "guiding":
                    with self.lock:
                        self._record_tick(joints)
                    goal = self._guide.step(motions.from_obs(joints))
                    self.robot.send_action(motions.to_action(goal))
                    if self.bus is not None and n % 3 == 0:
                        load = self.bus.sync_read("Present_Load", normalize=False)
                        with self.lock:
                            self.guide_load = load
                        if n % 30 == 0:
                            log.info("guide %.0f%% pos %s goal %s load %s", self.guide_level * 100,
                                     {k.removesuffix('.pos')[:5]: round(v, 1) for k, v in joints.items()},
                                     {k[:5]: round(v, 1) for k, v in goal.items()}, load)
                # idle: send nothing, the servos hold their last goal
                if self.bus is not None and n % 30 == 15:
                    try:
                        self._read_health()
                    except Exception as e:  # noqa: BLE001 - a missed read: next second
                        if n % 900 == 15:
                            log.warning("motor health read failed: %s", e)
                frames = {}
                if n % 2 == 0:   # 15 fps preview
                    for cam in self.cams:
                        if cam in obs:
                            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(obs[cam], cv2.COLOR_RGB2BGR),
                                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                            if ok:
                                frames[cam] = buf.tobytes()
                with self.lock:
                    self.joints = joints
                    self.frames.update(frames)
            except Exception as e:  # noqa: BLE001 - keep the arm loop alive, stop the run
                log.exception("control loop error")
                with self.lock:
                    self.error = f"{type(e).__name__}: {e}"
                    if self.state == "running":
                        self._finish("error")
                    if self.state == "moving":
                        self._end_motion("error")
                    if self._guide is not None:
                        self._guide = None
                        try:
                            self._restore_torque()
                        except Exception:  # noqa: BLE001
                            log.exception("could not restore torque limits (they come back at power-off)")
                    self._end_recording()
                    self._follow, self._face, self._pending_task, self._then = None, None, None, None
                    self.state = "idle"     # a failed homing, gesture, guiding, mirroring...: hold, report, wait
                time.sleep(0.2)
            dt = time.perf_counter() - t0
            with self.lock:
                self.tick_ms.append(dt * 1000)
                self.tick_ms = self.tick_ms[-300:]
            n += 1
            if period - dt > 0:
                time.sleep(period - dt)

    def shutdown(self):
        from lerobot.rollout.strategies.core import RolloutStrategy

        self.alive = False
        time.sleep(0.2)
        try:
            self._restore_torque()       # full torque before the trip back to the start pose
        except Exception:  # noqa: BLE001
            log.exception("could not restore torque limits")
        try:
            if self.leader is not None and not self.dry:
                self.leader.disconnect()
        except Exception:  # noqa: BLE001
            log.exception("leader arm disconnect failed")
        try:
            self.engine.stop()
        finally:
            inner = self.ctx.hardware.robot_wrapper.inner
            if inner.is_connected:
                if self.ctx.runtime.cfg.return_to_initial_position:
                    log.info("returning to the start pose before disconnecting")
                    RolloutStrategy._return_to_initial_position(self.ctx.hardware)
                inner.disconnect()


def build_body():
    port = int(pop_opt("--port", "8791"))
    timeout = float(pop_opt("--task-timeout", "30"))
    auto_home = not pop_opt("--no-auto-home", flag=True)
    dry = pop_opt("--dry", flag=True)
    dry_root = pop_opt("--dry-root")
    dry_repo = pop_opt("--dry-repo-id", "fouad1233/so101_pen_in_bowl_wrist")
    dry_ep = int(pop_opt("--dry-episode", "95"))
    leader_port = pop_opt("--leader-port")
    leader_id = pop_opt("--leader-id", "dash_so101")
    # a motion taught in --dry is the replayed episode (a pen pick down to the table), never one to replay
    # on the real arm: dry runs keep theirs apart
    motions_dir = pop_opt("--motions-dir", str(HERE / "run" / ("motions_dry" if dry else "motions")))

    import lerobot.rollout.context as C
    from lerobot.configs import parser
    from lerobot.rollout import RolloutConfig

    build = rollout_trt.install("fp16", None)
    if dry:
        C.make_robot_from_config = lambda rcfg: DatasetRobot(rcfg, dry_repo, dry_root, dry_ep)

    @parser.wrap()
    def parse(cfg: RolloutConfig):
        return cfg

    cfg = parse()
    ctx = build(cfg, threading.Event())
    return Body(ctx, timeout, auto_home, bool(dry), leader_port, leader_id, motions_dir), port, dry


class TaskReq(BaseModel):
    task: str
    max_seconds: float | None = None


class GestureReq(BaseModel):
    name: str


class MoveReq(BaseModel):
    part: str
    direction: str
    degrees: float | None = None   # gripper: percentage points of its opening; None: open/close fully


class GuideReq(BaseModel):
    on: bool = True
    level: float | None = None     # stiffness: fraction of the motors' torque limit, 0.1-1


class TeachReq(BaseModel):
    action: Literal["start", "stop", "cancel", "list", "delete"]
    source: Literal["hand", "leader"] | None = None
    name: str | None = None


class ReplayReq(BaseModel):
    name: str
    speed: float | None = None


class OnReq(BaseModel):
    on: bool = True


class PoseReq(BaseModel):
    name: str
    gentle: bool = False


class DanceReq(BaseModel):
    bpm: float = 110
    beats: int = 32
    plan_only: bool = False


class ExpressReq(BaseModel):
    kind: str


class StopReq(BaseModel):
    home: bool = False
    reason: Literal["stopped", "done"] | None = None   # "done": the voice side's camera check ended it


def make_app(body, dry):
    @asynccontextmanager
    async def lifespan(app):
        th = threading.Thread(target=body.loop, name="control", daemon=True)
        th.start()
        yield
        body.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.get("/status")
    def status():
        s = body.status()
        s["dry"] = dry
        if dry:
            s["dry_actions_sent"] = body.robot.inner.actions_sent
        return s

    @app.post("/task")
    def task(req: TaskReq):
        text = req.task.strip()
        if not text:
            raise HTTPException(400, "empty task")
        return body.ask(("task", text, float(req.max_seconds or body.task_timeout)))

    @app.post("/stop")
    def stop(req: StopReq | None = None):
        body.submit(("stop", bool(req and req.home), req.reason if req else None))
        return {"ok": True}

    @app.post("/home")
    def home():
        body.submit(("home",))
        return {"ok": True}

    @app.post("/gesture")
    def gesture(req: GestureReq):
        return body.ask(("gesture", req.name.strip().lower()))

    @app.post("/move")
    def move(req: MoveReq):
        return body.ask(("move", req.part.strip().lower(), req.direction.strip().lower(), req.degrees))

    @app.post("/guide")
    def guide(req: GuideReq):
        return body.ask(("guide", req.on, req.level))

    @app.post("/teach")
    def teach(req: TeachReq):
        return body.ask(("teach", req.action, req.source, req.name), timeout=6.0)   # may connect the leader

    @app.post("/replay")
    def replay(req: ReplayReq):
        return body.ask(("replay", req.name, req.speed))

    @app.post("/mirror")
    def mirror(req: OnReq):
        return body.ask(("mirror", req.on), timeout=6.0)

    @app.post("/follow")
    def follow(req: OnReq):
        return body.ask(("follow", req.on))

    @app.post("/pose")
    def pose(req: PoseReq):
        return body.ask(("pose", req.name.strip().lower(), req.gentle))

    @app.post("/dance")
    def dance(req: DanceReq):
        return body.ask(("dance", req.bpm, max(4, min(64, req.beats)), req.plan_only))

    @app.post("/express")
    def express(req: ExpressReq):
        return body.ask(("express", req.kind.strip().lower()))

    photo_cascade, photo_lock = [], threading.Lock()

    def frame(name):
        with body.lock:
            return body.frames.get(name)

    @app.get("/camera/{name}.jpg")
    def jpg(name: str, upright: bool = False):
        f = frame(name)
        if f is None:
            raise HTTPException(404, f"no frame from camera '{name}' (cameras: {body.cams})")
        headers = {"Cache-Control": "no-store"}
        if upright:     # a photo: turned so a face in it stands upright (else as the last face was)
            img = cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR)
            with photo_lock:
                if not photo_cascade:
                    photo_cascade.append(cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"))
                hit = body.detect_face(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (0, 180, 90, 270), photo_cascade[0])
            rot = hit[0] if hit else (body.face_rot or 0)
            if rot:
                img = cv2.rotate(img, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                       270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot])
            f = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
            headers.update({"X-Rotation": str(rot), "X-Face": "1" if hit else "0"})
        return Response(f, media_type="image/jpeg", headers=headers)

    @app.get("/camera/{name}.mjpg")
    async def mjpg(name: str):
        if name not in body.cams:
            raise HTTPException(404, f"unknown camera '{name}'")

        async def gen():
            last = None
            while True:
                f = frame(name)
                if f is not None and f is not last:
                    last = f
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
                await asyncio.sleep(1 / 15)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    body, port, dry = build_body()
    uvicorn.run(make_app(body, dry), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

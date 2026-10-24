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
  POST /guide {on}               hand-guiding: push the arm and it follows, let go and it holds (motions.Guide)
  GET  /camera/<name>.jpg|.mjpg  latest camera frame / MJPEG stream

--dry: no arm, no camera. The robot is replaced by a stand-in that replays a recorded episode of
so101_pen_in_bowl_wrist and swallows actions, so the policy, the API and the UI can be tested
without moving anything.
"""

import asyncio
import collections
import concurrent.futures
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
        self.frames = []
        for i in range(min(len(ds), max_frames)):
            item = ds[i]
            obs = {n if n.endswith(".pos") else f"{n}.pos": float(v) for n, v in zip(names, item["observation.state"])}
            for cam in cams:
                img = item[f"observation.images.{cam}"]
                obs[cam] = (img * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            self.frames.append(obs)
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


class Body:
    """Owns the robot. One control thread does every robot call; the API only sets requests."""

    def __init__(self, ctx, task_timeout, auto_home):
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
        self.state = "idle"            # idle | running | homing | moving (a gesture or part move) | guiding
        self.request = None            # ("task", text, max_s) | ("stop", home, reason) | ("home",) |
                                       # ("gesture", name, future) | ("move", part, direction, degrees, future)
                                       # | ("guide", on, future)
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
        self._chunks_before = 0
        self.engine.reset()
        self.engine.start()

    # ---- requests (API threads) ----
    def submit(self, req):
        with self.lock:
            old, self.request = self.request, req
        if old and isinstance(old[-1], concurrent.futures.Future) and not old[-1].done():
            old[-1].set_result({"error": "replaced by a newer request before it started"})

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
            return {
                "state": self.state, "task": self.task, "elapsed_s": None if el is None else round(el, 1),
                "max_seconds": self.task_max_s, "last_result": self.last_result, "error": self.error,
                "recent_results": list(self.recent_results),
                "motion": self._motion_info(), "last_motion": self.last_motion,
                "runs": self.runs, "joints": {k: round(v, 1) for k, v in self.joints.items()},
                "tick_ms_p50": round(float(np.percentile(tick, 50)), 2),
                "tick_ms_max": round(float(tick.max()), 1),
                "chunk_ms_last": round(self.chunk_ms[-1], 1) if self.chunk_ms else None,
                "cameras": self.cams, "fps": self.fps,
            }

    def _motion_info(self):
        if self.state != "moving" or not self._motion:
            return None
        plan, t0 = self._motion
        return {**plan.summary, "elapsed_s": round(time.perf_counter() - t0, 1)}

    # ---- control thread ----
    def _end_motion(self, how):
        plan, t0 = self._motion
        self.last_motion = {"name": plan.name, "ended": how, "seconds": round(time.perf_counter() - t0, 1),
                            "at": time.time()}
        log.info("motion %r ended (%s) after %.1f s", plan.name, how, self.last_motion["seconds"])
        self._motion = None
        self.state = "idle"

    def _start_motion(self, req, joints):
        kind, fut = req[0], req[-1]
        if self.state == "running":
            fut.set_result({"error": f"a task run is in progress ('{self.task}'); stop it first"})
            return
        try:
            pose = motions.from_obs(joints)
            plan = motions.plan_gesture(req[1], pose) if kind == "gesture" else motions.plan_move(pose, *req[1:4])
        except motions.Refused as e:
            fut.set_result({"error": str(e)})
            return
        if self.state == "moving":
            self._end_motion("replaced")
        self._motion = (plan, time.perf_counter())
        self.state = "moving"
        log.info("motion %r: %.1f s %s", plan.name, plan.seconds, plan.summary)
        fut.set_result({"started": True, **plan.summary})

    def _set_guide(self, on, joints, fut=None):
        res = {"guiding": bool(on)}
        if on and self.state == "running":
            res = {"error": f"a task run is in progress ('{self.task}'); stop it first"}
        elif on:
            if self.state == "moving":
                self._end_motion("replaced")
            if self.state != "guiding":
                self._guide = motions.Guide(motions.from_obs(joints))
                log.info("hand-guiding on")
            self.state = "guiding"
        elif self.state == "guiding":
            self._guide = None
            self.state = "idle"
            log.info("hand-guiding off")
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

    def _apply_request(self, req, joints):
        kind = req[0]
        if kind == "guide":
            self._set_guide(req[1], joints, req[2])
            return
        if self.state == "guiding":    # anything else ends hand-guiding; the arm holds where it is
            self._set_guide(False, joints)
        if kind in ("gesture", "move"):
            self._start_motion(req, joints)
            return
        if self.state == "moving":
            self._end_motion("stopped" if kind == "stop" else "replaced")
        if kind == "task":
            if self.state == "running":
                self._finish("replaced")
            _, text, max_s = req
            self.engine._task = text          # the sync engine passes this to the policy every tick
            self.engine.reset()
            self.interp.reset()
            self.task, self.task_max_s, self.task_started = text, max_s, time.time()
            self.error = None
            self.runs += 1
            self.state = "running"
            log.info("run %d: %r (max %.0f s)", self.runs, text, max_s)
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
                elif state == "moving":
                    plan, t_m = self._motion
                    t = time.perf_counter() - t_m
                    self.robot.send_action(motions.to_action(plan.at(t)))
                    if t >= plan.seconds:
                        with self.lock:
                            if self.state == "moving":
                                self._end_motion("done")
                elif state == "guiding":
                    self.robot.send_action(motions.to_action(self._guide.step(motions.from_obs(joints))))
                # idle: send nothing, the servos hold their last goal
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
                    self._guide = None
                    self.state = "idle"     # a failed homing, gesture or guiding stops too: hold, report, wait
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
    return Body(ctx, timeout, auto_home), port, dry


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
        body.submit(("task", text, float(req.max_seconds or body.task_timeout)))
        return {"ok": True, "task": text}

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
        return body.ask(("guide", req.on))

    def frame(name):
        with body.lock:
            return body.frames.get(name)

    @app.get("/camera/{name}.jpg")
    def jpg(name: str):
        f = frame(name)
        if f is None:
            raise HTTPException(404, f"no frame from camera '{name}' (cameras: {body.cams})")
        return Response(f, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

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

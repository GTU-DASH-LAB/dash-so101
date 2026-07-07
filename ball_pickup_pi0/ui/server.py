"""
Persistent pi0 rollout server + web UI for ball_pickup_pi0.

Loads the policy and connects to the robot/cameras ONCE at startup (the slow
part -- ~80s for a 4B-param pi0 checkpoint), then serves a local web UI with
live camera streams and Start/Pause/Complete/Manual/Reset controls. This
avoids paying the full reload cost every time you want to restart, pause to
fix a camera, or take over manually.

Uses lerobot's real RTCInferenceEngine (the same class 3_run_autonomous.sh's
--inference.type=rtc drives) rather than a hand-rolled approximation, so
autonomous motion matches the CLI script exactly -- same model computation,
same chunk smoothing. Manual jog and Reset-to-home are UI-only features with
no CLI equivalent, so they use their own independent rate limiting instead of
the robot's (deliberately unset here, matching 3_run_autonomous.sh's current
config) max_relative_target.

Run via ../4_run_ui.sh, not directly -- it needs config.env sourced into the
environment first.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.configuration_opencv import Cv2Rotation
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import make_robot_action
from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.utils.feature_utils import hw_to_dataset_features

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ui_server")

# --- Config, passed in via environment by 4_run_ui.sh ------------------------
FOLLOWER_PORT = os.environ["FOLLOWER_PORT"]
FOLLOWER_ID = os.environ["FOLLOWER_ID"]
TASK = os.environ["TASK"]
POLICY_PATH = os.environ["POLICY_PATH"]
DEVICE = os.environ.get("UI_DEVICE", "cuda")
CONTROL_FPS = float(os.environ.get("UI_FPS", "30"))
# Camera physical settings, matching config.env's CAMERAS but keyed by the
# policy's own expected feature names -- this sidesteps needing --rename_map
# in this script (see CLAUDE.md: rename_map only affects dataset columns, not
# a checkpoint's declared feature names, which stay base_0_rgb/left_wrist_0_rgb
# /right_wrist_0_rgb permanently).
CAMERA_CONFIG = {
    "base_0_rgb": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    "left_wrist_0_rgb": OpenCVCameraConfig(
        index_or_path=2, width=640, height=480, fps=30, rotation=Cv2Rotation.ROTATE_180, mirror=True
    ),
}
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Manual jog / reset-to-home have no CLI equivalent, so they get their own
# independent smoothing here rather than relying on the robot's own
# max_relative_target (deliberately left unset below, matching
# 3_run_autonomous.sh's current config exactly for autonomous/RTC motion).
MANUAL_STEP_MAX_DEG = 5.0
MANUAL_SLIDER_SPAN = 60.0  # how far a slider may range from where manual mode started
RESET_EPSILON = 1.5  # degrees; how close counts as "arrived" for Reset


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    mode: str = "idle"  # idle | running | paused | manual | resetting
    manual_targets: dict = field(default_factory=dict)
    home_pose: dict | None = None  # captured at connect time -- see build_rollout_context precedent
    joint_positions: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict)  # cam_key -> latest JPEG bytes
    last_error: str | None = None
    episodes_completed: int = 0
    started_at: float = field(default_factory=time.time)


state = SharedState()
app = FastAPI()


# --- One-time setup: policy + robot + RTC engine -----------------------------
def load_policy():
    log.info("Loading policy from %s (this takes ~60-90s for pi0)...", POLICY_PATH)
    device = torch.device(DEVICE)
    policy = PI0Policy.from_pretrained(POLICY_PATH)
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        POLICY_PATH,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    log.info("Policy loaded.")
    return policy, preprocess, postprocess, device


def connect_robot():
    # No max_relative_target here -- matches 3_run_autonomous.sh's current
    # config exactly. RTC's own chunk smoothing is the only thing shaping
    # autonomous motion, same as the CLI script.
    robot_cfg = SOFollowerRobotConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, cameras=CAMERA_CONFIG)
    robot = SOFollower(robot_cfg)
    robot.connect()
    log.info("Robot connected.")
    return robot


policy, preprocess, postprocess, device = load_policy()
robot = connect_robot()

action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

robot_wrapper = ThreadSafeRobot(robot)
rtc_engine = RTCInferenceEngine(
    policy=policy,
    preprocessor=preprocess,
    postprocessor=postprocess,
    robot_wrapper=robot_wrapper,
    rtc_config=RTCConfig(),
    hw_features=obs_features,
    task=TASK,
    fps=CONTROL_FPS,
    device=str(device),
)
rtc_engine.start()  # background thread launches now, idle until resume()
log.info("RTC inference engine started (idle until Start is pressed).")

# Capture the pose the arm was placed in before launch as the safe "home" for
# Reset -- same precedent as lerobot.rollout.context's initial_position, not
# a guessed/hardcoded coordinate.
_initial_obs = robot.get_observation()
state.home_pose = {k.removesuffix(".pos"): v for k, v in _initial_obs.items() if k.endswith(".pos")}
state.joint_positions = dict(state.home_pose)
log.info("Home pose captured: %s", state.home_pose)


# --- Control loop (background thread) ----------------------------------------
def encode_frame(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def step_toward(current: dict, target: dict, max_step: float) -> dict:
    """One rate-limited step from current towards target -- the only motion
    smoothing for manual/reset modes, since the robot's own clamp is unset."""
    out = {}
    for k, tgt in target.items():
        cur = current.get(k, tgt)
        delta = max(-max_step, min(max_step, tgt - cur))
        out[k] = cur + delta
    return out


def control_loop():
    control_interval = 1.0 / CONTROL_FPS
    while True:
        loop_start = time.perf_counter()
        try:
            with state.lock:
                mode = state.mode
                manual_targets = dict(state.manual_targets)
                home_pose = dict(state.home_pose) if state.home_pose else None

            obs = robot.get_observation()

            # Update camera preview regardless of mode.
            frames = {}
            for cam_key in CAMERA_CONFIG:
                if cam_key in obs:
                    frames[cam_key] = encode_frame(obs[cam_key])
            joint_positions = {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}
            loop_error = None

            if mode == "running" and rtc_engine.failed:
                # After enough consecutive errors the RTC background thread
                # exits entirely (see rtc.py) -- reset()/resume() alone can't
                # bring it back, only stop()+start() spins up a fresh thread.
                log.error("RTC engine failed; restarting it and pausing.")
                rtc_engine.stop()
                rtc_engine.start()
                loop_error = "RTC engine crashed and was restarted -- click Start to resume."
                with state.lock:
                    state.mode = "paused"
                mode = "paused"

            if mode == "running":
                # Feed the RTC engine the raw observation every tick, same as
                # BaseStrategy.run() does; it does its own preprocessing
                # internally (build_dataset_frame + prepare_observation_for_inference).
                rtc_engine.notify_observation(obs)
                action_tensor = rtc_engine.get_action(None)
                if action_tensor is not None:
                    # Queue already holds post-processed actions (RTC's
                    # background thread runs self._postprocessor before
                    # merging into the queue) -- no extra postprocess() here.
                    action = make_robot_action(action_tensor.unsqueeze(0), dataset_features)
                    robot.send_action(action)

            elif mode == "manual" and manual_targets:
                stepped = step_toward(joint_positions, manual_targets, MANUAL_STEP_MAX_DEG)
                robot.send_action({f"{k}.pos": v for k, v in stepped.items()})

            elif mode == "resetting" and home_pose:
                stepped = step_toward(joint_positions, home_pose, MANUAL_STEP_MAX_DEG)
                robot.send_action({f"{k}.pos": v for k, v in stepped.items()})
                if all(abs(joint_positions[k] - home_pose[k]) < RESET_EPSILON for k in home_pose):
                    with state.lock:
                        if state.mode == "resetting":
                            state.mode = "idle"
                    log.info("Reset complete.")

            # "idle" / "paused": hold position, no send_action call.

            with state.lock:
                state.frames = frames
                state.joint_positions = joint_positions
                state.last_error = loop_error

        except Exception as e:  # noqa: BLE001 -- keep the loop alive no matter what
            log.exception("Control loop error")
            with state.lock:
                state.last_error = str(e)
            time.sleep(0.5)
            continue

        dt = time.perf_counter() - loop_start
        if (sleep_t := control_interval - dt) > 0:
            time.sleep(sleep_t)


threading.Thread(target=control_loop, daemon=True).start()


# --- HTTP API -----------------------------------------------------------------
class JointCommand(BaseModel):
    value: float


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/status")
def status():
    with state.lock:
        return {
            "mode": state.mode,
            "joint_positions": state.joint_positions,
            "manual_targets": state.manual_targets,
            "home_pose": state.home_pose,
            "task": TASK,
            "policy_path": POLICY_PATH,
            "last_error": state.last_error,
            "episodes_completed": state.episodes_completed,
            "uptime_s": round(time.time() - state.started_at, 1),
        }


@app.post("/control/start")
def control_start():
    rtc_engine.reset()  # clear policy + processor state + action queue
    rtc_engine.resume()
    with state.lock:
        state.mode = "running"
    return {"mode": "running"}


@app.post("/control/pause")
def control_pause():
    rtc_engine.pause()
    with state.lock:
        state.mode = "paused"
    return {"mode": "paused"}


@app.post("/control/complete")
def control_complete():
    # Same visible effect as Reset (stop + go home) plus counting the
    # episode -- previously this just set mode="paused", identical to the
    # Pause button with no visible difference, which looked broken.
    rtc_engine.pause()
    rtc_engine.reset()
    with state.lock:
        state.mode = "resetting"
        state.episodes_completed += 1
    return {"mode": "resetting", "episodes_completed": state.episodes_completed}


@app.post("/control/manual")
def control_manual():
    rtc_engine.pause()
    rtc_engine.reset()  # discard cached chunk -- it was computed for whatever
    # pose the arm was in before manual control, now stale.
    with state.lock:
        # Seed sliders at the arm's current position so switching modes never
        # causes a jump.
        state.manual_targets = dict(state.joint_positions)
        state.mode = "manual"
    return {"mode": "manual", "manual_targets": state.manual_targets}


@app.post("/control/reset")
def control_reset():
    rtc_engine.pause()
    rtc_engine.reset()  # same reasoning as /control/manual
    with state.lock:
        state.mode = "resetting"
    return {"mode": "resetting"}


@app.post("/joint/{name}")
def set_joint(name: str, cmd: JointCommand):
    if name not in JOINT_NAMES:
        raise HTTPException(404, f"Unknown joint '{name}'")
    with state.lock:
        if state.mode != "manual":
            raise HTTPException(409, "Not in manual mode -- click 'Manual' first.")
        center = state.joint_positions.get(name, 0.0)
        clamped = max(center - MANUAL_SLIDER_SPAN, min(center + MANUAL_SLIDER_SPAN, cmd.value))
        state.manual_targets[name] = clamped
    return {"joint": name, "target": clamped}


def mjpeg_generator(cam_key: str):
    boundary = b"--frame"
    while True:
        with state.lock:
            frame = state.frames.get(cam_key)
        if frame:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(1 / 15)


@app.get("/video/{cam_key}")
def video(cam_key: str):
    if cam_key not in CAMERA_CONFIG:
        raise HTTPException(404, f"Unknown camera '{cam_key}'")
    return StreamingResponse(
        mjpeg_generator(cam_key), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.on_event("shutdown")
def shutdown():
    log.info("Shutting down: stopping RTC engine and disconnecting robot.")
    rtc_engine.stop()
    robot.disconnect()

#!/usr/bin/env python3
"""Record ONE session of teleoperated demos into its own small LeRobotDataset.

Started by 1_record.sh (don't run it by hand). It reuses lerobot's own `record_loop`
for teleoperation, so every frame is recorded exactly as `lerobot-record` would record
it. The safety and recovery logic around that loop is what this script adds:

* Each session is a separate dataset (<session>/dataset). An open dataset is only
  consistent on disk after finalize(), so a hard crash (power cut, kill -9) can only
  damage the session being recorded, never the earlier ones.
* Unplugged camera/arm, power loss, Ctrl+C: completed episodes are kept. The episode
  in progress is thrown away. An episode that finished just before the failure,
  during the reset pause, is saved. Exit code 3 tells 1_record.sh to wait for the
  hardware to come back and then continue the same session.
* The follower is blended smoothly onto the leader's pose at start-up and after every
  save. It never jumps to wherever the leader happens to be.
* Esc during a demo discards that demo. Esc during the reset pause keeps it.
* Demos shorter than --min-episode-s (accidental double key press) are discarded.

Exit codes: 0 session complete | 2 stopped with Esc | 3 hardware problem |
            130 Ctrl+C | 1 anything else.
"""

import argparse
import errno
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

EXIT_DONE, EXIT_ERROR, EXIT_STOPPED, EXIT_HARDWARE, EXIT_CTRL_C = 0, 1, 2, 3, 130

TTY = sys.stdout.isatty()
RED, GRN, YEL, BLU, BOLD, OFF = (
    ("\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[1m", "\033[0m") if TTY else ("",) * 6
)
log = logging.getLogger("record_session")


def banner(color: str, text: str) -> None:
    line = "=" * 72
    print(f"\n{color}{line}\n  {text}\n{line}{OFF}", flush=True)
    log.info("BANNER %s", text)


def note(text: str) -> None:
    print(f"   {text}", flush=True)
    log.info(text)


def looks_like_hardware_failure(exc: BaseException) -> bool:
    """Unplugged USB, powered-off servos, a camera that stopped delivering frames."""
    import serial  # pyserial, a lerobot dependency

    if isinstance(exc, (ConnectionError, TimeoutError, OSError, serial.SerialException)):
        return True
    name = type(exc).__name__
    text = str(exc).lower()
    keys = ("deviceconnected", "status packet", "camera", "read failed", "port", "frame", "motor", "torque")
    return "DeviceNotConnected" in name or "DeviceAlreadyConnected" in name or any(k in text for k in keys)


class Session:
    """session.json + episodes.jsonl next to the dataset: what 1_record.sh and
    manage_data.sh read to report progress without opening the dataset."""

    def __init__(self, session_dir: Path):
        self.dir = session_dir
        self.meta_file = session_dir / "session.json"
        self.episodes_file = session_dir / "episodes.jsonl"
        self.meta = json.loads(self.meta_file.read_text()) if self.meta_file.exists() else {}

    def update(self, **kw) -> None:
        self.meta.update(kw)
        self.meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        tmp = self.meta_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.meta, indent=2))
        os.replace(tmp, self.meta_file)  # atomic: never a half-written status file

    def add_episode(self, **kw) -> None:
        with open(self.episodes_file, "a") as f:
            f.write(json.dumps(kw) + "\n")


def align_follower_to_leader(robot, teleop, seconds: float, fps: int) -> None:
    """Blend the follower from its current pose onto the LIVE leader pose.

    Without this, the first command after connecting (or after a save pause) sends
    the follower straight to wherever the leader is, at full servo speed."""
    from lerobot.utils.robot_utils import precise_sleep

    keys = list(robot.action_features)
    start_obs = robot.get_observation()
    start = {k: float(start_obs[k]) for k in keys}
    n = max(1, int(seconds * fps))
    for i in range(1, n + 1):
        t0 = time.perf_counter()
        target = teleop.get_action()
        a = i / n
        a = a * a * (3 - 2 * a)  # smoothstep: gentle start and stop
        robot.send_action({k: start[k] + (float(target[k]) - start[k]) * a for k in keys})
        precise_sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))


def discard_demo(dataset) -> None:
    """Throw away the demo in the buffer, including its temp PNGs.

    lerobot's clear_episode_buffer() only deletes temp images for *image* features,
    not *video* ones, so the PNGs of a discarded demo stayed on disk and a shorter
    redo of the same demo index would have had stale frames encoded into its video."""
    ep = dataset.num_episodes
    dataset.clear_episode_buffer()
    dataset.writer._wait_image_writer()
    dataset.writer.cleanup_interrupted_episode(ep)


def rerun_status(display: bool, text: str) -> None:
    if not display:
        return
    try:
        import rerun as rr

        rr.log("status", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN))
    except Exception:  # noqa: BLE001 - the viewer is a convenience, never a failure
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--episodes", type=int, required=True, help="episodes this session should hold in total")
    ap.add_argument("--total-before", type=int, default=0, help="episodes already in OTHER sessions (display)")
    ap.add_argument("--total-target", type=int, default=100)
    ap.add_argument("--task", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--episode-time-s", type=float, default=30)
    ap.add_argument("--reset-time-s", type=float, default=15)
    ap.add_argument("--min-episode-s", type=float, default=3)
    ap.add_argument("--follower-port", required=True)
    ap.add_argument("--follower-id", required=True)
    ap.add_argument("--leader-port", required=True)
    ap.add_argument("--leader-id", required=True)
    ap.add_argument("--cameras", required=True, help="CAMERAS JSON from config.env")
    ap.add_argument("--vcodec", default="h264")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--display", action="store_true", help="open the rerun live viewer")
    a = ap.parse_args()

    a.session_dir.mkdir(parents=True, exist_ok=True)
    ds_root = a.session_dir / "dataset"
    session = Session(a.session_dir)

    from lerobot.utils.utils import init_logging

    init_logging(log_file=a.session_dir / "record.log", console_level="WARNING")

    # Imports after logging so their warnings land in record.log too.
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import (
        LeRobotDataset,
        VideoEncodingManager,
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.processor import make_default_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SOFollowerRobotConfig
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.so_leader import SOLeaderTeleopConfig
    from lerobot.utils.feature_utils import combine_feature_dicts
    from lerobot.utils.keyboard_input import init_keyboard_listener
    from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

    cameras = {}
    for name, spec in json.loads(a.cameras).items():
        spec = {k: v for k, v in spec.items() if k != "type"}
        cameras[name] = OpenCVCameraConfig(**spec)
    robot = make_robot_from_config(
        SOFollowerRobotConfig(port=a.follower_port, id=a.follower_id, cameras=cameras)
    )
    teleop = make_teleoperator_from_config(SOLeaderTeleopConfig(port=a.leader_port, id=a.leader_id))

    teleop_proc, robot_proc, obs_proc = make_default_processors()
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=obs_proc,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    rgb = RGBEncoderConfig(vcodec=a.vcodec, crf=a.crf)
    writer_threads = 4 * len(cameras)

    exit_code = EXIT_ERROR
    dataset = None
    listener = None
    phase = "setup"
    rerecord_flag = False
    try:
        if (ds_root / "meta" / "info.json").is_file():
            dataset = LeRobotDataset.resume(
                a.repo_id, root=ds_root, rgb_encoder=rgb, image_writer_threads=writer_threads
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, a.fps, features)
        else:
            if ds_root.exists():
                # A previous run died inside create(), before any metadata was written.
                # Nothing recoverable can be in there; set it aside instead of deleting.
                aside = a.session_dir / f"dataset.incomplete.{int(time.time())}"
                ds_root.rename(aside)
                note(f"{YEL}Set aside an incomplete dataset folder from a crashed start: {aside.name}{OFF}")
            dataset = LeRobotDataset.create(
                a.repo_id,
                a.fps,
                root=ds_root,
                robot_type=robot.name,
                features=features,
                use_videos=True,
                image_writer_threads=writer_threads,
                rgb_encoder=rgb,
            )
            session.update(created_at=datetime.now().isoformat(timespec="seconds"), repo_id=a.repo_id)
        session.update(status="recording", target_episodes=a.episodes)

        already = dataset.num_episodes
        if already >= a.episodes:
            note(f"This session already has {already}/{a.episodes} episodes.")
            exit_code = EXIT_DONE
            return exit_code

        phase = "connect"
        note("Connecting the follower arm, leader arm and wrist camera ...")
        robot.connect()
        teleop.connect()

        listener, events = init_keyboard_listener()
        if a.display:
            init_visualization("rerun", session_name=f"recording {a.session_dir.name}")

        phase = "align"
        note("Blending the follower onto the leader's pose (2 s). Hold the leader still.")
        align_follower_to_leader(robot, teleop, seconds=2.0, fps=a.fps)

        min_frames = int(a.min_episode_s * a.fps)
        with VideoEncodingManager(dataset):
            try:
                while dataset.num_episodes < a.episodes and not events["stop_recording"]:
                    ep = dataset.num_episodes
                    overall = a.total_before + ep + 1
                    phase = "record"
                    banner(
                        RED,
                        f"RECORDING  demo {ep + 1}/{a.episodes} of {a.session_dir.name}   "
                        f"(overall {overall}/{a.total_target})\n"
                        f"  ->  done (save)     <-  redo this demo     Esc  stop session",
                    )
                    rerun_status(a.display, f"# RECORDING\ndemo {ep + 1}/{a.episodes} - overall {overall}/{a.total_target}")
                    t0 = time.perf_counter()
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=a.fps,
                        teleop_action_processor=teleop_proc,
                        robot_action_processor=robot_proc,
                        robot_observation_processor=obs_proc,
                        teleop=teleop,
                        dataset=dataset,
                        control_time_s=a.episode_time_s,
                        single_task=a.task,
                        display_data=a.display,
                        display_mode="rerun",
                    )
                    secs = time.perf_counter() - t0
                    frames = dataset.writer.episode_buffer["size"]
                    stopped_during_demo = events["stop_recording"]
                    rerecord_flag = events["rerecord_episode"]

                    # Always pause after a demo, even when stopping: the follower goes limp
                    # when the session ends, so the arm must be brought down to rest first.
                    phase = "reset"
                    last = stopped_during_demo or dataset.num_episodes + 1 >= a.episodes
                    if stopped_during_demo:
                        banner(
                            YEL,
                            f"STOPPING: lower the arm to its REST pose now - motors relax in {a.reset_time_s:.0f} s\n"
                            f"  (the demo in progress is thrown away)      ->  finish now",
                        )
                    elif last:
                        banner(
                            YEL,
                            f"LAST DEMO OF THIS SESSION: lower the arm to its REST pose - motors relax in {a.reset_time_s:.0f} s\n"
                            f"  ->  finish now     <-  throw away the demo you just did",
                        )
                    else:
                        banner(
                            YEL,
                            f"RESET  ({a.reset_time_s:.0f} s): put the pen back, return the arm to its start pose\n"
                            f"  ->  start next demo now     <-  throw away the demo you just did     Esc  save & stop",
                        )
                    rerun_status(a.display, "# RESET\nput the scene back, arm to start pose")
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=a.fps,
                        teleop_action_processor=teleop_proc,
                        robot_action_processor=robot_proc,
                        robot_observation_processor=obs_proc,
                        teleop=teleop,
                        control_time_s=a.reset_time_s,
                        single_task=a.task,
                        display_data=a.display,
                        display_mode="rerun",
                    )
                    rerecord_flag = rerecord_flag or events["rerecord_episode"]

                    phase = "save"
                    if rerecord_flag or stopped_during_demo:
                        if stopped_during_demo:
                            note(f"{YEL}The demo in progress was thrown away (Esc during the demo).{OFF}")
                        else:
                            note(f"{YEL}Demo thrown away (you pressed <-). Record it again.{OFF}")
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                        rerecord_flag = False
                        discard_demo(dataset)
                        continue
                    if frames < min_frames:
                        note(f"{YEL}Demo was only {frames / a.fps:.1f} s (< {a.min_episode_s} s): discarded as an accidental key press.{OFF}")
                        discard_demo(dataset)
                        continue

                    t_save = time.perf_counter()
                    dataset.save_episode()
                    save_s = time.perf_counter() - t_save
                    session.add_episode(
                        episode_index=ep, frames=frames, seconds=round(frames / a.fps, 2),
                        wall_seconds=round(secs, 2), saved_at=datetime.now().isoformat(timespec="seconds"),
                    )
                    session.update(episodes=dataset.num_episodes)
                    note(f"{GRN}Saved demo {ep + 1} ({frames / a.fps:.1f} s, {frames} frames) in {save_s:.1f} s.{OFF}")
                    if save_s > 0.3 and dataset.num_episodes < a.episodes and not events["stop_recording"]:
                        phase = "align"
                        align_follower_to_leader(robot, teleop, seconds=1.0, fps=a.fps)
            except BaseException:
                # A demo that was fully recorded before the failure (we were in the reset
                # pause) is complete and valid. Save it before lerobot's cleanup runs.
                if phase == "reset" and not rerecord_flag and dataset.has_pending_frames():
                    frames = dataset.writer.episode_buffer["size"]
                    if frames >= min_frames:
                        try:
                            dataset.save_episode()
                            session.add_episode(
                                episode_index=dataset.num_episodes - 1, frames=frames,
                                seconds=round(frames / a.fps, 2), rescued=True,
                                saved_at=datetime.now().isoformat(timespec="seconds"),
                            )
                            session.update(episodes=dataset.num_episodes)
                            note(f"{GRN}The demo you had just finished was saved before stopping.{OFF}")
                        except Exception:  # noqa: BLE001
                            log.exception("rescue save failed")
                raise  # VideoEncodingManager then cleans up the half-recorded demo

        if dataset.num_episodes >= a.episodes:
            exit_code = EXIT_DONE
        else:
            exit_code = EXIT_STOPPED

    except KeyboardInterrupt:
        exit_code = EXIT_CTRL_C
        note(f"{YEL}Ctrl+C: stopping. Finished demos are kept.{OFF}")
    except Exception as e:  # noqa: BLE001
        log.error("Failure during phase %s:\n%s", phase, traceback.format_exc())
        if isinstance(e, OSError) and e.errno == errno.ENOSPC:
            print(f"\n{RED}DISK FULL.{OFF} Free space on / and run ./1_record.sh again.", flush=True)
            hw = False
        else:
            hw = looks_like_hardware_failure(e)
        exit_code = EXIT_HARDWARE if hw and phase != "setup" else EXIT_ERROR
        print(f"\n{RED}{'=' * 72}\n  STOPPED: {type(e).__name__}: {e}\n{'=' * 72}{OFF}", flush=True)
        if exit_code == EXIT_HARDWARE:
            note("This looks like a hardware disconnect (USB cable, arm power, or camera).")
            note("All demos that finished are safe on disk. Only the demo in progress was lost.")
        else:
            note(f"Full traceback: {a.session_dir / 'record.log'}")
    finally:
        # Each step on its own: a dead USB device must not stop the others from closing.
        if dataset is not None:
            try:
                dataset.finalize()
            except Exception:  # noqa: BLE001
                log.exception("finalize failed")
        for dev in (robot, teleop):
            try:
                if dev.is_connected:
                    dev.disconnect()
            except Exception:  # noqa: BLE001
                log.exception("disconnect failed for %s", dev)
        if listener is not None:
            try:
                listener.stop()
            except Exception:  # noqa: BLE001
                pass
        if a.display and phase not in ("setup", "connect"):
            try:
                shutdown_visualization("rerun")
            except Exception:  # noqa: BLE001
                pass
        n = dataset.num_episodes if dataset is not None else 0
        status = {EXIT_DONE: "complete", EXIT_STOPPED: "paused", EXIT_HARDWARE: "interrupted",
                  EXIT_CTRL_C: "paused"}.get(exit_code, "error")
        try:
            session.update(status=status, episodes=n, last_exit_code=exit_code)
        except Exception:  # noqa: BLE001
            pass
        print(f"\n   Session {a.session_dir.name}: {n}/{a.episodes} demos on disk "
              f"(overall {a.total_before + n}/{a.total_target}).", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

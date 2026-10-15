#!/usr/bin/env python3
"""Pre-flight hardware check (copied from pen_bowl_act). READ-ONLY: never enables torque,
never writes a register, never moves a motor.

For each arm it resolves the USB serial to the current /dev/ttyACMn, pings all six
motors, and reads the homing offsets stored in the motors' EEPROM. Those must equal
the calibration file, which proves the arm is the one we think it is (leader vs
follower can swap ACM numbers on every replug). For the camera it grabs frames
through lerobot's own OpenCVCamera, exactly as recording will.

Exit code 0 = everything needed is OK. The last stdout line is machine-readable:
    PORTS leader=/dev/ttyACM0 follower=/dev/ttyACM1
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAL_ROOT = Path.home() / ".cache/huggingface/lerobot/calibration"
CAL_FILES = {
    "leader": ("teleoperators/so_leader", "LEADER"),
    "follower": ("robots/so_follower", "FOLLOWER"),
}

GREEN, RED, YELLOW, OFF = ("\033[1;32m", "\033[1;31m", "\033[1;33m", "\033[0m") if sys.stdout.isatty() else ("",) * 4


def say(ok: bool | None, msg: str) -> None:
    tag = f"{GREEN}OK  {OFF}" if ok else (f"{YELLOW}WARN{OFF}" if ok is None else f"{RED}FAIL{OFF}")
    print(f"  [{tag}] {msg}", flush=True)


def port_for_serial(serial: str) -> str | None:
    links = glob.glob(f"/dev/serial/by-id/*{serial}*")
    return os.path.realpath(links[0]) if links else None


def check_arm(role: str, serial: str, cal_id: str) -> str | None:
    """Return the resolved port if the arm is present, powered and correctly calibrated."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    port = port_for_serial(serial)
    if port is None:
        say(False, f"{role} arm: USB serial {serial} not found. Is its USB cable plugged in?")
        present = sorted(os.path.basename(p) for p in glob.glob("/dev/serial/by-id/*"))
        print(f"         USB serial devices present now: {present or 'none'}")
        return None
    if not os.access(port, os.R_OK | os.W_OK):
        say(False, f"{role} arm: no permission on {port}. Reboot (dialout group) or prefix with: sg dialout -c ...")
        return None

    cal_file = CAL_ROOT / CAL_FILES[role][0] / f"{cal_id}.json"
    bus = FeetechMotorsBus(
        port=port,
        motors={n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(MOTORS)},
    )
    try:
        bus.connect(handshake=True)
    except Exception as e:  # noqa: BLE001 - report every failure mode the same way
        say(False, f"{role} arm on {port}: USB is fine but the motors do not answer ({type(e).__name__}).")
        print(f"         -> Is the arm's power supply plugged in and switched on? ({e})")
        return None
    try:
        offsets = bus.sync_read("Homing_Offset", normalize=False, num_retry=3)
        volts = bus.sync_read("Present_Voltage", normalize=False, num_retry=3)
        temps = bus.sync_read("Present_Temperature", normalize=False, num_retry=3)
    except Exception as e:  # noqa: BLE001
        say(False, f"{role} arm on {port}: motors answered the ping but reads failed: {e}")
        return None
    finally:
        # disable_torque=False: closing the port must not write anything.
        bus.disconnect(disable_torque=False)

    v = min(volts.values()) / 10.0
    t = max(temps.values())
    say(True, f"{role} arm: {port} (serial {serial}), 6/6 motors, {v:.1f} V, max {t} C")
    if role == "follower" and v < 10.0:
        say(None, f"follower supply reads {v:.1f} V; it should be ~12 V. Check the power brick.")
    if t >= 60:
        say(None, f"{role} motor temperature {t} C is high. Let the arm cool down.")

    if not cal_file.is_file():
        say(False, f"{role} calibration file missing: {cal_file}")
        return None
    cal = json.loads(cal_file.read_text())
    mismatched = [m for m in MOTORS if cal[m]["homing_offset"] != offsets[m]]
    if mismatched:
        say(False, f"{role} arm's stored calibration does not match {cal_file.name} ({', '.join(mismatched)}).")
        print("         -> Either the two arms' USB serials are swapped in config.env, or the arm was")
        print("            recalibrated elsewhere. Do not record until this is resolved.")
        return None
    say(True, f"{role} arm calibration matches {cal_file}")
    return port


def check_camera(cameras_json: str, snapshot: Path | None) -> bool:
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    spec = dict(json.loads(cameras_json)["wrist"])
    spec.pop("type", None)
    path = spec["index_or_path"]
    if isinstance(path, str) and not os.path.exists(path):
        say(False, f"wrist camera not found at {path}. Is its USB cable plugged in? (./find_camera.sh lists cameras)")
        return False
    if isinstance(path, str) and not os.access(os.path.realpath(path), os.R_OK):
        say(False, f"no permission on {os.path.realpath(path)}. Reboot (video group) or prefix with: sg video -c ...")
        return False
    cam = OpenCVCamera(OpenCVCameraConfig(**spec))
    try:
        cam.connect()
        frames, t0, last = 0, time.perf_counter(), None
        while time.perf_counter() - t0 < 2.0:
            img = cam.async_read(timeout_ms=1000)
            frames += 1
            last = img
    except Exception as e:  # noqa: BLE001
        say(False, f"wrist camera at {path} failed: {type(e).__name__}: {e}")
        print("         -> Unplug and replug the camera, wait 5 s, try again. If another program")
        print("            (a browser, lelab, a previous recording) has it open, close that first.")
        return False
    finally:
        try:
            cam.disconnect()
        except Exception:  # noqa: BLE001
            pass
    fps = frames / 2.0
    say(True, f"wrist camera: {os.path.realpath(path)} {last.shape[1]}x{last.shape[0]}, ~{fps:.0f} fps")
    if fps < 15:
        say(None, "camera fps is low. More light on the scene usually fixes this (auto-exposure).")
    if snapshot is not None:
        import cv2

        snapshot.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snapshot), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leader-serial")
    ap.add_argument("--follower-serial", required=True)
    ap.add_argument("--leader-id", default="dash_so101")
    ap.add_argument("--follower-id", default="dash_so101")
    ap.add_argument("--cameras", required=True, help="the CAMERAS JSON from config.env")
    ap.add_argument("--no-leader", action="store_true", help="skip the leader arm (for running a policy)")
    ap.add_argument("--snapshot", type=Path, help="save one camera frame here")
    a = ap.parse_args()

    print("Hardware check (read-only, nothing will move):")
    ports = {}
    good = True
    if not a.no_leader:
        ports["leader"] = check_arm("leader", a.leader_serial, a.leader_id)
        good &= ports["leader"] is not None
    ports["follower"] = check_arm("follower", a.follower_serial, a.follower_id)
    good &= ports["follower"] is not None
    good &= check_camera(a.cameras, a.snapshot)

    load1 = os.getloadavg()[0]
    ncpu = os.cpu_count() or 1
    if load1 > ncpu * 0.75:
        say(None, f"CPU load is high ({load1:.0f} on {ncpu} threads). The arm bus can time out under")
        print("         CPU starvation; stop other heavy jobs (training, Isaac Sim) before recording.")
    print("PORTS " + " ".join(f"{k}={v}" for k, v in ports.items() if v))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())

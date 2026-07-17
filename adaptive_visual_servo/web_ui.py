"""Web UI backend for the SO-101 adaptive visual servo system.

Flask + WebSocket server that exposes the robot and camera over HTTP,
replacing the Tkinter UI with a modern browser-based interface.

Usage:
    python web_ui.py                 # starts on http://localhost:5001
    python run_real.py --web-ui      # same, integrated with CLI
"""

import base64
import json
import os
import sys
import threading
import time
import traceback
from io import BytesIO
from dataclasses import asdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO

from config import RealConfig, ServoConfig
from aruco_tracker import (ArucoDetector, CalibrationData,
                           calibrate_workspace, estimate_default_intrinsics,
                           calibrate_camera_charuco, pixel_to_workspace,
                           workspace_to_pixel)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web-ui", "dist")

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
app.config["SECRET_KEY"] = "avs-web-ui"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

# Load saved calibration if available on startup to restore last_port and calibration state
_default_cfg = RealConfig()
_cal_data = CalibrationData()
_cal_path = os.path.join(os.path.dirname(__file__), _default_cfg.calibration_file)
if os.path.exists(_cal_path):
    try:
        _cal_data = CalibrationData.load(_cal_path)
        if _cal_data.last_port:
            _default_cfg.port = _cal_data.last_port
        if _cal_data.grasp_z is not None:
            _default_cfg.grasp_z = _cal_data.grasp_z
    except Exception:
        pass

state = {
    "rig": None,
    "model": None,
    "cfg": _default_cfg,
    "detector": None,        # ArucoDetector singleton
    "calibration": _cal_data,
    "background": None,
    "J": None,
    "home_q": None,
    "episode_running": False,
    "charuco_frames": [],    # collected for intrinsic calibration
    "rng": np.random.default_rng(0),
    "video_running": False,
    "last_markers": {},      # last ArUco detection results
    "before_ports": None,    # used for port unplug-detect diffing
}

log_lock = threading.Lock()


def emit_log(msg: str):
    """Send a log line to connected browsers."""
    socketio.emit("log", {"msg": msg, "ts": time.time()})


def _busy_error():
    """Distinct guard errors so 'reset failed' isn't ambiguous in the log."""
    if state["rig"] is None:
        return jsonify({"error": "Not connected"}), 400
    return jsonify({"error": "Episode still running -- wait for it to finish "
                             "(watch the log) or use E-Stop"}), 400


def get_aruco_detector() -> ArucoDetector:
    if state["detector"] is None:
        state["detector"] = ArucoDetector()
    return state["detector"]


def _marker_sizes_map() -> dict:
    """Build {marker_id: physical_size} from config."""
    cfg = state["cfg"]
    sizes = {}
    for mid in cfg.workspace_marker_ids:
        sizes[mid] = cfg.aruco_ws_marker_size
    sizes[cfg.gripper_marker_id] = cfg.aruco_robot_marker_size
    sizes[cfg.wrist_marker_id] = cfg.aruco_robot_marker_size
    return sizes


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(WEB_DIR, path)


# ---------------------------------------------------------------------------
# Camera endpoints
# ---------------------------------------------------------------------------

@app.route("/api/cameras", methods=["GET"])
def list_cameras():
    """Probe available cameras (limited to avoid macOS AVFoundation crash)."""
    from camera_nanodet_tester import probe_cameras
    cameras = probe_cameras()
    return jsonify([{"index": i, "width": w, "height": h} for i, w, h in cameras])


# ---------------------------------------------------------------------------
# Connection endpoints
# ---------------------------------------------------------------------------

@app.route("/api/connect", methods=["POST"])
def connect():
    data = request.json or {}
    cfg = state["cfg"]
    cfg.port = data.get("port", cfg.port)
    cfg.camera_index = data.get("camera_index", cfg.camera_index)

    if state["rig"] is not None:
        return jsonify({"error": "Already connected"}), 400

    try:
        from run_real import SO101Rig, _FrameTap
        from lerobot_ik import PlacoModel

        emit_log("Connecting to arm...")
        rig = _FrameTap(SO101Rig(cfg))
        state["rig"] = rig
        state["model"] = PlacoModel()
        state["home_q"] = rig.get_q()

        # Load saved calibration if available
        cal_path = os.path.join(os.path.dirname(__file__), cfg.calibration_file)
        if os.path.exists(cal_path):
            state["calibration"] = CalibrationData.load(cal_path)
            emit_log(f"Loaded calibration from {cfg.calibration_file}")

        # Auto-save last port in calibration
        state["calibration"].last_port = cfg.port
        state["calibration"].save(cal_path)

        emit_log("Connected successfully.")
        _start_video_stream()
        return jsonify({"status": "connected"})
    except Exception as e:
        emit_log(f"Connection failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/disconnect", methods=["POST"])
def disconnect():
    if state["rig"] is None:
        return jsonify({"error": "Not connected"}), 400
    state["video_running"] = False
    time.sleep(0.2)
    try:
        state["rig"].set_gripper(1.0)
        state["rig"].close()
    except Exception:
        pass
    state["rig"] = None
    state["model"] = None
    emit_log("Disconnected.")
    return jsonify({"status": "disconnected"})


@app.route("/api/status", methods=["GET"])
def status():
    connected = state["rig"] is not None
    cal = state["calibration"]
    markers = {}
    for mid, m in state["last_markers"].items():
        markers[mid] = {
            "center": m.center.tolist() if m.center is not None else None,
            "has_pose": m.tvec is not None,
        }
    return jsonify({
        "connected": connected,
        "has_intrinsics": cal.has_intrinsics(),
        "has_workspace": cal.has_workspace(),
        "reprojection_error": cal.reprojection_error if cal.has_workspace() else None,
        "calibrated_at": cal.calibrated_at,
        "markers": markers,
        "episode_running": state["episode_running"],
        "charuco_frames_collected": len(state["charuco_frames"]),
    })


# ---------------------------------------------------------------------------
# Video streaming
# ---------------------------------------------------------------------------

def _start_video_stream():
    """Start background thread that streams video frames."""
    if state["video_running"]:
        return
    state["video_running"] = True

    def stream():
        detector = get_aruco_detector()
        while state["video_running"] and state["rig"] is not None:
            try:
                frame = state["rig"].produce_frame()
                if frame is None:
                    time.sleep(0.03)
                    continue

                # ArUco detection + overlay
                cal = state["calibration"]
                cam_mtx = cal.camera_matrix
                dist = cal.dist_coeffs
                msizes = _marker_sizes_map()

                markers = detector.detect_all(frame, cam_mtx, dist,
                                             marker_size=state["cfg"].aruco_ws_marker_size,
                                             marker_sizes=msizes)
                state["last_markers"] = markers

                # Draw overlays
                img = frame.copy()
                detector.draw_markers(img, markers,
                                      draw_axes=cal.has_intrinsics(),
                                      camera_matrix=cam_mtx,
                                      dist_coeffs=dist)

                # Label markers with IDs and special roles
                for mid, m in markers.items():
                    cx, cy = int(m.center[0]), int(m.center[1])
                    label = f"ID:{mid}"
                    if mid in state["cfg"].workspace_marker_ids:
                        label += " [WS]"
                    elif mid == state["cfg"].gripper_marker_id:
                        label += " [GRIP]"
                    elif mid == state["cfg"].wrist_marker_id:
                        label += " [WRIST]"
                    cv2.putText(img, label, (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Draw 3D position if calibrated
                    if m.tvec is not None:
                        pos_str = f"({m.tvec[0]:.3f},{m.tvec[1]:.3f},{m.tvec[2]:.3f})"
                        cv2.putText(img, pos_str, (cx + 10, cy + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

                # Draw gripper→wrist vector if both visible
                grip_id = state["cfg"].gripper_marker_id
                wrist_id = state["cfg"].wrist_marker_id
                if grip_id in markers and wrist_id in markers:
                    g = markers[grip_id].center.astype(int)
                    w = markers[wrist_id].center.astype(int)
                    cv2.arrowedLine(img, tuple(w), tuple(g), (0, 255, 0), 2)

                # Encode as JPEG and emit via WebSocket
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")

                # Also send marker data as structured info
                marker_data = {}
                for mid, m in markers.items():
                    marker_data[str(mid)] = {
                        "center": m.center.tolist(),
                        "has_pose": m.tvec is not None,
                        "tvec": m.tvec.tolist() if m.tvec is not None else None,
                    }

                socketio.emit("frame", {
                    "image": b64,
                    "markers": marker_data,
                })
                time.sleep(0.066)  # ~15fps
            except Exception as e:
                time.sleep(0.1)

    threading.Thread(target=stream, daemon=True).start()


# ---------------------------------------------------------------------------
# Calibration endpoints
# ---------------------------------------------------------------------------

@app.route("/api/calibrate/collect_frame", methods=["POST"])
def collect_charuco_frame():
    """Collect one frame for ChArUco intrinsic calibration."""
    if state["rig"] is None:
        return jsonify({"error": "Not connected"}), 400
    frame = state["rig"].read()
    state["charuco_frames"].append(frame.copy())
    n = len(state["charuco_frames"])
    emit_log(f"Collected ChArUco frame {n}")
    return jsonify({"frames_collected": n})


@app.route("/api/calibrate/intrinsics", methods=["POST"])
def calibrate_intrinsics():
    """Run ChArUco intrinsic calibration from collected frames."""
    frames = state["charuco_frames"]
    if len(frames) < 5:
        return jsonify({"error": f"Need at least 5 frames, have {len(frames)}"}), 400

    emit_log(f"Calibrating intrinsics from {len(frames)} frames...")
    cam_mtx, dist, rms = calibrate_camera_charuco(frames)
    if cam_mtx is None:
        emit_log("Intrinsic calibration failed — not enough ChArUco corners detected.")
        return jsonify({"error": "Calibration failed"}), 400

    cal = state["calibration"]
    cal.camera_matrix = cam_mtx
    cal.dist_coeffs = dist
    cal.reprojection_error = rms
    cal.calibrated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Save
    cal_path = os.path.join(os.path.dirname(__file__), state["cfg"].calibration_file)
    cal.save(cal_path)

    emit_log(f"Intrinsic calibration done! RMS reprojection error: {rms:.3f}px")
    return jsonify({
        "rms": rms,
        "fx": float(cam_mtx[0, 0]),
        "fy": float(cam_mtx[1, 1]),
        "cx": float(cam_mtx[0, 2]),
        "cy": float(cam_mtx[1, 2]),
    })


@app.route("/api/calibrate/estimate_intrinsics", methods=["POST"])
def estimate_intrinsics():
    """Use a reasonable estimate for camera intrinsics (no ChArUco board needed)."""
    if state["rig"] is None:
        return jsonify({"error": "Not connected"}), 400
    frame = state["rig"].read()
    h, w = frame.shape[:2]
    cam_mtx, dist = estimate_default_intrinsics(w, h)

    cal = state["calibration"]
    cal.camera_matrix = cam_mtx
    cal.dist_coeffs = dist
    cal.calibrated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    cal_path = os.path.join(os.path.dirname(__file__), state["cfg"].calibration_file)
    cal.save(cal_path)

    emit_log(f"Estimated intrinsics from {w}x{h} frame (fx={cam_mtx[0,0]:.1f})")
    return jsonify({
        "fx": float(cam_mtx[0, 0]),
        "fy": float(cam_mtx[1, 1]),
        "cx": float(cam_mtx[0, 2]),
        "cy": float(cam_mtx[1, 2]),
    })


@app.route("/api/calibrate/workspace", methods=["POST"])
def calibrate_ws():
    """Calibrate workspace extrinsics from visible table markers."""
    if state["rig"] is None:
        return jsonify({"error": "Not connected"}), 400

    cal = state["calibration"]
    if not cal.has_intrinsics():
        return jsonify({"error": "Calibrate or estimate intrinsics first"}), 400

    data = request.json or {}
    # Allow overriding marker positions from the UI
    marker_positions = {}
    if "marker_positions" in data:
        for k, v in data["marker_positions"].items():
            marker_positions[int(k)] = np.array(v, dtype=np.float64)
    else:
        cfg = state["cfg"]
        for k, v in cfg.workspace_marker_positions.items():
            marker_positions[int(k)] = np.array(v, dtype=np.float64)

    marker_size = data.get("marker_size", state["cfg"].aruco_ws_marker_size)

    detector = get_aruco_detector()
    frame = state["rig"].read()

    rvec, tvec, rms = calibrate_workspace(
        detector, frame, marker_positions,
        cal.camera_matrix, cal.dist_coeffs, marker_size)

    if rvec is None:
        emit_log("Workspace calibration failed — need at least 1 visible workspace marker.")
        return jsonify({"error": "Not enough markers visible"}), 400

    cal.ws_rvec = rvec
    cal.ws_tvec = tvec
    cal.workspace_marker_positions = marker_positions

    cal_path = os.path.join(os.path.dirname(__file__), state["cfg"].calibration_file)
    cal.save(cal_path)

    emit_log(f"Workspace calibration done! RMS: {rms:.3f}px")
    return jsonify({"rms": rms})


@app.route("/api/calibrate/reset", methods=["POST"])
def reset_calibration():
    """Reset all calibration data."""
    state["calibration"] = CalibrationData()
    state["charuco_frames"] = []
    cal_path = os.path.join(os.path.dirname(__file__), state["cfg"].calibration_file)
    if os.path.exists(cal_path):
        os.remove(cal_path)
    emit_log("Calibration data reset.")
    return jsonify({"status": "reset"})


# ---------------------------------------------------------------------------
# Settings & Port Detection
# ---------------------------------------------------------------------------

@app.route("/api/ports/snapshot", methods=["POST"])
def ports_snapshot():
    """Take snapshot of connected serial ports."""
    from run_real import find_serial_ports
    state["before_ports"] = set(find_serial_ports())
    emit_log("Took serial port snapshot. Unplug the arm cable now, then click OK/Finish.")
    return jsonify({"status": "snapshot_taken", "ports": list(state["before_ports"])})


@app.route("/api/ports/diff", methods=["POST"])
def ports_diff():
    """Diff current serial ports against snapshot to detect the unplugged port."""
    from run_real import find_serial_ports
    if state["before_ports"] is None:
        return jsonify({"error": "No snapshot taken. Start over."}), 400

    after = set(find_serial_ports())
    diff = state["before_ports"] - after
    state["before_ports"] = None  # Reset

    if len(diff) == 1:
        port = diff.pop()
        state["cfg"].port = port
        state["calibration"].last_port = port
        
        # Persist immediately
        cal_path = os.path.join(os.path.dirname(__file__), state["cfg"].calibration_file)
        state["calibration"].save(cal_path)
        
        emit_log(f"Detected arm on port: {port}. Plug cable back in now!")
        return jsonify({"detected": True, "port": port})
    elif not diff:
        emit_log("No port disappeared. Make sure you unplugged the right cable.")
        return jsonify({"detected": False, "error": "No port disappeared."})
    else:
        ports_list = sorted(list(diff))
        emit_log(f"Multiple ports disappeared: {ports_list}")
        return jsonify({"detected": False, "error": f"Multiple ports disappeared: {ports_list}"})


@app.route("/api/settings", methods=["GET"])
def get_settings():
    cfg = state["cfg"]
    cal = state["calibration"]
    return jsonify({
        "port": cal.last_port or cfg.port,
        "camera_index": cfg.camera_index,
        "aruco_ws_marker_size": cfg.aruco_ws_marker_size,
        "aruco_robot_marker_size": cfg.aruco_robot_marker_size,
        "gripper_marker_id": cfg.gripper_marker_id,
        "wrist_marker_id": cfg.wrist_marker_id,
        "grasp_z": cfg.grasp_z,
        "workspace_marker_ids": list(cfg.workspace_marker_ids),
        "workspace_marker_positions": {
            str(k): list(v) if isinstance(v, (list, np.ndarray)) else v
            for k, v in (cfg.workspace_marker_positions or {}).items()
        },
    })


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.json or {}
    cfg = state["cfg"]
    cal = state["calibration"]
    
    if "port" in data:
        cfg.port = data["port"]
        cal.last_port = data["port"]
    if "aruco_ws_marker_size" in data:
        cfg.aruco_ws_marker_size = float(data["aruco_ws_marker_size"])
    if "aruco_robot_marker_size" in data:
        cfg.aruco_robot_marker_size = float(data["aruco_robot_marker_size"])
    if "gripper_marker_id" in data:
        cfg.gripper_marker_id = int(data["gripper_marker_id"])
    if "wrist_marker_id" in data:
        cfg.wrist_marker_id = int(data["wrist_marker_id"])
    if "grasp_z" in data:
        cfg.grasp_z = float(data["grasp_z"])
    if "workspace_marker_positions" in data:
        cfg.workspace_marker_positions = {
            int(k): list(v) for k, v in data["workspace_marker_positions"].items()
        }
        
    cal_path = os.path.join(os.path.dirname(__file__), cfg.calibration_file)
    cal.save(cal_path)
    
    emit_log("Settings updated.")
    return jsonify({"status": "updated"})


# ---------------------------------------------------------------------------
# Robot control endpoints
# ---------------------------------------------------------------------------

@app.route("/api/capture_background", methods=["POST"])
def capture_bg():
    if state["rig"] is None:
        return jsonify({"error": "Not connected"}), 400
    state["background"] = state["rig"].capture_background()
    emit_log("Background captured.")
    return jsonify({"status": "captured"})


@app.route("/api/reset_arm", methods=["POST"])
def reset_arm():
    if state["rig"] is None or state["episode_running"]:
        return _busy_error()
    if state["home_q"] is None:
        return jsonify({"error": "No home position"}), 400

    def worker():
        state["episode_running"] = True
        try:
            rig = state["rig"]
            q = rig.get_q()
            home = state["home_q"]
            
            # Close the gripper first to avoid hitting anything
            rig.set_gripper(0.0)
            time.sleep(0.5)
            
            rig.set_q(home)
            emit_log("Arm reset complete.")
        except Exception as e:
            emit_log(f"Reset failed: {e}")
        finally:
            state["episode_running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "resetting"})


@app.route("/api/estop", methods=["POST"])
def estop_arm():
    emit_log("🚨 EMERGENCY STOP TRIGGERED! Disabling torques and disconnecting...")
    state["episode_running"] = False
    
    if state["rig"] is not None:
        rig = state["rig"]
        cfg = state["cfg"]
        
        # Try to write Torque_Enable = 0 to Feetech motors directly
        try:
            for j in cfg.joints:
                rig.robot.bus.write("Torque_Enable", 0, j)
            rig.robot.bus.write("Torque_Enable", 0, "gripper")
        except Exception:
            pass
            
        try:
            rig.close()  # disconnects the serial port
        except Exception:
            pass
            
        state["rig"] = None
        
    emit_log("🚨 Arm safety disabled and disconnected.")
    return jsonify({"status": "estopped"})


@app.route("/api/draw_shape", methods=["POST"])
def draw_shape():
    if state["rig"] is None or state["episode_running"]:
        return _busy_error()
    data = request.json or {}
    shape = data.get("shape", "circle")

    def worker():
        state["episode_running"] = True
        try:
            rig = state["rig"]
            model = state["model"]
            home_q = state["home_q"]

            q_L = np.zeros(5)
            q_L[0] = home_q[0]
            q_L[1] = np.radians(30.0)
            q_L[2] = np.radians(-30.0)
            xyz_L = model.ee(q_L)
            theta = q_L[0]

            v = np.array([-np.sin(theta), np.cos(theta), 0.0])
            w = np.array([0.0, 0.0, 1.0])
            R = 0.1

            if shape == "heart":
                def point(t):
                    dh = np.sin(t) ** 3
                    dv = (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t) + 2.5) / 15.0
                    return xyz_L + 0.1*dh*v + 0.1*dv*w
            else:
                def point(t):
                    return xyz_L + R*np.cos(t)*v + R*np.sin(t)*w

            xyz0 = point(0)
            q_start = rig.get_q()
            q0 = model.solve_ik_xyz(q_start, xyz0, orientation_weight=0.0)

            emit_log(f"Drawing {shape}...")
            rig.set_q(q0)
            time.sleep(0.5)

            steps = 180
            t_arr = np.linspace(0, 6*np.pi, steps)
            q_curr = rig.get_q()
            for t in t_arr:
                xyz_t = point(t)
                q_next = model.solve_ik_xyz(q_curr, xyz_t, orientation_weight=0.0)
                rig.set_q_raw(q_next)
                q_curr = q_next
                time.sleep(0.025)

            q_end = rig.get_q()
            rig.set_q(home_q)

            emit_log(f"{shape.title()} drawing complete.")
        except Exception as e:
            emit_log(f"Draw failed: {e}")
        finally:
            state["episode_running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": f"drawing_{shape}"})


@app.route("/api/test_tracking", methods=["POST"])
def test_tracking():
    """Test ArUco gripper tracking."""
    if state["rig"] is None:
        return jsonify({"error": "Not connected"}), 400

    data = request.json or {}
    marker_id = data.get("marker_id", state["cfg"].gripper_marker_id)

    from perception import locate_by_aruco
    frame = state["rig"].read()
    px = locate_by_aruco(frame, marker_id=marker_id)

    if px is not None:
        emit_log(f"ArUco marker {marker_id} found at [{px[0]:.0f}, {px[1]:.0f}]")
        return jsonify({"found": True, "px": px.tolist()})
    else:
        emit_log(f"ArUco marker {marker_id} not found in frame.")
        return jsonify({"found": False})


@app.route("/api/run_episode", methods=["POST"])
def run_episode_endpoint():
    """Run one pick-and-place episode."""
    if state["rig"] is None or state["episode_running"]:
        return _busy_error()

    data = request.json or {}
    speed = data.get("speed", 1.0)
    probes = data.get("probes", 22)
    tracking_mode = data.get("tracking_mode", "aruco")
    pick_px = data.get("pick_px")
    drop_px = data.get("drop_px")

    if state["background"] is None:
        if tracking_mode == "aruco":
            # Auto-capture current frame as background on the fly
            state["background"] = state["rig"].read()
            emit_log("Auto-captured frame for background (ArUco mode).")
        else:
            return jsonify({"error": "Capture background first"}), 400

    def worker():
        state["episode_running"] = True
        try:
            from control import run_episode
            from nanodet_detector import TABLETOP_CLASSES, NanodetDetector

            rig = state["rig"]
            model = state["model"]
            cfg = state["cfg"]
            cal = state["calibration"]

            scfg = ServoConfig(
                tol_coarse_px=18.0, tol_fine_px=12.0, reject_px=45.0,
                measure_every=3, blink_null_gap_s=0.15,
                use_aruco=(tracking_mode == "aruco"),
                aruco_id=cfg.gripper_marker_id,
                aruco_wrist_id=cfg.wrist_marker_id,
                stiction_comp=True,
                dq_max=0.06 * speed,
                babble_probes=int(probes),
                grasp_z=cfg.grasp_z)
            cfg.max_step_deg = 4.0 * speed

            manual_target = np.array(pick_px) if pick_px else None
            manual_pad = np.array(drop_px) if drop_px else None

            detector = NanodetDetector(class_names=TABLETOP_CLASSES) if manual_target is None else None

            # 1-shot hand-eye offset auto-calibration if workspace calibration is available
            tracker = None
            if tracking_mode == "aruco" and cal.has_intrinsics() and cal.has_workspace():
                from control import scan_wrist_for_marker
                emit_log("Scanning/verifying wrist marker visibility...")
                scan_wrist_for_marker(rig, scfg)
                
                wrist_id = cfg.wrist_marker_id
                wrist_marker = None
                for _ in range(5):
                    fr = rig.read()
                    det = get_aruco_detector()
                    msizes = _marker_sizes_map()
                    markers = det.detect_all(fr, cal.camera_matrix, cal.dist_coeffs,
                                             marker_size=cfg.aruco_ws_marker_size,
                                             marker_sizes=msizes)
                    if wrist_id in markers and markers[wrist_id].tvec is not None:
                        wrist_marker = markers[wrist_id]
                        break
                    time.sleep(0.1)

                if wrist_marker is not None:
                    emit_log(f"Auto-calibrating robot base offset via wrist marker (ID {wrist_id})...")
                    R, _ = cv2.Rodrigues(cal.ws_rvec)
                    wrist_ws = R.T @ (wrist_marker.tvec - cal.ws_tvec)
                    wrist_robot = model.ee(rig.get_q())
                    t_robot_ws = wrist_ws - wrist_robot
                    emit_log(f"Calibrated base offset: [{t_robot_ws[0]:.3f}, {t_robot_ws[1]:.3f}, {t_robot_ws[2]:.3f}]")

                    from fused_tracking import AnalyticalFusedTracker
                    tracker = AnalyticalFusedTracker(
                        model.ee,
                        cal.camera_matrix, cal.dist_coeffs,
                        cal.ws_rvec, cal.ws_tvec,
                        t_robot_ws
                    )
                else:
                    emit_log("Wrist marker not detected at start pose. Falling back to active babble.")

            def debug_log(ev):
                if ev and "msg" in ev:
                    emit_log(ev["msg"])

            res = run_episode(
                rig, model, scfg, state["background"], state["rng"],
                detector=detector, J=state["J"],
                manual_target_px=manual_target, manual_pad_px=manual_pad,
                tracker=tracker, debug=debug_log)

            state["J"] = res["J"]
            emit_log(f"Episode result: {res['reason']}")
        except Exception as e:
            emit_log(f"Episode failed: {e}")
            traceback.print_exc()
        finally:
            state["episode_running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "running"})


@app.route("/api/pixel_to_world", methods=["POST"])
def pixel_to_world():
    """Back-project a pixel to workspace coordinates."""
    cal = state["calibration"]
    if not cal.has_intrinsics() or not cal.has_workspace():
        return jsonify({"error": "Calibration incomplete"}), 400

    data = request.json or {}
    px = np.array(data.get("px", [0, 0]), dtype=np.float64)
    z_plane = data.get("z_plane", 0.0)

    pt = pixel_to_workspace(px, cal.camera_matrix, cal.dist_coeffs,
                            cal.ws_rvec, cal.ws_tvec, z_plane)
    if pt is None:
        return jsonify({"error": "Ray parallel to plane"}), 400
    return jsonify({"world": pt.tolist()})


# ---------------------------------------------------------------------------
# Preview-only mode (no arm connected)
# ---------------------------------------------------------------------------

@app.route("/api/preview/start", methods=["POST"])
def start_preview():
    """Start camera preview without arm connection."""
    data = request.json or {}
    cam_index = data.get("camera_index", 0)

    if state.get("preview_cap") is not None:
        return jsonify({"error": "Preview already running"}), 400

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        return jsonify({"error": f"Cannot open camera {cam_index}"}), 400

    state["preview_cap"] = cap
    state["preview_running"] = True

    def stream():
        detector = get_aruco_detector()
        while state.get("preview_running", False):
            ok, frame = state["preview_cap"].read()
            if not ok:
                time.sleep(0.1)
                continue

            cal = state["calibration"]
            msizes = _marker_sizes_map()
            markers = detector.detect_all(
                frame,
                cal.camera_matrix, cal.dist_coeffs,
                marker_size=state["cfg"].aruco_ws_marker_size,
                marker_sizes=msizes)
            state["last_markers"] = markers

            img = frame.copy()
            detector.draw_markers(img, markers,
                                  draw_axes=cal.has_intrinsics(),
                                  camera_matrix=cal.camera_matrix,
                                  dist_coeffs=cal.dist_coeffs)

            for mid, m in markers.items():
                cx, cy = int(m.center[0]), int(m.center[1])
                label = f"ID:{mid}"
                if mid in state["cfg"].workspace_marker_ids:
                    label += " [WS]"
                elif mid == state["cfg"].gripper_marker_id:
                    label += " [GRIP]"
                elif mid == state["cfg"].wrist_marker_id:
                    label += " [WRIST]"
                cv2.putText(img, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            marker_data = {}
            for mid, m in markers.items():
                marker_data[str(mid)] = {
                    "center": m.center.tolist(),
                    "has_pose": m.tvec is not None,
                    "tvec": m.tvec.tolist() if m.tvec is not None else None,
                }
            socketio.emit("frame", {"image": b64, "markers": marker_data})
            time.sleep(0.066)

    threading.Thread(target=stream, daemon=True).start()
    return jsonify({"status": "preview_started"})


@app.route("/api/preview/stop", methods=["POST"])
def stop_preview():
    state["preview_running"] = False
    cap = state.get("preview_cap")
    if cap is not None:
        cap.release()
        state["preview_cap"] = None
    return jsonify({"status": "preview_stopped"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_web_ui(cfg=None, port=5001):
    if cfg is not None:
        state["cfg"] = cfg
    print(f"Starting web UI at http://localhost:{port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False,
                 allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    run_web_ui()

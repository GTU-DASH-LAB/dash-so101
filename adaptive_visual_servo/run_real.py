"""Real SO-101 hardware runner for adaptive_visual_servo.

Wraps lerobot's SOFollower (SO-100/101 driver) + an OpenCV camera behind the
same duck-typed rig interface control.py already exercises against SimWorld
and pb_sim.PyBulletWorld: get_q/set_q/set_gripper/gripper_contact/read (all
in radians / [0,1] / BGR frames). control.py and perception.py are otherwise
unchanged from sim to real -- only this file talks to lerobot. Drives all 6
motors: 5 arm joints (shoulder_pan/lift, elbow_flex, wrist_flex, wrist_roll)
directly through the visual servo's Jacobian, gripper separately.

Object detection defaults to NanoDet-Plus (nanodet_detector.py) -- a real
trained COCO detector suits the real camera and everyday objects (ball,
bottle, cup, fruit...). `--detector bgsub` falls back to background
subtraction for arbitrary non-COCO objects; `--classes` tunes the allowlist.

*** UNTESTED against real hardware. *** First run checklist:
  - Arm powered, workspace clear, YOUR hand near the power switch.
  - Camera fixed overhead/eye-to-hand, matching the sim's assumption.
  - Position the arm somewhere safe before connecting -- home is wherever
    it's sitting when this script starts (same pattern as ui/server.py's
    Reset), not a hardcoded pose.
  - `--calibrate-gripper` FIRST: config.py's gripper_open_pos/closed_pos and
    load_threshold are placeholders. Run it, watch the printed load values,
    and set real numbers in config.py before trusting grasp detection.
  - Z-descend uses lerobot_ik.PlacoModel against the real downloaded URDF
    (assets/SO101/so101_new_calib.urdf) -- no per-arm length calibration
    needed there; XY stays visually closed regardless of any residual error.
"""

import argparse
import os
import sys
import time

import numpy as np
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RealConfig, ServoConfig
from control import run_episode
from lerobot_ik import PlacoModel
from nanodet_detector import TABLETOP_CLASSES, NanodetDetector

MAX_DISPLAY_W, MAX_DISPLAY_H = 480, 360  # scale the UI's video panel down for display only


class EStopped(Exception):
    """Raised from rig motion/IO calls after an emergency stop. Deliberately
    NOT a RuntimeError: control.py catches RuntimeError for recoverable
    failures (babble etc.) and would otherwise swallow the stop."""


class SO101Rig:
    """rig interface in radians / [0,1] / BGR, converted to the bus's native
    degrees/0-100/RGB at this one boundary so ServoConfig's numbers mean the
    same thing in sim and on hardware.

    stop_event: set by the E-Stop endpoint (another thread). Every motor
    command checks it FIRST and raises EStopped, so a running episode bails
    out at its next command and releases the serial bus -- writing torque-off
    from the E-Stop thread while this thread is mid-sync_write fails with
    'Port is in use!' (observed live)."""

    def __init__(self, cfg: RealConfig):
        from lerobot.cameras.configs import ColorMode
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower

        self.cfg = cfg
        cam = OpenCVCameraConfig(index_or_path=cfg.camera_index, fps=30,
                                 width=640, height=480, color_mode=ColorMode.BGR)
        self.robot = SOFollower(SOFollowerRobotConfig(
            port=cfg.port, id=cfg.robot_id, cameras={"cam": cam},
            max_relative_target=None,  # we apply our own smaller per-step clamp
            use_degrees=True))
        # Feetech buses routinely fail the very first write after power-on
        # ("no status packet" on some motor id) and succeed on the next try --
        # observed on the first real connect. One automatic retry.
        try:
            self.robot.connect(calibrate=True)
        except Exception:
            try:
                self.robot.disconnect()
            except Exception:
                pass
            time.sleep(1.0)
            self.robot.connect(calibrate=True)
        self._g = 1.0
        self.stop_event = threading.Event()

    def close(self):
        self.robot.disconnect()

    def _check_stop(self):
        if self.stop_event.is_set():
            raise EStopped("emergency stop")

    # ---- rig interface ----
    def get_q(self):
        self._check_stop()
        pos = self.robot.bus.sync_read("Present_Position", list(self.cfg.joints))
        return np.radians([pos[j] for j in self.cfg.joints])

    def _settle(self, moved, full_range):
        """Settle time proportional to how far this command actually moved,
        not a flat worst-case wait -- most visual-servo steps are tiny."""
        frac = min(abs(moved) / full_range, 1.0) if full_range else 0.0
        time.sleep(self.cfg.settle_floor_s + frac * self.cfg.settle_full_move_s)

    def set_q(self, q):
        self._check_stop()
        c = self.cfg
        q = np.clip(q, np.radians(c.q_min_deg), np.radians(c.q_max_deg))
        deg_before = np.degrees(self.get_q())
        target_deg = np.degrees(q)
        max_diff = np.max(np.abs(target_deg - deg_before))

        # Smoothly interpolate large movements (e.g. initial approaches or transitions)
        if max_diff > 1.5:
            # 1.0 degree per step max, run at ~30Hz
            steps = int(max(5, min(45, max_diff / 1.0)))
            for a in np.linspace(1.0/steps, 1.0, steps):
                self._check_stop()  # abort long interpolations within ~33ms
                interp_deg = deg_before + a * (target_deg - deg_before)
                action = {f"{j}.pos": float(v) for j, v in zip(c.joints, interp_deg)}
                self.robot.send_action(action)
                time.sleep(0.033)
        else:
            action = {f"{j}.pos": float(v) for j, v in zip(c.joints, target_deg)}
            self.robot.send_action(action)
            self._settle(max_diff, c.max_step_deg)

    def set_q_raw(self, q):
        self._check_stop()
        c = self.cfg
        q = np.clip(q, np.radians(c.q_min_deg), np.radians(c.q_max_deg))
        deg = np.degrees(q)
        action = {f"{j}.pos": float(v) for j, v in zip(c.joints, deg)}
        self.robot.send_action(action)

    def set_gripper(self, g):
        self._check_stop()
        c = self.cfg
        g_before = self._g
        self._g = float(np.clip(g, 0.0, 1.0))
        pos_before = c.gripper_closed_pos + g_before * (c.gripper_open_pos - c.gripper_closed_pos)
        pos = c.gripper_closed_pos + self._g * (c.gripper_open_pos - c.gripper_closed_pos)
        self.robot.send_action({"gripper.pos": float(pos)})
        # gripper gets its own (longer) full-move settle: measured on this arm,
        # a full sweep is still in flight at 0.5s -- much slower than a clamped
        # 4-degree arm step
        frac = min(abs(pos - pos_before) / abs(c.gripper_open_pos - c.gripper_closed_pos), 1.0)
        time.sleep(c.settle_floor_s + frac * c.gripper_settle_full_s)

    def gripper_contact(self):
        # An object between the jaws physically STOPS them short of the close
        # command; empty air lets them arrive (measured: within ~2 normalized
        # units). Position error is the primary signal because instantaneous
        # load cannot distinguish 'working hard to move' from 'squeezing':
        # measured +500 mid-travel on EMPTY AIR (same as the torque limit).
        # The signed-load check (closing/holding = positive on this arm)
        # confirms the jaw is actively pressing, not just parked.
        self._check_stop()
        c = self.cfg
        if self._g >= 0.5:
            return False
        cmd = c.gripper_closed_pos + self._g * (c.gripper_open_pos - c.gripper_closed_pos)
        pos = self.robot.bus.read("Present_Position", "gripper")
        toward_open = 1.0 if c.gripper_open_pos > c.gripper_closed_pos else -1.0
        stalled = (pos - cmd) * toward_open > c.grasp_stall_gap
        load = self.robot.bus.read("Present_Load", "gripper")
        return stalled and load > c.load_threshold

    def read(self):
        return self.robot.cameras["cam"].read_latest()

    def capture_background(self):
        return self.read()


class _FrameTap:
    """Wraps a rig so only the streaming thread reads the camera.
    The controller thread blocks on a condition variable until a fresh
    frame is available. This avoids multi-threaded camera access races
    and keeps the video feed updated at 30Hz even during episodes.
    """

    def __init__(self, rig):
        self._rig = rig
        self.last_frame = None
        self.cond = threading.Condition()
        self.new_frame_flag = False

    def read(self):
        # Wait for the streaming thread to read a new frame
        with self.cond:
            self.new_frame_flag = False
            # Wait up to 250ms for a fresh frame; if timeout, return latest cached
            self.cond.wait(timeout=0.25)
            return self.last_frame

    def produce_frame(self):
        # Called by the UI streaming thread at 30Hz to grab a fresh frame
        frame = self._rig.read()
        with self.cond:
            self.last_frame = frame
            self.new_frame_flag = True
            self.cond.notify_all()
        return frame

    def __getattr__(self, name):
        return getattr(self._rig, name)


def calibrate_gripper(rig: SO101Rig):
    """Cycle the gripper and print Present_Load so you can pick real
    gripper_open_pos/gripper_closed_pos/load_threshold values for config.py."""
    print("Watch these values: open/closed positions and the load spike when "
          "fingers meet resistance (press an object, or just close on nothing "
          "to see the free-swing baseline). Ctrl-C to stop.")
    g = 1.0
    try:
        while True:
            rig.set_gripper(g)
            load = rig.robot.bus.read("Present_Load", "gripper")
            print(f"g={g:.2f} load={load}")
            time.sleep(0.5)
            g = 0.0 if g > 0.5 else 1.0
    except KeyboardInterrupt:
        rig.set_gripper(1.0)


def find_serial_ports():
    from serial.tools import list_ports
    return [p.device for p in list_ports.comports()]


def detect_arm_port_dialog(root):
    """Same approach as lerobot's own lerobot-find-port: SO-101 uses a
    generic USB-serial chip, so the only reliable way to identify its port is
    to snapshot ports, have you unplug it, and diff. Returns the port string,
    or None if it couldn't be determined uniquely."""
    from tkinter import messagebox
    before = set(find_serial_ports())
    messagebox.showinfo(
        "Detect arm port",
        "Now unplug the arm's USB cable, then click OK.")
    after = set(find_serial_ports())
    diff = before - after
    if len(diff) == 1:
        port = diff.pop()
        messagebox.showinfo("Detect arm port",
                            f"Found it: {port}\nPlug the cable back in, then click OK.")
        return port
    if not diff:
        messagebox.showerror("Detect arm port",
                             "No port disappeared -- was the right cable unplugged?")
    else:
        messagebox.showerror("Detect arm port",
                             f"More than one port disappeared: {sorted(diff)}")
    return None


class RealUI:
    """Interactive front end for run_real.py: browse cameras, auto-detect the
    arm's USB port, and either let NanoDet pick the target automatically or
    click a pixel yourself -- pick target and drop point are independent, so
    you can mix manual and NanoDet-driven picks. Either way it's still full
    adaptive control underneath (babble/Broyden/visual servo): a clicked
    pixel is exactly as valid a target as a detected one, run_episode never
    needs to know how the pixel was chosen.

    The window is unresponsive while an episode runs -- deliberate: robot
    motor commands only ever come from one thread, never racing the camera
    preview loop.
    """

    def __init__(self, root, cfg=None):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root = root
        root.title("SO-101 adaptive visual servo")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cfg = cfg if cfg is not None else RealConfig()
        self.rig = None
        self.model = None
        self.detector = None
        self._detector_classes = None  # what class_names the loaded detector was built with
        self.background = None
        self.J = None
        self.preview_cap = None       # standalone cv2.VideoCapture, pre-connect only
        self.last_frame = None
        self.disp_size = (1, 1)
        self.click_mode = None        # None | "pick" | "drop"
        self.pick_px = None
        self.drop_px = None
        self.last_debug = {}
        self.episode_running = False
        self._episode_result = None

        self._build_ui()
        self.refresh_cameras()

    # ---------- UI construction ----------
    def _build_ui(self):
        tk, ttk = self.tk, self.ttk
        conn = ttk.LabelFrame(self.root, text="Connection", padding=8)
        conn.pack(fill="x", padx=8, pady=(8, 4))

        ttk.Label(conn, text="Camera:").grid(row=0, column=0, sticky="w")
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(conn, textvariable=self.cam_var, state="readonly", width=20)
        self.cam_combo.grid(row=0, column=1, padx=4)
        ttk.Button(conn, text="Refresh", command=self.refresh_cameras).grid(row=0, column=2, padx=2)
        self.preview_btn = ttk.Button(conn, text="Preview", command=self.toggle_preview)
        self.preview_btn.grid(row=0, column=3, padx=2)

        ttk.Label(conn, text="Arm port:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.port_var = tk.StringVar(value=self.cfg.port)
        ttk.Entry(conn, textvariable=self.port_var, width=22).grid(row=1, column=1, pady=(6, 0))
        ttk.Button(conn, text="Detect Port",
                  command=self.on_detect_port).grid(row=1, column=2, pady=(6, 0))
        self.connect_btn = ttk.Button(conn, text="Connect Arm", command=self.toggle_connect)
        self.connect_btn.grid(row=1, column=3, pady=(6, 0))

        det = ttk.LabelFrame(self.root, text="Target selection", padding=8)
        det.pack(fill="x", padx=8, pady=4)
        self.nanodet_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(det, text="Auto-detect pick target with NanoDet",
                       variable=self.nanodet_var).grid(row=0, column=0, columnspan=2, sticky="w")
        self.classes_var = tk.StringVar(value="tabletop")
        ttk.Label(det, text="Classes:").grid(row=0, column=2, sticky="e", padx=(12, 4))
        ttk.Combobox(det, textvariable=self.classes_var, state="readonly", width=10,
                    values=["tabletop", "all"]).grid(row=0, column=3)
        ttk.Button(det, text="Set Pick Target (click video)",
                  command=lambda: self.arm_click("pick")).grid(row=1, column=0, pady=(6, 0), sticky="w")
        ttk.Button(det, text="Set Drop Location (click video)",
                  command=lambda: self.arm_click("drop")).grid(row=1, column=1, pady=(6, 0), sticky="w")
        ttk.Button(det, text="Clear points", command=self.clear_points).grid(
            row=1, column=2, pady=(6, 0))
        self.points_var = tk.StringVar(value="pick: (manual click needed)  drop: (color pad)")
        ttk.Label(det, textvariable=self.points_var).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # Speed & Learning Tuning
        tuning = ttk.LabelFrame(self.root, text="Speed & Learning Tuning", padding=8)
        tuning.pack(fill="x", padx=8, pady=4)
        
        ttk.Label(tuning, text="Arm Speed:").grid(row=0, column=0, sticky="w")
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_slider = tk.Scale(tuning, from_=1.0, to=4.0, resolution=0.5, orient="horizontal",
                                     variable=self.speed_var, command=self._on_speed_change, showvalue=False)
        self.speed_slider.grid(row=0, column=1, padx=4, sticky="ew")
        self.speed_label = ttk.Label(tuning, text="1.0x (dq_max=0.06rad, max_step=4.0deg)")
        self.speed_label.grid(row=0, column=2, padx=4, sticky="w")
        
        ttk.Label(tuning, text="Babble Probes:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.probes_var = tk.IntVar(value=22)
        self.probes_slider = tk.Scale(tuning, from_=10, to=30, resolution=1, orient="horizontal",
                                      variable=self.probes_var, command=self._on_probes_change, showvalue=False)
        self.probes_slider.grid(row=1, column=1, padx=4, pady=(6, 0), sticky="ew")
        self.probes_label = ttk.Label(tuning, text="22 probes")
        self.probes_label.grid(row=1, column=2, padx=4, pady=(6, 0), sticky="w")
        
        tuning.grid_columnconfigure(1, weight=1)

        # Gripper Tracking Mode
        track_frame = ttk.LabelFrame(self.root, text="Gripper Tracking Mode", padding=8)
        track_frame.pack(fill="x", padx=8, pady=4)
        
        self.track_mode_var = tk.StringVar(value="blink")
        
        ttk.Radiobutton(track_frame, text="Sync-Blink (Motion Diff)", variable=self.track_mode_var,
                        value="blink", command=self._on_track_mode_change).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(track_frame, text="HSV Color Marker", variable=self.track_mode_var,
                        value="marker", command=self._on_track_mode_change).grid(row=0, column=1, sticky="w", padx=12)
        ttk.Radiobutton(track_frame, text="AruCo Fiducial Marker", variable=self.track_mode_var,
                        value="aruco", command=self._on_track_mode_change).grid(row=0, column=2, sticky="w")
        
        # Sub-frame for HSV Color Marker controls
        self.hsv_frame = ttk.Frame(track_frame)
        self.hsv_preview_var = tk.BooleanVar(value=False)
        self.hsv_preview_cb = ttk.Checkbutton(self.hsv_frame, text="Show Binary Segmentation Mask in Video Feed",
                                              variable=self.hsv_preview_var)
        self.hsv_preview_cb.grid(row=0, column=0, columnspan=3, sticky="w", pady=(4, 6))

        # Hue slider
        ttk.Label(self.hsv_frame, text="Hue:").grid(row=1, column=0, sticky="w")
        self.hue_var = tk.IntVar(value=self.cfg.gripper_marker_hue if self.cfg.gripper_marker_hue is not None else 60)
        self.hue_slider = tk.Scale(self.hsv_frame, from_=0, to=179, orient="horizontal", variable=self.hue_var, showvalue=True)
        self.hue_slider.grid(row=1, column=1, padx=4, sticky="ew")

        # Tolerance slider
        ttk.Label(self.hsv_frame, text="Tolerance:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.tol_var = tk.IntVar(value=14)
        self.tol_slider = tk.Scale(self.hsv_frame, from_=1, to=50, orient="horizontal", variable=self.tol_var, showvalue=True)
        self.tol_slider.grid(row=2, column=1, padx=4, pady=(4, 0), sticky="ew")

        # Saturation slider
        ttk.Label(self.hsv_frame, text="Min Saturation:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.sat_var = tk.IntVar(value=60)
        self.sat_slider = tk.Scale(self.hsv_frame, from_=0, to=255, orient="horizontal", variable=self.sat_var, showvalue=True)
        self.sat_slider.grid(row=3, column=1, padx=4, pady=(4, 0), sticky="ew")

        # Value slider
        ttk.Label(self.hsv_frame, text="Min Value:").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self.val_var = tk.IntVar(value=50)
        self.val_slider = tk.Scale(self.hsv_frame, from_=0, to=255, orient="horizontal", variable=self.val_var, showvalue=True)
        self.val_slider.grid(row=4, column=1, padx=4, pady=(4, 0), sticky="ew")
        self.hsv_frame.grid_columnconfigure(1, weight=1)

        # Sub-frame for AruCo controls
        self.aruco_frame = ttk.Frame(track_frame)
        ttk.Label(self.aruco_frame, text="Marker ID:").grid(row=0, column=0, sticky="w")
        self.aruco_id_var = tk.IntVar(value=0)
        self.aruco_id_spin = ttk.Spinbox(self.aruco_frame, from_=0, to=999, textvariable=self.aruco_id_var, width=6)
        self.aruco_id_spin.grid(row=0, column=1, padx=6)

        # Kinematic Motion & Shapes
        shapes_frame = ttk.LabelFrame(self.root, text="Kinematic Motion & Shapes (Inverse Kinematics)", padding=8)
        shapes_frame.pack(fill="x", padx=8, pady=4)
        
        self.reset_btn = ttk.Button(shapes_frame, text="Reset Arm", command=self.on_reset_arm)
        self.reset_btn.pack(side="left", padx=4)
        
        self.circle_btn = ttk.Button(shapes_frame, text="Draw Circle", command=self.on_draw_circle)
        self.circle_btn.pack(side="left", padx=4)
        
        self.heart_btn = ttk.Button(shapes_frame, text="Draw Heart", command=self.on_draw_heart)
        self.heart_btn.pack(side="left", padx=4)

        run = ttk.LabelFrame(self.root, text="Run", padding=8)
        run.pack(fill="x", padx=8, pady=4)
        self.bg_btn = ttk.Button(run, text="Capture Background",
                                 command=self.on_capture_background)
        self.bg_btn.pack(side="left")
        self.blink_btn = ttk.Button(run, text="Test Blink", command=self.on_test_blink)
        self.blink_btn.pack(side="left", padx=6)
        self.run_btn = ttk.Button(run, text="Run One Episode", command=self.on_run_episode)
        self.run_btn.pack(side="left", padx=6)
        self.bg_var = tk.StringVar(value="background: not captured")
        ttk.Label(run, textvariable=self.bg_var).pack(side="left", padx=12)

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(padx=8, pady=4)
        self.video_label.bind("<Button-1>", self._on_click)

        self.log = tk.Text(self.root, height=8, width=90, state="disabled")
        self.log.pack(fill="both", padx=8, pady=(0, 8), expand=True)
        
        self._on_track_mode_change()

    def _on_track_mode_change(self):
        mode = self.track_mode_var.get()
        if mode == "blink":
            self.hsv_frame.grid_forget()
            self.aruco_frame.grid_forget()
            self.hsv_preview_var.set(False)
            if hasattr(self, "blink_btn"):
                self.blink_btn.config(text="Test Blink")
        elif mode == "marker":
            self.aruco_frame.grid_forget()
            self.hsv_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))
            if hasattr(self, "blink_btn"):
                self.blink_btn.config(text="Test Marker")
        elif mode == "aruco":
            self.hsv_frame.grid_forget()
            self.hsv_preview_var.set(False)
            self.aruco_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))
            if hasattr(self, "blink_btn"):
                self.blink_btn.config(text="Test AruCo")

    def _disable_buttons(self):
        for b in (self.connect_btn, self.preview_btn, self.bg_btn, self.blink_btn,
                  self.run_btn, self.reset_btn, self.circle_btn, self.heart_btn):
            if b is not None:
                b.config(state="disabled")

    def _enable_buttons(self):
        for b in (self.connect_btn, self.preview_btn, self.bg_btn, self.blink_btn,
                  self.run_btn, self.reset_btn, self.circle_btn, self.heart_btn):
            if b is not None:
                b.config(state="normal")

    def on_reset_arm(self):
        if self.rig is None or self.episode_running:
            self.log_line("Connect the arm first (and wait for any running episode).")
            return
        if not hasattr(self, "home_q") or self.home_q is None:
            self.log_line("Home position not captured (should be saved on connection).")
            return
        self.log_line("Resetting arm to starting home pose...")
        
        def reset_worker():
            try:
                self.episode_running = True
                self._disable_buttons()
                q = self.rig.get_q()
                for a in np.linspace(0.2, 1.0, 15):
                    self.rig.set_q(q + a * (self.home_q - q))
                    time.sleep(0.08)
                self.log_line("Reset complete.")
            except Exception as e:
                self.log_line(f"Reset failed: {e}")
            finally:
                self.episode_running = False
                self._enable_buttons()
                
        import threading
        threading.Thread(target=reset_worker, daemon=True).start()

    def on_draw_circle(self):
        if self.rig is None or self.episode_running:
            self.log_line("Connect the arm first (and wait for any running episode).")
            return
        self.log_line("Moving to shape starting coordinate in front of the arm...")
        
        def circle_worker():
            try:
                self.episode_running = True
                self._disable_buttons()
                
                # 1. Define center pose: a safe, comfortable upright pose high in the air
                # Keep Joint 1 pan matching self.home_q[0] so it points forward, but set Joint 2 (lift)=30 deg, Joint 3 (elbow)=-30 deg, Joint 4=0 deg
                q_L = np.zeros(5)
                q_L[0] = self.home_q[0]
                q_L[1] = np.radians(30.0)
                q_L[2] = np.radians(-30.0)
                q_L[3] = 0.0
                q_L[4] = 0.0
                xyz_L = self.model.ee(q_L)
                theta = q_L[0]
                
                # Perpendicular horizontal direction in vertical plane facing the arm (YZ-like plane)
                v = np.array([-np.sin(theta), np.cos(theta), 0.0])
                # Vertical direction
                w = np.array([0.0, 0.0, 1.0])
                
                R = 0.1  # 4 cm radius
                
                # 2. Move smoothly from current pose to start point of the circle (t = 0)
                xyz0 = xyz_L + R * v
                q_start = self.rig.get_q()
                q0 = self.model.solve_ik_xyz(q_start, xyz0, orientation_weight=0.0)
                
                self.log_line("Interpolating to circle starting coordinate...")
                for a in np.linspace(0.0, 1.0, 40):
                    self.rig.set_q_raw(q_start + a * (q0 - q_start))
                    time.sleep(0.025)
                time.sleep(0.2)
                
                # 3. Trace the circle in the vertical space facing the robot
                self.log_line("Tracing circle...")
                steps = 3*60
                t_arr = np.linspace(0, 6 * np.pi, steps)
                q_curr = self.rig.get_q()
                for t in t_arr:
                    xyz_t = xyz_L + R * np.cos(t) * v + R * np.sin(t) * w
                    q_next = self.model.solve_ik_xyz(q_curr, xyz_t, orientation_weight=0.0)
                    self.rig.set_q_raw(q_next)
                    q_curr = q_next
                    time.sleep(0.025)
                
                # 4. Return smoothly to home
                self.log_line("Returning to home pose...")
                q_end = self.rig.get_q()
                for a in np.linspace(0.0, 1.0, 45):
                    self.rig.set_q_raw(q_end + a * (self.home_q - q_end))
                    time.sleep(0.025)
                    
                self.log_line("Circle drawing complete.")
            except Exception as e:
                self.log_line(f"Drawing failed: {e}")
            finally:
                self.episode_running = False
                self._enable_buttons()
                
        import threading
        threading.Thread(target=circle_worker, daemon=True).start()

    def on_draw_heart(self):
        if self.rig is None or self.episode_running:
            self.log_line("Connect the arm first (and wait for any running episode).")
            return
        self.log_line("Moving to shape starting coordinate in front of the arm...")
        
        def heart_worker():
            try:
                self.episode_running = True
                self._disable_buttons()
                
                # 1. Define center pose: a safe, comfortable upright pose high in the air
                # Keep Joint 1 pan matching self.home_q[0] so it points forward, but set Joint 2 (lift)=30 deg, Joint 3 (elbow)=-30 deg, Joint 4=0 deg
                q_L = np.zeros(5)
                q_L[0] = self.home_q[0]
                q_L[1] = np.radians(30.0)
                q_L[2] = np.radians(-30.0)
                q_L[3] = 0.0
                q_L[4] = 0.0
                xyz_L = self.model.ee(q_L)
                theta = q_L[0]
                
                # Perpendicular horizontal direction in vertical plane facing the arm (YZ-like plane)
                v = np.array([-np.sin(theta), np.cos(theta), 0.0])
                # Vertical direction
                w = np.array([0.0, 0.0, 1.0])
                
                R = 0.1  # 4 cm radius
                
                # 2. Move smoothly from current pose to start point of the circle (t = 0)
                xyz0 = xyz_L + R * v
                q_start = self.rig.get_q()
                q0 = self.model.solve_ik_xyz(q_start, xyz0, orientation_weight=0.0)
                
                self.log_line("Interpolating to circle starting coordinate...")
                for a in np.linspace(0.0, 1.0, 40):
                    self.rig.set_q_raw(q_start + a * (q0 - q_start))
                    time.sleep(0.025)
                time.sleep(0.2)
                
                scale_h = 0.1
                scale_v = 0.1
                
                def heart_point(t):
                    dh = np.sin(t) ** 3
                    dv = (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t) + 2.5) / 15.0
                    return xyz_L + scale_h * dh * v + scale_v * dv * w
                
                # 3. Trace the heart in the vertical space facing the robot
                self.log_line("Tracing heart...")
                steps = 3*60
                t_arr = np.linspace(0, 6 * np.pi, steps)
                q_curr = self.rig.get_q()
                for t in t_arr:
                    xyz_t = heart_point(t)
                    q_next = self.model.solve_ik_xyz(q_curr, xyz_t, orientation_weight=0.0)
                    self.rig.set_q_raw(q_next)
                    q_curr = q_next
                    time.sleep(0.025)
                    
                # 4. Return smoothly to home
                self.log_line("Returning to home pose...")
                q_end = self.rig.get_q()
                for a in np.linspace(0.0, 1.0, 45):
                    self.rig.set_q_raw(q_end + a * (self.home_q - q_end))
                    time.sleep(0.025)
                    
                self.log_line("Heart drawing complete.")
            except Exception as e:
                self.log_line(f"Drawing failed: {e}")
            finally:
                self.episode_running = False
                self._enable_buttons()
                
        import threading
        threading.Thread(target=heart_worker, daemon=True).start()

    def _on_speed_change(self, val):
        mult = float(val)
        self.speed_label.config(text=f"{mult:.1f}x (dq_max={0.06*mult:.3f}rad, max_step={4.0*mult:.1f}deg)")

    def _on_probes_change(self, val):
        probes = int(val)
        self.probes_label.config(text=f"{probes} probes")

    def log_line(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    # ---------- cameras ----------
    def refresh_cameras(self):
        from camera_nanodet_tester import probe_cameras
        self.log_line("Probing cameras...")
        self.root.update_idletasks()
        self.cameras = probe_cameras()
        values = [f"{i}: {w}x{h}" for i, w, h in self.cameras]
        self.cam_combo["values"] = values
        if values:
            self.cam_combo.current(0)
        self.log_line(f"Found {len(self.cameras)} camera(s).")

    def toggle_preview(self):
        import cv2
        if self.preview_cap is not None:
            self.preview_cap.release()
            self.preview_cap = None
            self.preview_btn.config(text="Preview")
            return
        if self.rig is not None:
            self.log_line("Already connected -- showing the connected camera feed.")
            return
        if not self.cameras or not self.cam_var.get():
            self.log_line("No camera selected -- Refresh first.")
            return
        index = self.cameras[self.cam_combo.current()][0]
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            self.log_line(f"Could not open camera {index}.")
            return
        self.preview_cap = cap
        self.preview_btn.config(text="Stop Preview")
        self._update_frame()

    # ---------- port ----------
    def on_detect_port(self):
        port = detect_arm_port_dialog(self.root)
        if port:
            self.port_var.set(port)
            self.log_line(f"Arm port set to {port}.")

    # ---------- connect ----------
    def toggle_connect(self):
        if self.rig is not None:
            self.rig.close()
            self.rig = None
            self.connect_btn.config(text="Connect Arm")
            self.log_line("Disconnected.")
            return
        if not self.cameras or not self.cam_var.get():
            self.log_line("Select a camera first.")
            return
        if self.preview_cap is not None:  # SO101Rig needs exclusive camera access
            self.preview_cap.release()
            self.preview_cap = None
        self.cfg.port = self.port_var.get()
        self.cfg.camera_index = self.cameras[self.cam_combo.current()][0]
        self.log_line("Connecting -- watch the terminal for lerobot calibration "
                      "prompts if this is a new arm id...")
        self.root.update_idletasks()
        try:
            self.rig = _FrameTap(SO101Rig(self.cfg))
        except Exception as e:  # surfaced to the user, not a crash
            self.log_line(f"Connect failed: {e}")
            return
        self.model = PlacoModel()
        self.home_q = self.rig.get_q()  # capture initial joints as reset home
        self.connect_btn.config(text="Disconnect")
        self.log_line("Connected.")
        self._update_frame()

    # ---------- target/drop selection ----------
    def arm_click(self, mode):
        self.click_mode = mode
        self.log_line(f"Click the video to set the {mode.upper()} point.")

    def clear_points(self):
        self.pick_px = self.drop_px = None
        self._refresh_points_label()

    def _refresh_points_label(self):
        pick = "manual click needed" if self.pick_px is None else np.round(self.pick_px, 0).tolist()
        if self.nanodet_var.get():
            pick = "NanoDet auto-detect"
        drop = "color pad" if self.drop_px is None else np.round(self.drop_px, 0).tolist()
        self.points_var.set(f"pick: {pick}   drop: {drop}")

    def _on_click(self, event):
        if self.click_mode is None or self.last_frame is None:
            return
        raw_h, raw_w = self.last_frame.shape[:2]
        disp_w, disp_h = self.disp_size
        px = np.array([event.x * raw_w / disp_w, event.y * raw_h / disp_h])
        if self.click_mode == "pick":
            self.pick_px = px
        else:
            self.drop_px = px
        self.click_mode = None
        self._refresh_points_label()

    # ---------- background ----------
    def on_capture_background(self):
        if self.rig is None:
            self.log_line("Connect the arm first.")
            return
        self.log_line("Capturing background -- workspace should be clear of objects.")
        self.background = self.rig.capture_background()
        self.bg_var.set("background: captured")

    # ---------- run ----------
    def on_test_blink(self):
        """One gripper blink or color marker detection with full diagnostics."""
        if self.rig is None or self.episode_running:
            self.log_line("Connect the arm first (and wait for any running episode).")
            return
        
        mode = self.track_mode_var.get()
        if mode == "marker":
            from perception import locate_by_marker
            hue = self.hue_var.get()
            tol = self.tol_var.get()
            min_sat = self.sat_var.get()
            min_val = self.val_var.get()
            
            frame = self.rig.read()
            px = locate_by_marker(frame, hue, tol, min_sat, min_val,
                                  self.cfg.gripper_marker_area_min, self.cfg.gripper_marker_area_max)
            if px is not None:
                self.last_debug["s"] = px
                self.log_line(
                    f"Test Marker OK: gripper marker seen at {np.round(px, 0).tolist()} -- yellow cross.")
            else:
                self.log_line(
                    f"Test Marker FAILED: no marker found with hue {hue}. "
                    "Make sure the tape is visible, colors are correct, and lighting is adequate.")
            return
        elif mode == "aruco":
            from perception import locate_by_aruco
            marker_id = self.aruco_id_var.get()
            frame = self.rig.read()
            px = locate_by_aruco(frame, marker_id=marker_id)
            if px is not None:
                self.last_debug["s"] = px
                self.log_line(
                    f"Test AruCo OK: marker ID {marker_id} seen at {np.round(px, 0).tolist()} -- yellow cross.")
            else:
                self.log_line(
                    f"Test AruCo FAILED: no marker with ID {marker_id} found in the camera view.")
            return

        from perception import blink_measure
        scfg = ServoConfig(blink_null_gap_s=0.15)
        px, info = blink_measure(self.rig, scfg)
        if px is not None:
            self.last_debug["s"] = px
            self.log_line(
                f"Test Blink OK: gripper seen at {np.round(px, 0).tolist()} "
                f"({info['n_kept']} blob(s), areas {info['areas']}px^2, "
                f"scene-noise mask {info['noise_px']}px) -- yellow cross.")
        elif info["n_raw"] > info["n_kept"]:
            self.log_line(
                f"Test Blink: only oversized blobs survived the sync filter "
                f"(>{scfg.blink_max_blob}px^2) -- looks like a global "
                "image change, not finger motion. Lock camera exposure if possible.")
        else:
            self.log_line(
                f"Test Blink: no command-synchronized motion found "
                f"(scene-noise mask {info['noise_px']}px). Either the fingers "
                "barely move between blink positions (check gripper_open_pos/"
                "closed_pos) or the gripper is out of the camera's view.")

    def on_run_episode(self):
        if self.episode_running:
            return  # button is disabled during a run, but guard anyway
        if self.rig is None:
            self.log_line("Connect the arm first.")
            return
        if self.background is None:
            self.log_line("Capture the background first.")
            return
        if not self.nanodet_var.get() and self.pick_px is None:
            self.log_line("NanoDet is off -- click 'Set Pick Target' and click the video first.")
            return

        detector = None
        if self.nanodet_var.get():
            classes = None if self.classes_var.get() == "all" else TABLETOP_CLASSES
            if self.detector is None or self._detector_classes != classes:
                self.log_line("Loading NanoDet...")
                self.root.update_idletasks()
                self.detector = NanodetDetector(class_names=classes)
                self._detector_classes = classes
            detector = self.detector

        speed_val = self.speed_var.get()
        probes_val = self.probes_var.get()
        mode = self.track_mode_var.get()
        marker_hue = self.hue_var.get() if mode == "marker" else None
        use_aruco = (mode == "aruco")
        aruco_id = self.aruco_id_var.get()

        rc = self.rig.cfg if self.rig is not None else RealConfig()
        scfg = ServoConfig(tol_coarse_px=18.0, tol_fine_px=12.0, reject_px=45.0,
                          measure_every=3, blink_null_gap_s=0.15,
                          gripper_marker_hue=marker_hue,
                          gripper_marker_sat_min=self.sat_var.get(),
                          gripper_marker_val_min=self.val_var.get(),
                          hue_tol=self.tol_var.get(),
                          use_aruco=use_aruco,
                          aruco_id=aruco_id,
                          stiction_comp=True,
                          dq_max=0.06 * speed_val,
                          babble_probes=int(probes_val),
                          # null-space joint-limit repulsion needs the URDF limits
                          q_lo=tuple(np.radians(rc.q_min_deg)),
                          q_hi=tuple(np.radians(rc.q_max_deg)))
        if self.rig is not None:
            self.rig.cfg.max_step_deg = 4.0 * speed_val
        if not hasattr(self, "rng"):
            self.rng = np.random.default_rng(0)  # persists across episodes, not reseeded each run
        manual_target = None if self.nanodet_var.get() else self.pick_px
        manual_pad = self.drop_px

        # Run on a background thread so the UI (and the live preview, fed by
        # the same frames the controller itself reads via _FrameTap) stays
        # responsive while the arm moves. Only this thread ever calls rig
        # methods during the run -- Connect/Preview/Capture/Run are disabled
        # below so nothing else can touch the camera or motors concurrently.
        self.episode_running = True
        self._episode_result = None
        self._disable_buttons()
        self.log_line("Running episode (window stays responsive; video keeps updating)...")

        def worker():
            try:
                res = run_episode(
                    self.rig, self.model, scfg, self.background, self.rng,
                    detector=detector, J=self.J,
                    manual_target_px=manual_target, manual_pad_px=manual_pad,
                    debug=lambda ev: self.last_debug.update(ev))
            except Exception as e:  # surfaced in the log, not a crash
                res = dict(ok=False, reason=f"exception: {e}", J=self.J)
            self._episode_result = res

        import threading
        threading.Thread(target=worker, daemon=True).start()
        self.root.after(150, self._poll_episode)

    def _poll_episode(self):
        if self._episode_result is None:
            self.root.after(150, self._poll_episode)
            return
        res = self._episode_result
        self.J = res["J"]
        self.log_line(f"Result: {res['reason']!r}")
        self.episode_running = False
        self._enable_buttons()

    # ---------- live preview ----------
    def _update_frame(self):
        from PIL import Image, ImageTk
        import cv2
        if self.rig is not None and self.episode_running:
            # an episode is running on a background thread and already
            # calling rig.read() itself -- show its actual frames (cached by
            # _FrameTap) instead of reading the camera again from this
            # thread too, which would race the episode thread
            frame = self.rig.last_frame
        elif self.rig is not None:
            frame = self.rig.read()  # rig.read() returns the frame directly
        elif self.preview_cap is not None:
            ok, frame = self.preview_cap.read()  # raw cv2.VideoCapture: (ok, frame)
            if not ok:
                frame = None
        else:
            return
        if frame is None:
            self.root.after(200, self._update_frame)
            return
        self.last_frame = frame
        mode = self.track_mode_var.get()
        if mode == "marker" and self.hsv_preview_var.get():
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue = self.hue_var.get()
            tol = self.tol_var.get()
            min_sat = self.sat_var.get()
            min_val = self.val_var.get()
            lo, hi = hue - tol, hue + tol
            if lo < 0:
                mask = cv2.inRange(hsv, (0, min_sat, min_val), (hi, 255, 255)) | \
                       cv2.inRange(hsv, (180 + lo, min_sat, min_val), (179, 255, 255))
            elif hi > 179:
                mask = cv2.inRange(hsv, (lo, min_sat, min_val), (179, 255, 255)) | \
                       cv2.inRange(hsv, (0, min_sat, min_val), (hi - 180, 255, 255))
            else:
                mask = cv2.inRange(hsv, (lo, min_sat, min_val), (hi, 255, 255))
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
            img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            img = frame.copy()
            if mode == "aruco":
                try:
                    dictionary_id = cv2.aruco.DICT_4X4_50
                    try:
                        # OpenCV 4.7.0+ API
                        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
                        parameters = cv2.aruco.DetectorParameters()
                        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
                        corners, ids, rejected = detector.detectMarkers(frame)
                    except AttributeError:
                        # Older OpenCV API
                        dictionary = cv2.aruco.Dictionary_get(dictionary_id)
                        parameters = cv2.aruco.DetectorParameters_create()
                        corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
                    
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(img, corners, ids)
                except Exception:
                    pass
        if self.pick_px is not None:
            cv2.drawMarker(img, tuple(np.int32(self.pick_px)), (0, 0, 255),
                           cv2.MARKER_CROSS, 18, 2)
        if self.drop_px is not None:
            cv2.drawMarker(img, tuple(np.int32(self.drop_px)), (200, 0, 200),
                           cv2.MARKER_TILTED_CROSS, 18, 2)
        s = self.last_debug.get("s")
        if s is not None:
            cv2.drawMarker(img, tuple(np.int32(s)), (0, 220, 255), cv2.MARKER_CROSS, 14, 2)
        # markers are drawn in raw-frame coordinates above; resize for display
        # AFTER drawing, and record the resized size so _on_click's raw_w/disp_w
        # math maps clicks back to raw-frame pixels correctly
        h, w = img.shape[:2]
        scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
        if scale < 0.999:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        dh, dw = img.shape[:2]
        self.disp_size = (dw, dh)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.config(image=self._imgtk)
        self._refresh_points_label()
        self.root.after(30, self._update_frame)

    def on_close(self):
        if self.episode_running:
            from tkinter import messagebox
            messagebox.showwarning(
                "Episode running",
                "An episode is still running on the arm -- wait for it to "
                "finish before closing (closing now would disconnect the "
                "hardware while it's still moving).")
            return
        if self.preview_cap is not None:
            self.preview_cap.release()
        if self.rig is not None:
            self.rig.set_gripper(1.0)
            self.rig.close()
        self.root.destroy()


def run_ui(cfg):
    import tkinter as tk
    root = tk.Tk()
    RealUI(root, cfg)
    root.mainloop()


def run_web_ui(cfg):
    from web_ui import run_web_ui as start_server
    start_server(cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cfg = RealConfig()
    ap.add_argument("--ui", action="store_true",
                    help="launch the beautiful React-based Web UI (default)")
    ap.add_argument("--tk-ui", action="store_true",
                    help="launch the legacy Tkinter desktop UI")
    ap.add_argument("--port", default=cfg.port)
    ap.add_argument("--camera", type=int, default=cfg.camera_index)
    ap.add_argument("--id", default=cfg.robot_id)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--hue", type=int, default=None,
                    help="only pick the object with this OpenCV hue (0-179)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate-gripper", action="store_true")
    ap.add_argument("--detector", choices=["nanodet", "bgsub"], default="nanodet",
                    help="nanodet (default): real trained COCO detector, best for "
                         "everyday objects on the real camera. bgsub: background "
                         "subtraction, any object but needs a clean background photo")
    ap.add_argument("--classes", default=None,
                    help="comma-separated COCO class allowlist for nanodet "
                         f"(default: {','.join(TABLETOP_CLASSES)}; 'all' = all 80)")
    ap.add_argument("--gripper-marker-hue", type=int, default=None,
                    help="OpenCV Hue (0-179) of the color marker tape on the gripper tip "
                         "to track it passively/instantaneously without wiggling/blinking.")
    args = ap.parse_args()

    cfg.port, cfg.camera_index, cfg.robot_id = args.port, args.camera, args.id
    cfg.gripper_marker_hue = args.gripper_marker_hue

    if args.ui:
        run_web_ui(cfg)
        return

    if args.tk_ui:
        run_ui(cfg)
        return

    rig = SO101Rig(cfg)
    try:
        if args.calibrate_gripper:
            calibrate_gripper(rig)
            return
        model = PlacoModel()
        # pb_sim testing (same URDF gripper mesh) found the toy sim's tight
        # tol_coarse_px=7/tol_fine_px=4 just chatters forever on real-mesh
        # blink_locate noise -- start looser here too, tune further once you
        # can watch the real camera's actual blink noise on hardware.
        scfg = ServoConfig(tol_coarse_px=18.0, tol_fine_px=12.0, reject_px=45.0,
                           measure_every=3,  # blink 1/3 as often; J dead-reckons between
                           blink_null_gap_s=0.15,
                           gripper_marker_hue=cfg.gripper_marker_hue,
                           stiction_comp=True)
        rng = np.random.default_rng(args.seed)
        detector = None
        if args.detector == "nanodet":
            classes = (None if args.classes == "all" else
                       tuple(args.classes.split(",")) if args.classes else
                       TABLETOP_CLASSES)
            detector = NanodetDetector(class_names=classes)
        # background photo still needed even with nanodet: locate_by_diff
        # (carry-phase tracking) diffs against it regardless of the detector
        input("Workspace clear of objects for the background photo -- "
              "press ENTER when ready...")
        background = rig.capture_background()
        J = None
        for ep in range(args.episodes):
            print(f"episode {ep + 1}/{args.episodes}")
            res = run_episode(rig, model, scfg, background, rng,
                              target_hue=args.hue, detector=detector, J=J)
            J = res["J"]
            print(f"  {res['reason']!r}")
            if res["reason"] in ("no objects detected", "drop pad not found"):
                break
    finally:
        rig.set_gripper(1.0)
        rig.close()


if __name__ == "__main__":
    main()

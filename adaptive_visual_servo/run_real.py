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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RealConfig, ServoConfig
from control import run_episode
from lerobot_ik import PlacoModel
from nanodet_detector import TABLETOP_CLASSES, NanodetDetector

MAX_DISPLAY_W, MAX_DISPLAY_H = 480, 360  # scale the UI's video panel down for display only


class SO101Rig:
    """rig interface in radians / [0,1] / BGR, converted to the bus's native
    degrees/0-100/RGB at this one boundary so ServoConfig's numbers mean the
    same thing in sim and on hardware."""

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

    def close(self):
        self.robot.disconnect()

    # ---- rig interface ----
    def get_q(self):
        pos = self.robot.bus.sync_read("Present_Position", list(self.cfg.joints))
        return np.radians([pos[j] for j in self.cfg.joints])

    def _settle(self, moved, full_range):
        """Settle time proportional to how far this command actually moved,
        not a flat worst-case wait -- most visual-servo steps are tiny."""
        frac = min(abs(moved) / full_range, 1.0) if full_range else 0.0
        time.sleep(self.cfg.settle_floor_s + frac * self.cfg.settle_full_move_s)

    def set_q(self, q):
        c = self.cfg
        q = np.clip(q, np.radians(c.q_min_deg), np.radians(c.q_max_deg))
        deg_before = np.degrees(self.get_q())
        step = np.clip(np.degrees(q) - deg_before, -c.max_step_deg, c.max_step_deg)
        deg = deg_before + step
        action = {f"{j}.pos": float(v) for j, v in zip(c.joints, deg)}
        self.robot.send_action(action)
        self._settle(np.max(np.abs(step)), c.max_step_deg)

    def set_gripper(self, g):
        c = self.cfg
        g_before = self._g
        self._g = float(np.clip(g, 0.0, 1.0))
        pos_before = c.gripper_closed_pos + g_before * (c.gripper_open_pos - c.gripper_closed_pos)
        pos = c.gripper_closed_pos + self._g * (c.gripper_open_pos - c.gripper_closed_pos)
        self.robot.send_action({"gripper.pos": float(pos)})
        self._settle(pos - pos_before, abs(c.gripper_open_pos - c.gripper_closed_pos))

    def gripper_contact(self):
        load = abs(self.robot.bus.read("Present_Load", "gripper"))
        return self._g < 0.5 and load > self.cfg.load_threshold

    def read(self):
        return self.robot.cameras["cam"].read_latest()

    def capture_background(self):
        return self.read()


class _FrameTap:
    """Wraps a rig so every rig.read() call also caches the frame in
    .last_frame -- lets the UI's live preview show the ACTUAL frames the
    controller is reading during a background-threaded episode, without the
    UI thread ever touching the camera itself (that would race the episode
    thread's own reads). Everything else passes through unchanged."""

    def __init__(self, rig):
        self._rig = rig
        self.last_frame = None

    def read(self):
        frame = self._rig.read()
        self.last_frame = frame
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

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root = root
        root.title("SO-101 adaptive visual servo")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cfg = RealConfig()
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
        """One gripper blink with full diagnostics -- the first thing to try
        when babble reports unusable probes. Shows whether the camera can see
        the gripper move at all, and how strongly."""
        if self.rig is None or self.episode_running:
            self.log_line("Connect the arm first (and wait for any running episode).")
            return
        from perception import diff_mask, find_blobs
        scfg = ServoConfig()
        g_open, g_mid = scfg.blink_dg
        self.rig.set_gripper(g_open)
        a = self.rig.read()
        self.rig.set_gripper(g_mid)
        b = self.rig.read()
        self.rig.set_gripper(g_open)
        blobs = find_blobs(diff_mask(a, b, scfg.diff_thresh), scfg.min_blob)
        kept = [x for x in blobs if x.area <= scfg.blink_max_blob]
        if not blobs:
            self.log_line(
                "Test Blink: NO pixel change seen between gripper positions. "
                "Either the fingers barely move (check gripper_open_pos/"
                "gripper_closed_pos -- run --calibrate-gripper) or the gripper "
                "is out of the camera's view.")
        elif not kept:
            self.log_line(
                f"Test Blink: only huge diff blobs (largest {blobs[0].area}px^2 > "
                f"blink_max_blob={scfg.blink_max_blob}) -- looks like a global "
                "image change (auto-exposure flicker, something else moving), "
                "not finger motion. Lock the camera's exposure if possible.")
        else:
            w = np.array([x.area for x in kept], float)
            c = np.stack([x.center for x in kept])
            px = (w @ c) / w.sum()
            self.last_debug["s"] = px
            self.log_line(
                f"Test Blink OK: gripper seen at {np.round(px, 0).tolist()} "
                f"({len(kept)} blob(s), areas {[x.area for x in kept]}px^2) -- "
                "marked with the yellow cross.")

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

        scfg = ServoConfig(tol_coarse_px=18.0, tol_fine_px=12.0, reject_px=45.0,
                          measure_every=3)
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
        for b in (self.connect_btn, self.preview_btn, self.bg_btn, self.blink_btn, self.run_btn):
            b.config(state="disabled")
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
        for b in (self.connect_btn, self.preview_btn, self.bg_btn, self.blink_btn, self.run_btn):
            b.config(state="normal")

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
        img = frame.copy()
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


def run_ui():
    import tkinter as tk
    root = tk.Tk()
    RealUI(root)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cfg = RealConfig()
    ap.add_argument("--ui", action="store_true",
                    help="interactive UI: browse cameras, auto-detect the arm port, "
                         "click a pick target and/or drop location, run episodes")
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
    args = ap.parse_args()

    if args.ui:
        run_ui()
        return

    cfg.port, cfg.camera_index, cfg.robot_id = args.port, args.camera, args.id
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
                           measure_every=3)  # blink 1/3 as often; J dead-reckons between
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

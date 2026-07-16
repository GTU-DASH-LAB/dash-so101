"""Standalone camera + NanoDet tester -- a UI to check what cameras this
machine sees and exercise nanodet_detector.py's full detection capacity on a
live feed, independent of the pick-and-place pipeline.

    python adaptive_visual_servo/camera_nanodet_tester.py

- "Refresh" re-probes camera indices 0-9 (OpenCV has no clean device-listing
  API without extra platform-specific deps, so this opens/reads/releases each
  index briefly -- the standard approach, and why the list can flicker other
  apps' camera lights for a moment).
- "Open" starts a live preview of the selected camera.
- "Run NanoDet" toggles live detection overlay (all 80 COCO classes by
  default, or the tabletop subset) with FPS/inference-time/detection-count
  in the status bar.

macOS note: the first run will prompt for camera permission (System
Settings -> Privacy & Security -> Camera) -- grant it to your terminal/Python
and re-run.
"""

import os
import sys
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nanodet_detector import COCO_CLASSES, TABLETOP_CLASSES, NanodetDetector

PROBE_RANGE = 10
DISPLAY_W, DISPLAY_H = 640, 480


def probe_cameras(max_index=PROBE_RANGE):
    """Return [(index, width, height), ...] for indices that actually open
    and deliver a frame."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                found.append((i, w, h))
        cap.release()
    return found


class CameraNanodetApp:
    def __init__(self, root):
        self.root = root
        root.title("Camera + NanoDet Tester")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.detector = None
        self.cameras = []

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Camera:").pack(side="left")
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(top, textvariable=self.cam_var, state="readonly", width=28)
        self.cam_combo.pack(side="left", padx=6)
        ttk.Button(top, text="Refresh", command=self.refresh_cameras).pack(side="left")
        self.open_btn = ttk.Button(top, text="Open", command=self.toggle_camera)
        self.open_btn.pack(side="left", padx=6)

        det_frame = ttk.Frame(root, padding=(8, 0, 8, 8))
        det_frame.pack(fill="x")
        self.run_nanodet = tk.BooleanVar(value=False)
        ttk.Checkbutton(det_frame, text="Run NanoDet", variable=self.run_nanodet,
                        command=self.on_toggle_nanodet).pack(side="left")
        ttk.Label(det_frame, text="Classes:").pack(side="left", padx=(16, 4))
        self.class_var = tk.StringVar(value="all")
        ttk.Combobox(det_frame, textvariable=self.class_var, state="readonly", width=14,
                    values=["all", "tabletop"]).pack(side="left")
        ttk.Label(det_frame, text="Score >").pack(side="left", padx=(16, 4))
        self.score_var = tk.DoubleVar(value=0.35)
        ttk.Scale(det_frame, from_=0.1, to=0.9, variable=self.score_var,
                 orient="horizontal", length=120).pack(side="left")

        self.video_label = ttk.Label(root)
        self.video_label.pack(padx=8, pady=(0, 8))

        self.status_var = tk.StringVar(value="Click Refresh to find cameras.")
        ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w",
                 padding=4).pack(fill="x", side="bottom")

        self._imgtk = None  # keep a reference so tkinter doesn't garbage-collect it
        self._last_t = time.time()
        self.refresh_cameras()

    def refresh_cameras(self):
        self.status_var.set("Probing camera indices 0-9...")
        self.root.update_idletasks()
        self.cameras = probe_cameras()
        if self.cameras:
            values = [f"{i}: {w}x{h}" for i, w, h in self.cameras]
            self.cam_combo["values"] = values
            self.cam_combo.current(0)
            self.status_var.set(f"Found {len(self.cameras)} camera(s).")
        else:
            self.cam_combo["values"] = []
            self.cam_var.set("")
            self.status_var.set(
                "No cameras found. On macOS, grant camera access to your "
                "terminal/Python in System Settings > Privacy & Security > "
                "Camera, then Refresh again.")

    def toggle_camera(self):
        if self.cap is not None:
            self.stop_camera()
            return
        if not self.cameras or not self.cam_var.get():
            self.status_var.set("No camera selected -- Refresh first.")
            return
        index = self.cameras[self.cam_combo.current()][0]
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            self.status_var.set(f"Could not open camera {index}.")
            return
        self.cap = cap
        self.open_btn.config(text="Stop")
        self.cam_combo.config(state="disabled")
        self._last_t = time.time()
        self.update_frame()

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.open_btn.config(text="Open")
        self.cam_combo.config(state="readonly")
        self.video_label.config(image="")
        self.status_var.set("Camera stopped.")

    def on_toggle_nanodet(self):
        if self.run_nanodet.get() and self.detector is None:
            self.status_var.set("Loading NanoDet-Plus (ONNX)...")
            self.root.update_idletasks()
            self.detector = NanodetDetector(class_names=None, score_thresh=0.1)
            self.status_var.set("NanoDet loaded.")

    def _classes(self):
        return None if self.class_var.get() == "all" else TABLETOP_CLASSES

    def update_frame(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set("Camera read failed.")
            self.stop_camera()
            return

        n_det = 0
        infer_ms = 0.0
        if self.run_nanodet.get() and self.detector is not None:
            self.detector.allowed = (None if self._classes() is None else
                                     {COCO_CLASSES.index(c) for c in self._classes()})
            self.detector.score_thresh = self.score_var.get()
            t0 = time.time()
            blobs = self.detector(frame)
            infer_ms = (time.time() - t0) * 1000
            n_det = len(blobs)
            for b in blobs:
                x, y, w, h = b.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
                cv2.circle(frame, tuple(int(v) for v in b.center), 4, (0, 0, 255), -1)

        now = time.time()
        fps = 1.0 / max(now - self._last_t, 1e-6)
        self._last_t = now
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{w}x{h}  {fps:4.1f} FPS", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        if self.run_nanodet.get():
            self.status_var.set(f"NanoDet: {n_det} detection(s), {infer_ms:.0f}ms/frame")
        else:
            self.status_var.set(f"Live -- {w}x{h} @ {fps:.1f} FPS")

        disp = cv2.resize(frame, (DISPLAY_W, DISPLAY_H)) if (w, h) != (DISPLAY_W, DISPLAY_H) else frame
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.config(image=self._imgtk)
        self.root.after(15, self.update_frame)

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    CameraNanodetApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

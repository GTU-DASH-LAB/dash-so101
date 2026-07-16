"""Perception vs. sim ground truth: detection, hue selection, blink, tracking."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ServoConfig
from perception import blink_locate, detect_objects, detect_pad, locate_by_diff, pick_by_hue
from sim import SimWorld

SCFG = ServoConfig()


def test_detect_objects_any_color():
    w = SimWorld(seed=10)
    bg = w.capture_background()
    w.spawn_random(3)
    found = detect_objects(w.read(), bg, SCFG.bg_thresh, SCFG.obj_min_area)
    assert len(found) == 3, f"expected 3 objects, got {len(found)}"
    for o in w.objects:
        true_px = w.cam.project(o.pos)
        d = min(np.linalg.norm(b.center - true_px) for b in found)
        assert d < 6, f"object at {true_px} localized {d:.1f}px off"
    # gray objects must be detected too (seed 10 may not have one; force it)
    w2 = SimWorld(seed=11)
    bg2 = w2.capture_background()
    w2.spawn_object((0.16, 0.02), 0.013, "blob", (90, 90, 90))
    got = detect_objects(w2.read(), bg2, SCFG.bg_thresh, SCFG.obj_min_area)
    assert len(got) == 1, "colorless object missed by bg-sub"


def test_pad_and_hue_selection():
    w = SimWorld(seed=12)
    w.spawn_object((0.17, -0.03), 0.013, "circle", (30, 30, 210))   # red
    w.spawn_object((0.15, 0.06), 0.013, "circle", (40, 200, 40))    # green
    bg = w.capture_background()
    frame = w.read()
    pad_px = detect_pad(frame, SCFG.pad_hue, SCFG.hue_tol)
    assert pad_px is not None
    assert np.linalg.norm(pad_px - w.cam.project(w.pad)) < 6
    found = detect_objects(frame, bg, SCFG.bg_thresh, SCFG.obj_min_area)
    red = pick_by_hue(found, 0)
    green = pick_by_hue(found, 60)
    assert red is not None and green is not None
    assert np.linalg.norm(red.center - w.cam.project(w.objects[0].pos)) < 6
    assert np.linalg.norm(green.center - w.cam.project(w.objects[1].pos)) < 6
    # a hue nothing matches -> None
    assert pick_by_hue(found, 120) is None


def test_blink_locate():
    for seed, dq in [(13, None), (14, np.array([0.5, -0.2, 0.3])),
                     (15, np.array([-0.3, 0.3, -0.4]))]:
        w = SimWorld(seed=seed)
        if dq is not None:
            w.set_q(np.array(w.cfg.home_q) + dq)
        px = blink_locate(w, SCFG)
        assert px is not None, f"blink found nothing (seed {seed})"
        err = np.linalg.norm(px - w.grip_px())
        assert err < 8, f"blink {err:.1f}px off ground truth (seed {seed})"


def test_locate_by_diff_tracks_carried_object():
    w = SimWorld(seed=16)
    bg = w.capture_background()
    (o,) = w.spawn_random(1)
    from sim import solve_ik_true
    q = solve_ik_true(w, np.array([o.pos[0], o.pos[1], w.cfg.grasp_ee_z]))
    w.set_q(q)
    w.set_gripper(0.1)
    assert w.gripper_contact(), "setup: grasp must succeed for this test"
    s = w.grip_px()
    rng = np.random.default_rng(0)
    for _ in range(8):
        w.set_q(w.get_q() + rng.uniform(-0.05, 0.05, 3))
        pred = s  # a real caller would predict via J @ dq; ground truth stands in here
        s = locate_by_diff(w.read(), bg, pred, SCFG.track_roi, SCFG.bg_thresh,
                           SCFG.obj_min_area, SCFG.track_max_jump)
        assert s is not None, "lost the carried object mid-transport"
    err = np.linalg.norm(s - w.grip_px())
    assert err < 20, f"bg-diff drifted {err:.1f}px after 8 moves"


def test_locate_by_diff_rejects_distant_distractor():
    w = SimWorld(seed=17)
    bg = w.capture_background()
    w.spawn_object((0.16, 0.06), 0.013, "circle", (40, 200, 40))  # untouched distractor
    ee_px = w.grip_px()
    far_prediction = ee_px + np.array([300.0, 0.0])  # nowhere near the arm or the distractor
    assert locate_by_diff(w.read(), bg, far_prediction, SCFG.track_roi, SCFG.bg_thresh,
                          SCFG.obj_min_area, SCFG.track_max_jump) is None


def test_diff_mask_ignores_global_exposure_shift():
    """First real-hardware run failed babble with 'only 4 usable probes': the
    webcam's auto-exposure shifted the WHOLE frame between the two blink
    captures, so the raw diff lit up all 307k pixels and the blink centroid
    was frame-center garbage every time. diff_mask now subtracts the median
    diff (the global pedestal) so thresholding is about motion, not lighting."""
    from perception import diff_mask, find_blobs
    rng = np.random.default_rng(0)
    a = rng.integers(80, 170, (480, 640, 3)).astype(np.uint8)
    b = np.clip(a.astype(int) + 12, 0, 255).astype(np.uint8)  # global AE shift
    b[200:230, 300:320] = 255                                  # real motion patch
    m = diff_mask(a, b, SCFG.diff_thresh)
    assert m.sum() < 8000, "global exposure shift must not light up the frame"
    blobs = find_blobs(m, SCFG.min_blob)
    assert blobs, "the genuinely moved patch must still be found"
    assert np.linalg.norm(blobs[0].center - [310, 215]) < 15, \
        "blob must be at the moved patch, not the frame center"


def test_blink_sync_detection_rejects_noise_and_passersby():
    """Reproduces the real-camera failure: Test Blink reported the gripper at
    a different random place every time (13-33 scattered diff blobs from
    compression shimmer + a person moving in frame). blink_measure's
    synchronous detection must find the one region that oscillates WITH the
    gripper command and ignore everything else."""
    from perception import blink_measure
    from config import ServoConfig

    rng = np.random.default_rng(4)
    H, W = 480, 640
    base = rng.integers(60, 190, (H, W, 3)).astype(np.uint8)
    scfg = ServoConfig()

    class NoisyRig:
        """Gripper = small patch at (320,240) that shifts with g; plus
        per-frame speckle noise blobs (compression shimmer) and a 'person'
        drifting right regardless of commands."""
        def __init__(self):
            self.g = 1.0
            self.t = 0
        def set_gripper(self, g):
            self.g = g
        def read(self):
            self.t += 1
            f = base.copy()
            # person: 60x40 block moving continuously with time
            x = 40 + self.t * 12
            f[80:140, x:x + 40] = 30
            # compression shimmer: 25 random small speckles, new every frame
            r = np.random.default_rng(self.t)
            for _ in range(25):
                y, xx = int(r.integers(0, H - 8)), int(r.integers(0, W - 8))
                f[y:y + 7, xx:xx + 7] = r.integers(0, 255)
            # gripper fingers: 24x10 bar whose x-position tracks the command
            gx = int(320 + (1.0 - self.g) * 30)
            f[230:254, gx:gx + 10] = 255
            return f

    px, info = blink_measure(NoisyRig(), scfg)
    assert px is not None, f"sync detection found nothing: {info}"
    err = np.linalg.norm(px - [325, 242])
    assert err < 30, f"gripper localized {err:.0f}px off: {px} vs ~(325,242); {info}"


def test_filter_by_background_drops_arm_detection():
    """A learned detector will happily box the robot arm itself (seen live
    with nanodet in the pybullet GUI: its biggest detection was the arm, and
    the servo chased its own arm). Detections must be corroborated by the
    bg-diff, where an arm parked at its background-photo pose can't appear."""
    from perception import Blob, filter_by_background
    w = SimWorld(seed=18)
    bg = w.capture_background()  # arm at home is part of the background
    (o,) = w.spawn_random(1)
    frame = w.read()
    obj_px = w.cam.project(o.pos)
    arm_px = w.grip_px()  # arm is at its background pose -> no bg-diff evidence
    fake_detections = [
        Blob(np.asarray(arm_px), 50000, (0, 0, 200, 200), (0, 0, 0)),   # "the arm"
        Blob(np.asarray(obj_px), 900, (0, 0, 30, 30), (0, 0, 0)),       # the object
    ]
    kept = filter_by_background(fake_detections, frame, bg,
                                SCFG.bg_thresh, SCFG.obj_min_area)
    assert len(kept) == 1, f"expected only the real object to survive, got {len(kept)}"
    assert np.linalg.norm(kept[0].center - obj_px) < 1e-9


if __name__ == "__main__":
    for fn in [test_detect_objects_any_color, test_pad_and_hue_selection,
               test_blink_locate, test_locate_by_diff_tracks_carried_object,
               test_locate_by_diff_rejects_distant_distractor,
               test_diff_mask_ignores_global_exposure_shift,
               test_blink_sync_detection_rejects_noise_and_passersby,
               test_filter_by_background_drops_arm_detection]:
        fn()
        print(f"ok {fn.__name__}")
    print("ALL OK")

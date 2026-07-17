"""FusedTracker (encoder+vision fusion) vs the toy sim's ground truth."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aruco_tracker import workspace_to_pixel
from config import ServoConfig, SimConfig
from fused_tracking import (AnalyticalFusedTracker, FusedTracker,
                            fit_projection, project, spread_ok)
from perception import blink_locate
from sim import NominalModel, SimWorld

SCFG = ServoConfig()


def test_dlt_exact_on_true_camera():
    """Noise-free resectioning against the sim's true pinhole camera must be
    sub-pixel."""
    w = SimWorld(seed=90)
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(0.05, 0.25, 24),
                         rng.uniform(-0.15, 0.15, 24),
                         rng.uniform(0.01, 0.15, 24)])
    px = np.stack([w.cam.project(p) for p in X])
    P, rms = fit_projection(X, px)
    assert rms < 0.1, f"noise-free DLT rms {rms:.3f}px"
    p_new = np.array([0.15, 0.05, 0.08])
    err = np.linalg.norm(project(P, p_new) - w.cam.project(p_new))
    assert err < 0.2, f"held-out projection off by {err:.2f}px"


def test_spread_guard_rejects_coplanar():
    rng = np.random.default_rng(1)
    flat = np.column_stack([rng.uniform(0, 0.3, 20), rng.uniform(-0.2, 0.2, 20),
                            np.full(20, 0.05)])
    assert not spread_ok(flat), "coplanar cloud must be rejected (DLT degenerate)"
    thick = flat + np.column_stack([np.zeros(20), np.zeros(20),
                                    rng.uniform(-0.05, 0.05, 20)])
    assert spread_ok(thick)


def test_fused_tracker_from_real_blinks():
    """End-to-end fit exactly as run_episode will do it: babble-style joint
    poses, REAL blink measurements (noisy, jaw-biased), the deliberately
    wrong NominalModel FK (+/-5% link lengths) -- the fused prediction must
    still track the true grip pixel closely, and its Jacobian must agree
    with ground truth in direction."""
    w = SimWorld(seed=91)
    rng = np.random.default_rng(91)
    model = NominalModel(w.cfg, rng)
    tracker = FusedTracker(model.ee)

    q0 = w.get_q()
    qs, pxs = [], []
    for _ in range(20):
        q = q0 + rng.uniform(-1.5, 1.5, 3) * np.array([0.045, 0.05, 0.05])
        w.set_q(q)
        px = blink_locate(w, SCFG)
        if px is None:
            continue
        qs.append(w.get_q())
        pxs.append(px)
    tracker.add_pairs(qs, pxs)
    ok = tracker.fit(max_rms=10.0)
    print(f"  fit rms {tracker.rms:.1f}px over {len(qs)} blink pairs, ok={ok}")
    assert ok, f"fit should succeed (rms {tracker.rms:.1f}px)"

    # prediction accuracy at fresh poses, zero extra robot motion
    errs = []
    for _ in range(8):
        q = q0 + rng.uniform(-1.2, 1.2, 3) * np.array([0.045, 0.05, 0.05])
        w.set_q(q)
        errs.append(np.linalg.norm(tracker.ee_px(w.get_q()) - w.grip_px()))
    print(f"  held-out tracking error: mean {np.mean(errs):.1f}px max {np.max(errs):.1f}px")
    assert np.mean(errs) < 12, f"fused tracking too far off: {errs}"

    # Jacobian direction vs ground truth: finite diff of the true ANALYTIC
    # pipeline (grip_center + true camera), NOT via set_q -- actuation noise
    # (0.0015 rad) would swamp a 1e-5 probe commanded through the sim
    q = w.get_q()
    J = tracker.jac(q)
    for i in range(3):
        d = np.zeros(3); d[i] = 1e-5
        gt = (w.cam.project(w.grip_center(q + d)) -
              w.cam.project(w.grip_center(q - d))) / 2e-5
        cos = J[:, i] @ gt / (np.linalg.norm(J[:, i]) * np.linalg.norm(gt) + 1e-9)
        # the +/-5% wrong link lengths tilt per-joint image velocity a little;
        # closed-loop feedback tolerates far worse (servo contraction holds to
        # ~60deg). The tight gate is the sub-2px POSITION accuracy above.
        assert cos > 0.9, f"joint {i} Jacobian direction off (cos {cos:.2f})"


def _synthetic_analytical_tracker(t_init):
    """Tilted overhead camera looking at a tabletop workspace; fk=identity so
    'joint state' IS the 3D robot-frame position -- the projection geometry
    (what the EKF linearizes) is exercised for real, the arm model isn't the
    thing under test."""
    cam_mtx = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    dist = np.zeros(5)
    # camera 0.6m above the table, pitched ~25deg off straight-down
    R_ws2cam = (cv2.Rodrigues(np.array([0.44, 0.0, 0.0]))[0]
                @ cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0])
    rvec = cv2.Rodrigues(R_ws2cam)[0].ravel()
    tvec = -R_ws2cam @ np.array([0.15, 0.35, 0.6])
    return AnalyticalFusedTracker(lambda q: np.asarray(q, float),
                                  cam_mtx, dist, rvec, tvec, t_init)


def test_ekf_base_offset_converges():
    """Start the base offset a few cm off (a bad one-shot wrist-marker
    calibration), feed noisy (q, pixel) pairs -- the EKF must recover the
    true offset and cut the pixel prediction error to ~noise level."""
    rng = np.random.default_rng(7)
    t_true = np.array([-0.073, 0.249, 0.068])
    t_bad = t_true + np.array([0.03, -0.025, 0.02])
    trk = _synthetic_analytical_tracker(t_bad)
    ref = _synthetic_analytical_tracker(t_true)

    qs = np.column_stack([rng.uniform(0.05, 0.30, 60),
                          rng.uniform(-0.15, 0.15, 60),
                          rng.uniform(0.02, 0.15, 60)])
    err0 = np.mean([np.linalg.norm(trk.ee_px(q) - ref.ee_px(q)) for q in qs])
    for q in qs[:40]:
        trk.update(q, ref.ee_px(q) + rng.normal(0, 2.0, 2))
    assert trk.n_updates == 40
    t_err = np.linalg.norm(trk.t_robot_ws - t_true)
    err1 = np.mean([np.linalg.norm(trk.ee_px(q) - ref.ee_px(q)) for q in qs[40:]])
    print(f"  offset err {np.linalg.norm(t_bad - t_true)*1000:.1f} -> {t_err*1000:.1f}mm, "
          f"pixel err {err0:.1f} -> {err1:.1f}px")
    assert t_err < 0.008, f"offset error {t_err*1000:.1f}mm after 40 updates"
    assert err1 < 4.0, f"held-out pixel error {err1:.1f}px"


def test_ekf_gates_outliers():
    """A wild misdetection (beyond gate_px) must be rejected untouched."""
    trk = _synthetic_analytical_tracker(np.array([-0.073, 0.249, 0.068]))
    q = np.array([0.15, 0.0, 0.05])
    t_before = trk.t_robot_ws.copy()
    assert trk.update(q, trk.ee_px(q) + np.array([200.0, 150.0])) is None
    assert np.array_equal(trk.t_robot_ws, t_before)
    assert trk.n_updates == 0
    assert trk.update(q, trk.ee_px(q) + np.array([3.0, -2.0])) is not None


if __name__ == "__main__":
    for fn in [test_dlt_exact_on_true_camera, test_spread_guard_rejects_coplanar,
               test_fused_tracker_from_real_blinks,
               test_ekf_base_offset_converges, test_ekf_gates_outliers]:
        fn()
        print(f"ok {fn.__name__}")
    print("ALL OK")

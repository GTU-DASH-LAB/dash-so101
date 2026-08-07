"""Tests for the ArUco tracker module, verify detection APIs and projection math."""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aruco_tracker import (
    ArucoDetector,
    CalibrationData,
    estimate_default_intrinsics,
    pixel_to_workspace,
    workspace_to_pixel
)

def test_aruco_detector_creation():
    detector = ArucoDetector()
    assert detector is not None

def test_synthetic_aruco_detection():
    detector = ArucoDetector()
    # Create a blank white image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a synthetic ArUco marker (ID 48) on the image
    # Note: DICT_4X4_50
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_img = cv2.aruco.generateImageMarker(dictionary, 48, 200)
    except AttributeError:
        # Older OpenCV versions
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        marker_img = cv2.aruco.drawMarker(dictionary, 48, 200)
        
    # Place marker in the center of the image
    marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    img[140:340, 220:420] = marker_rgb
    
    # Detect
    markers = detector.detect_all(img)
    assert 48 in markers
    m = markers[48]
    assert m.marker_id == 48
    # Center should be close to (320, 240)
    assert np.allclose(m.center, [320, 240], atol=10.0)

def test_projection_math():
    # Intrinsic parameters
    img_w, img_h = 640, 480
    camera_matrix, dist_coeffs = estimate_default_intrinsics(img_w, img_h)
    
    # Camera poses (extrinsics)
    # Let's say camera is 1 meter above table center looking straight down
    # Workspace frame: table surface is Z=0
    # Camera looking down -> Rotation R is pointing along +Z in camera frame, which is -Z in workspace.
    # We can represent this with a rotation vector
    # Let's define camera-to-workspace transform where camera is at (0.3, 0.2, 1.0) looking down
    rvec = np.array([np.pi, 0, 0], dtype=np.float64) # 180 deg rot around X
    tvec = np.array([0.3, 0.2, 1.0], dtype=np.float64) # Camera offset
    
    # Test point in workspace (meters)
    pt_ws = np.array([0.1, 0.15, 0.0], dtype=np.float64)
    
    # Project to pixel
    px = workspace_to_pixel(pt_ws, camera_matrix, dist_coeffs, rvec, tvec)
    
    # Back-project from pixel
    pt_reconstructed = pixel_to_workspace(px, camera_matrix, dist_coeffs, rvec, tvec, z_plane=0.0)
    
    assert np.allclose(pt_ws, pt_reconstructed, atol=1e-5)

def test_calibration_data_save_load(tmp_path):
    cal = CalibrationData()
    cal.camera_matrix = np.eye(3)
    cal.dist_coeffs = np.zeros(5)
    cal.reprojection_error = 0.05
    cal.ws_rvec = np.array([0.1, 0.2, 0.3])
    cal.ws_tvec = np.array([1.0, 2.0, 3.0])
    cal.workspace_marker_positions = {
        0: np.array([0, 0, 0]),
        1: np.array([0.5, 0.5, 0.0])
    }
    
    p = tmp_path / "cal.json"
    cal.save(str(p))
    
    loaded = CalibrationData.load(str(p))
    assert np.allclose(loaded.camera_matrix, cal.camera_matrix)
    assert np.allclose(loaded.dist_coeffs, cal.dist_coeffs)
    assert loaded.reprojection_error == cal.reprojection_error
    assert np.allclose(loaded.ws_rvec, cal.ws_rvec)
    assert np.allclose(loaded.ws_tvec, cal.ws_tvec)
    assert 0 in loaded.workspace_marker_positions
    assert np.allclose(loaded.workspace_marker_positions[0], cal.workspace_marker_positions[0])


def test_charuco_calibration_recovers_focal_length():
    """End-to-end ChArUco intrinsic calibration on synthetic views: render the
    board plane through a known pinhole camera at several poses, calibrate,
    and the recovered focal length must match. Guards the OpenCV>=4.8 API path
    (legacy interpolateCornersCharuco/calibrateCameraCharuco were removed --
    observed live as 'Calibration failed' on every frame with cv2 4.13)."""
    from aruco_tracker import calibrate_camera_charuco, create_charuco_board

    board, _ = create_charuco_board()  # project default: 11x8, 50/30mm
    board_img = board.generateImage((1650, 1200), marginSize=20, borderBits=1)
    bw, bh = 0.55, 0.40  # 11x8 squares of 50mm, meters
    K_true = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])

    def view(rx, ry, tx, ty, tz):
        # small tilts around identity: board x/y axes align with image u/v,
        # so the warp preserves marker chirality (a mirrored ArUco is
        # undecodable)
        rvec = np.array([rx, ry, 0.0])
        tvec = np.array([tx, ty, tz])
        obj = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], float)
        img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K_true, None)
        h_px, w_px = board_img.shape[:2]
        src = np.array([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]], np.float32)
        H = cv2.getPerspectiveTransform(src, img_pts.reshape(4, 2).astype(np.float32))
        return cv2.warpPerspective(board_img, H, (640, 480), borderValue=255)

    frames = [view(rx, ry, tx, ty, tz) for rx, ry, tx, ty, tz in [
        (0.00,  0.00, -0.28, -0.21, 0.72),
        (0.25,  0.00, -0.30, -0.18, 0.78),
        (-0.25, 0.00, -0.26, -0.23, 0.76),
        (0.00,  0.25, -0.31, -0.20, 0.80),
        (0.00, -0.25, -0.25, -0.21, 0.74),
        (0.20, -0.20, -0.29, -0.19, 0.82),
    ]]
    cam_mtx, dist, rms = calibrate_camera_charuco(frames)
    assert cam_mtx is not None, "calibration returned None on clean synthetic views"
    fx_err = abs(cam_mtx[0, 0] - 600.0) / 600.0
    assert fx_err < 0.05, f"fx {cam_mtx[0,0]:.1f} vs true 600 ({fx_err*100:.1f}% off)"
    assert rms < 2.0, f"reprojection rms {rms:.2f}px"
    # true camera has zero distortion: wild coefficients mean the model is
    # overfitting views again (live failure mode: k2=-29 from 5 frames)
    assert np.max(np.abs(dist)) < 0.5, f"distortion overfit: {dist}"


def test_charuco_calibration_fails_cleanly_on_wrong_board():
    """Frames showing NO matching board (blank) must return (None, None, inf),
    not raise -- the web endpoint relies on the None signal."""
    from aruco_tracker import calibrate_camera_charuco
    frames = [np.full((480, 640), 255, np.uint8) for _ in range(5)]
    cam_mtx, dist, rms = calibrate_camera_charuco(frames)
    assert cam_mtx is None and rms == float('inf')


def _marker_frame(mid, x, y, size=48, hole=False):
    """White 640x480 canvas with a DICT_4X4_50 marker at (x, y). hole=True
    wipes the payload bits with a gray blotch (like specular glare): decoding
    fails deterministically, but the outer corners stay LK-trackable."""
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tile = cv2.aruco.generateImageMarker(dic, mid, size)
    img = np.full((480, 640), 255, np.uint8)
    img[y:y + size, x:x + size] = tile
    if hole:
        q = size // 3
        img[y + q:y + size - q, x + q:x + size - q] = 128
    return img


def test_lk_tracking_bridges_decode_dropout():
    """A marker that decoded last frame but is too blurred to decode now must
    still be reported (LK-tracked corners), near its true shifted position."""
    det = ArucoDetector()
    f1 = _marker_frame(48, 300, 200)
    markers = det.detect_all(f1)
    assert 48 in markers, "sharp marker must decode"

    # shifted + payload wiped: a FRESH detector must fail on it
    # (precondition proving the cached detector's answer comes from tracking)
    f2 = _marker_frame(48, 312, 208, hole=True)
    assert 48 not in ArucoDetector().detect_all(f2), \
        "precondition: wiped payload should defeat decoding"

    m = det.detect_all(f2)
    assert 48 in m, "LK fallback must bridge the dropout"
    err = np.linalg.norm(m[48].center - (np.array([300 + 24, 200 + 24]) + [12, 8]))
    assert err < 6, f"tracked center off by {err:.1f}px"


def test_lk_tracking_expires():
    """Without re-detection, the track must die after track_max_age frames."""
    det = ArucoDetector(track_max_age=3)
    det.detect_all(_marker_frame(48, 300, 200))
    wiped = _marker_frame(48, 300, 200, hole=True)
    seen = [48 in det.detect_all(wiped) for _ in range(6)]
    assert seen[0], "first dropout frame should be bridged"
    assert not seen[-1], "track must expire, not persist forever"


def test_locate_by_aruco_prediction_gate():
    """Detections implausibly far from the motion-model prediction are
    rejected (the predicted_px/max_dist params were silently unused)."""
    from perception import locate_by_aruco
    frame = cv2.cvtColor(_marker_frame(48, 500, 300), cv2.COLOR_GRAY2BGR)
    near = locate_by_aruco(frame, 48, predicted_px=np.array([520.0, 320.0]))
    assert near is not None
    far = locate_by_aruco(frame, 48, predicted_px=np.array([100.0, 100.0]),
                          max_dist=120.0)
    assert far is None

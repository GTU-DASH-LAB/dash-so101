"""ArUco marker-based perception: detection, pose estimation, workspace
calibration, gripper/wrist tracking, and camera intrinsic calibration.

Marker layout (user-defined, DICT_4X4_50):
  IDs 0-3  — four corners of the workspace (table edges)
  ID 48    — gripper tip
  ID 49    — wrist joint (link before gripper)

Camera calibration flow:
  1. Intrinsics: collect ChArUco board images → calibrate_camera_charuco()
     (or fall back to a reasonable pinhole estimate from resolution).
  2. Extrinsics: detect workspace markers 0-3 whose 3D positions are known
     → solvePnP gives camera-to-workspace transform.

Once calibrated, any detected pixel can be back-projected onto a known
Z-plane in the workspace (e.g. table surface), and the gripper's 3D
workspace position can be estimated from marker 48.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MarkerInfo:
    """Detection result for a single ArUco marker."""
    marker_id: int
    corners: np.ndarray        # (4, 2) pixel corners
    center: np.ndarray         # (2,) pixel center
    rvec: Optional[np.ndarray] = None  # rotation vector (if pose estimated)
    tvec: Optional[np.ndarray] = None  # translation vector (if pose estimated)


@dataclass
class CalibrationData:
    """Persistent camera + workspace calibration."""
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    reprojection_error: float = float('inf')
    # workspace extrinsics: camera → workspace transform
    ws_rvec: Optional[np.ndarray] = None
    ws_tvec: Optional[np.ndarray] = None
    # known 3D positions of workspace markers {id: (x,y,z)}
    workspace_marker_positions: Dict[int, np.ndarray] = field(default_factory=dict)
    calibrated_at: Optional[str] = None
    last_port: Optional[str] = None
    grasp_z: Optional[float] = None

    def has_intrinsics(self) -> bool:
        return self.camera_matrix is not None and self.dist_coeffs is not None

    def has_workspace(self) -> bool:
        return self.ws_rvec is not None and self.ws_tvec is not None

    def save(self, path: str):
        """Save calibration to JSON."""
        d = {}
        if self.camera_matrix is not None:
            d["camera_matrix"] = self.camera_matrix.tolist()
        if self.dist_coeffs is not None:
            d["dist_coeffs"] = self.dist_coeffs.tolist()
        d["reprojection_error"] = self.reprojection_error
        if self.ws_rvec is not None:
            d["ws_rvec"] = self.ws_rvec.tolist()
        if self.ws_tvec is not None:
            d["ws_tvec"] = self.ws_tvec.tolist()
        if self.workspace_marker_positions:
            d["workspace_marker_positions"] = {
                str(k): v.tolist() for k, v in self.workspace_marker_positions.items()
            }
        d["calibrated_at"] = self.calibrated_at or time.strftime("%Y-%m-%dT%H:%M:%S")
        if self.last_port is not None:
            d["last_port"] = self.last_port
        if self.grasp_z is not None:
            d["grasp_z"] = self.grasp_z
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationData":
        """Load calibration from JSON."""
        with open(path) as f:
            d = json.load(f)
        cal = cls()
        if "camera_matrix" in d:
            cal.camera_matrix = np.array(d["camera_matrix"], dtype=np.float64)
        if "dist_coeffs" in d:
            cal.dist_coeffs = np.array(d["dist_coeffs"], dtype=np.float64)
        cal.reprojection_error = d.get("reprojection_error", float('inf'))
        if "ws_rvec" in d:
            cal.ws_rvec = np.array(d["ws_rvec"], dtype=np.float64)
        if "ws_tvec" in d:
            cal.ws_tvec = np.array(d["ws_tvec"], dtype=np.float64)
        if "workspace_marker_positions" in d:
            cal.workspace_marker_positions = {
                int(k): np.array(v, dtype=np.float64)
                for k, v in d["workspace_marker_positions"].items()
            }
        cal.calibrated_at = d.get("calibrated_at")
        cal.last_port = d.get("last_port")
        cal.grasp_z = d.get("grasp_z")
        return cal


# ---------------------------------------------------------------------------
# ArUco Detector (singleton-friendly, handles both old and new OpenCV APIs)
# ---------------------------------------------------------------------------

class ArucoDetector:
    """Wraps OpenCV ArUco detection with API compatibility for 4.7+ and older."""

    def __init__(self, dictionary_id=None):
        if dictionary_id is None:
            dictionary_id = cv2.aruco.DICT_4X4_50
        self._dict_id = dictionary_id
        # Try new API first (OpenCV 4.7+)
        try:
            self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            params = cv2.aruco.DetectorParameters()
            # Tune for small markers at moderate distance
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 23
            params.adaptiveThreshWinSizeStep = 4
            params.minMarkerPerimeterRate = 0.01
            params.maxMarkerPerimeterRate = 4.0
            params.polygonalApproxAccuracyRate = 0.03
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params.errorCorrectionRate = 0.8
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, params)
            self._new_api = True
        except AttributeError:
            self._dictionary = cv2.aruco.Dictionary_get(dictionary_id)
            self._params = cv2.aruco.DetectorParameters_create()
            self._params.adaptiveThreshWinSizeMin = 3
            self._params.adaptiveThreshWinSizeMax = 23
            self._params.adaptiveThreshWinSizeStep = 4
            self._params.minMarkerPerimeterRate = 0.01
            self._params.maxMarkerPerimeterRate = 4.0
            self._params.polygonalApproxAccuracyRate = 0.03
            try:
                self._params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            except AttributeError:
                self._params.cornerRefinementMethod = 1
            self._params.errorCorrectionRate = 0.8
            self._detector = None
            self._new_api = False

    def detect(self, frame: np.ndarray) -> Tuple[list, np.ndarray, list]:
        """Detect markers. Returns (corners, ids, rejected)."""
        h, w = frame.shape[:2]
        upscaled = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        if self._new_api:
            corners, ids, rejected = self._detector.detectMarkers(upscaled)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                upscaled, self._dictionary, parameters=self._params)
        if corners is not None and len(corners) > 0:
            corners = [c / 2.0 for c in corners]
        if rejected is not None and len(rejected) > 0:
            rejected = [r / 2.0 for r in rejected]
        return corners, ids, rejected

    def detect_all(self, frame: np.ndarray,
                   camera_matrix: np.ndarray = None,
                   dist_coeffs: np.ndarray = None,
                   marker_size: float = 0.04,
                   marker_sizes: Dict[int, float] = None) -> Dict[int, MarkerInfo]:
        """Detect all markers, optionally with pose estimation.

        Args:
            frame: BGR image
            camera_matrix: 3x3 intrinsic matrix (None = skip pose)
            dist_coeffs: distortion coefficients (None = skip pose)
            marker_size: default physical side length in meters
            marker_sizes: optional per-ID size overrides, e.g. {0: 0.033, 48: 0.018}

        Returns:
            dict mapping marker_id → MarkerInfo
        """
        corners, ids, _ = self.detect(frame)
        result = {}
        if ids is None:
            return result

        for idx, mid in enumerate(ids.flatten()):
            c = corners[idx][0]  # (4, 2)
            center = np.mean(c, axis=0)
            info = MarkerInfo(marker_id=int(mid), corners=c, center=center)

            # Pose estimation if camera is calibrated
            if camera_matrix is not None and dist_coeffs is not None:
                sz = marker_size
                if marker_sizes and int(mid) in marker_sizes:
                    sz = marker_sizes[int(mid)]
                obj_pts = np.array([
                    [-sz / 2,  sz / 2, 0],
                    [ sz / 2,  sz / 2, 0],
                    [ sz / 2, -sz / 2, 0],
                    [-sz / 2, -sz / 2, 0],
                ], dtype=np.float64)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, c.astype(np.float64),
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    info.rvec = rvec.flatten()
                    info.tvec = tvec.flatten()

            result[int(mid)] = info
        return result

    def draw_markers(self, frame: np.ndarray,
                     markers: Dict[int, MarkerInfo],
                     draw_axes: bool = False,
                     camera_matrix: np.ndarray = None,
                     dist_coeffs: np.ndarray = None,
                     axis_length: float = 0.03) -> np.ndarray:
        """Draw detected markers onto frame (in-place)."""
        if not markers:
            return frame
        # Reconstruct corners/ids arrays for drawDetectedMarkers
        corners_list = [m.corners.reshape(1, 4, 2) for m in markers.values()]
        ids_arr = np.array([[m.marker_id] for m in markers.values()])
        cv2.aruco.drawDetectedMarkers(frame, corners_list, ids_arr)

        if draw_axes and camera_matrix is not None and dist_coeffs is not None:
            for m in markers.values():
                if m.rvec is not None and m.tvec is not None:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                     m.rvec, m.tvec, axis_length)
        return frame


# ---------------------------------------------------------------------------
# Workspace calibration
# ---------------------------------------------------------------------------

def estimate_default_intrinsics(img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reasonable pinhole estimate when no ChArUco calibration is available.
    Assumes ~60° horizontal FOV (typical webcam)."""
    fx = fy = img_w / (2 * np.tan(np.radians(30)))  # ~60° HFOV
    cx, cy = img_w / 2.0, img_h / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros(5, dtype=np.float64)
    return camera_matrix, dist_coeffs


def calibrate_workspace(detector: ArucoDetector,
                        frame: np.ndarray,
                        marker_positions: Dict[int, np.ndarray],
                        camera_matrix: np.ndarray,
                        dist_coeffs: np.ndarray,
                        marker_size: float = 0.04) -> Tuple[Optional[np.ndarray],
                                                             Optional[np.ndarray],
                                                             float]:
    """Calibrate camera extrinsics from known workspace marker positions.

    Args:
        detector: ArucoDetector instance
        frame: BGR image with workspace markers visible
        marker_positions: {marker_id: np.array([x, y, z])} in workspace frame (meters)
        camera_matrix: 3x3 intrinsic matrix
        dist_coeffs: distortion coefficients
        marker_size: physical side length (meters)

    Returns:
        (rvec, tvec, reprojection_error) or (None, None, inf) if failed.
        rvec/tvec define the camera-to-workspace transform.
    """
    markers = detector.detect_all(frame, camera_matrix, dist_coeffs, marker_size)

    # Collect all visible workspace markers
    obj_points = []  # 3D in workspace
    img_points = []  # 2D in image
    half = marker_size / 2.0

    for mid, pos_3d in marker_positions.items():
        if mid not in markers:
            continue
        m = markers[mid]
        # Each marker has 4 corners; generate their 3D positions
        # relative to workspace origin, assuming markers lie flat on the table (Z=pos_3d[2])
        # Marker local corners (centered at marker center, lying in XY plane):
        local_corners = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float64)
        for lc, ic in zip(local_corners, m.corners):
            obj_points.append(pos_3d + lc)
            img_points.append(ic)

    if len(obj_points) < 4:
        return None, None, float('inf')

    obj_points = np.array(obj_points, dtype=np.float64)
    img_points = np.array(img_points, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points,
                                   camera_matrix, dist_coeffs,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, float('inf')

    # Compute reprojection error
    projected, _ = cv2.projectPoints(obj_points, rvec, tvec,
                                     camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - img_points, axis=1)
    rms = float(np.sqrt(np.mean(errors ** 2)))

    return rvec.flatten(), tvec.flatten(), rms


def pixel_to_workspace(px: np.ndarray,
                       camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray,
                       rvec: np.ndarray,
                       tvec: np.ndarray,
                       z_plane: float = 0.0) -> Optional[np.ndarray]:
    """Back-project a pixel onto a Z-plane in workspace coordinates.

    Uses the calibrated camera extrinsics to cast a ray from the camera
    through the pixel and intersect it with the plane Z = z_plane in
    workspace coordinates.

    Args:
        px: (2,) pixel coordinates (u, v)
        camera_matrix: 3x3 intrinsic matrix
        dist_coeffs: distortion coefficients
        rvec: rotation vector (camera-to-workspace)
        tvec: translation vector (camera-to-workspace)
        z_plane: Z-coordinate of the plane in workspace frame

    Returns:
        (3,) point in workspace coordinates, or None if ray is parallel to plane.
    """
    # Undistort the pixel
    pts = np.array([[px]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, camera_matrix, dist_coeffs,
                                P=camera_matrix)
    u, v = undist[0, 0]

    # Camera-frame ray direction
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])

    # Transform to workspace frame
    R, _ = cv2.Rodrigues(rvec)
    # Camera position in workspace: -R^T @ tvec
    cam_pos_ws = -R.T @ tvec.reshape(3)
    ray_ws = R.T @ ray_cam

    # Intersect with Z = z_plane
    if abs(ray_ws[2]) < 1e-10:
        return None  # ray parallel to plane
    t = (z_plane - cam_pos_ws[2]) / ray_ws[2]
    if t < 0:
        return None  # behind camera
    return cam_pos_ws + t * ray_ws


def workspace_to_pixel(point_3d: np.ndarray,
                       camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray,
                       rvec: np.ndarray,
                       tvec: np.ndarray) -> np.ndarray:
    """Project a 3D workspace point to pixel coordinates."""
    projected, _ = cv2.projectPoints(
        point_3d.reshape(1, 1, 3).astype(np.float64),
        rvec.reshape(3, 1), tvec.reshape(3, 1),
        camera_matrix, dist_coeffs)
    return projected[0, 0]


# ---------------------------------------------------------------------------
# Camera intrinsic calibration (ChArUco)
# ---------------------------------------------------------------------------

def create_charuco_board(squares_x: int = 7, squares_y: int = 5,
                         square_length: float = 0.035,
                         marker_length: float = 0.022):
    """Create a ChArUco board for calibration.
    Returns (board, dictionary) for detection."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, dictionary)
    except TypeError:
        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_length, marker_length, dictionary)
    return board, dictionary


def calibrate_camera_charuco(frames: List[np.ndarray],
                             squares_x: int = 7, squares_y: int = 5,
                             square_length: float = 0.035,
                             marker_length: float = 0.022) -> Tuple[Optional[np.ndarray],
                                                                     Optional[np.ndarray],
                                                                     float]:
    """Calibrate camera intrinsics from ChArUco board images.

    Args:
        frames: list of BGR images containing the ChArUco board
        squares_x, squares_y: board dimensions
        square_length: chessboard square side (meters)
        marker_length: ArUco marker side (meters)

    Returns:
        (camera_matrix, dist_coeffs, reprojection_error) or (None, None, inf)
    """
    board, dictionary = create_charuco_board(squares_x, squares_y,
                                             square_length, marker_length)
    img_size = None

    # OpenCV >= 4.8 removed interpolateCornersCharuco/calibrateCameraCharuco
    # (gone entirely by 4.9+); CharucoDetector + matchImagePoints +
    # cv2.calibrateCamera is the supported path. Keep the legacy branch for
    # older installs (e.g. other machines running this project).
    use_new = hasattr(cv2.aruco, "CharucoDetector")
    if use_new:
        cdet = cv2.aruco.CharucoDetector(board)
        all_obj, all_img = [], []
    else:
        detector = ArucoDetector(cv2.aruco.DICT_4X4_50)
        all_charuco_corners, all_charuco_ids = [], []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        if img_size is None:
            img_size = gray.shape[::-1]

        if use_new:
            charuco_corners, charuco_ids, _, _ = cdet.detectBoard(gray)
            if charuco_corners is None or len(charuco_corners) < 4:
                continue
            obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_pts is None or len(obj_pts) < 4:
                continue
            all_obj.append(obj_pts)
            all_img.append(img_pts)
        else:
            corners, ids, _ = detector.detect(frame)
            if ids is None or len(ids) < 4:
                continue
            try:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board)
            except Exception:
                continue
            if ret >= 4:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

    try:
        if use_new:
            if len(all_obj) < 3:
                return None, None, float('inf')
            ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
                all_obj, all_img, img_size, None, None)
        else:
            if len(all_charuco_corners) < 3:
                return None, None, float('inf')
            ret, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
                all_charuco_corners, all_charuco_ids, board, img_size, None, None)
    except Exception:
        return None, None, float('inf')

    return camera_matrix, dist_coeffs.flatten(), float(ret)


# ---------------------------------------------------------------------------
# Gripper tracking helpers
# ---------------------------------------------------------------------------

def track_gripper(detector: ArucoDetector,
                  frame: np.ndarray,
                  gripper_id: int = 48,
                  camera_matrix: np.ndarray = None,
                  dist_coeffs: np.ndarray = None,
                  marker_size: float = 0.04) -> Optional[MarkerInfo]:
    """Track the gripper marker. Returns MarkerInfo or None."""
    markers = detector.detect_all(frame, camera_matrix, dist_coeffs, marker_size)
    return markers.get(gripper_id)


def track_wrist(detector: ArucoDetector,
                frame: np.ndarray,
                wrist_id: int = 49,
                camera_matrix: np.ndarray = None,
                dist_coeffs: np.ndarray = None,
                marker_size: float = 0.04) -> Optional[MarkerInfo]:
    """Track the wrist joint marker. Returns MarkerInfo or None."""
    markers = detector.detect_all(frame, camera_matrix, dist_coeffs, marker_size)
    return markers.get(wrist_id)


def compute_gripper_vector(detector: ArucoDetector,
                           frame: np.ndarray,
                           gripper_id: int = 48,
                           wrist_id: int = 49) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute the gripper approach vector from wrist→gripper markers.

    Returns:
        (direction_2d, gripper_px, wrist_px) or None if either marker not visible.
        direction_2d is a unit vector in pixel space from wrist to gripper.
    """
    markers = detector.detect_all(frame)
    if gripper_id not in markers or wrist_id not in markers:
        return None
    g_px = markers[gripper_id].center
    w_px = markers[wrist_id].center
    d = g_px - w_px
    norm = np.linalg.norm(d)
    if norm < 1.0:
        return None
    return d / norm, g_px, w_px

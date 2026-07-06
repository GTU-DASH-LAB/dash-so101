"""URDF-based kinematics for the SO-101 + camera-table calibration.

Bridges three coordinate systems:
  normalized joints (dataset / bus units, -100..100) <-> radians (URDF)
  camera pixels (u, v) <-> robot-frame table coordinates (x, y), via a homography
  fitted from the mined demos: FK(grasp pose) gives where the gripper (= the pen)
  was in 3D at each demonstrated grasp; pairing with the pen's pixel gives 50
  pixel<->table correspondences — no manual camera calibration.

Fit + validate (writes the "table" section into pid_calib.json):

    PYTHONPATH=. .venv/bin/python so_brain/kinematics.py
"""

import json
from pathlib import Path

import numpy as np

URDF = Path(__file__).parent / "urdf" / "so101_new_calib.urdf"
FOLLOWER_CALIB = Path(
    "~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json"
).expanduser()
JOINTS5 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class SO101Kinematics:
    """The dataset and the bus both use use_degrees=True (the so101_follower
    default), so joint values ARE URDF-convention degrees — no unit conversion."""

    def __init__(self, follower_calib: Path = FOLLOWER_CALIB):
        from lerobot.model.kinematics import RobotKinematics

        self.kin = RobotKinematics(str(URDF), target_frame_name="gripper_frame_link")

    def fk(self, pose5_deg: np.ndarray) -> np.ndarray:
        """5-joint pose (degrees) -> 4x4 gripper frame in the robot base frame."""
        return self.kin.forward_kinematics(np.append(np.asarray(pose5_deg, dtype=float), 0.0))

    def refine_position(self, seed5_norm: np.ndarray, target_xyz: np.ndarray,
                        iters: int = 8, tol: float = 1e-3, damping: float = 0.02) -> np.ndarray:
        """Locally correct a demo-family pose so the gripper lands on target_xyz.

        Damped-least-squares on a finite-difference position Jacobian. The seed
        (from the pixel->pose regression) fixes the arm configuration; this only
        polishes position, so it cannot jump to a flipped IK branch the demos
        never used. Raises RuntimeError if the residual stays above 1.5cm.
        """
        q = np.asarray(seed5_norm, dtype=float).copy()
        target_xyz = np.asarray(target_xyz, dtype=float)
        for _ in range(iters):
            err = target_xyz - self.fk(q)[:3, 3]
            if np.linalg.norm(err) < tol:
                break
            J = np.zeros((3, 5))
            for j in range(5):
                dq = np.zeros(5)
                dq[j] = 0.5  # normalized units; ~0.5 deg-scale probe
                J[:, j] = (self.fk(q + dq)[:3, 3] - self.fk(q - dq)[:3, 3]) / 1.0
            # Damping relative to the Jacobian scale — an absolute constant here
            # silently over-damps (JJ^T entries are ~1e-4 m^2 for this arm).
            lam = damping * np.trace(J @ J.T) / 3
            step = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
            q += np.clip(step, -8, 8)  # cap per-iteration joint motion
        residual = np.linalg.norm(target_xyz - self.fk(q)[:3, 3])
        if residual > 0.015:
            raise RuntimeError(f"position refinement did not converge (residual {residual*100:.1f}cm)")
        return q


def fit_homography(uv: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, float]:
    """DLT homography pixel (u,v) -> table (x,y); returns (H, RMS error in meters)."""
    import cv2

    H, _ = cv2.findHomography(uv.astype(np.float64), xy.astype(np.float64), method=cv2.RANSAC,
                              ransacReprojThreshold=0.02)
    proj = apply_homography(H, uv)
    rms = float(np.sqrt(((proj - xy) ** 2).sum(axis=1).mean()))
    return H, rms


def apply_homography(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    uv1 = np.concatenate([np.atleast_2d(uv), np.ones((np.atleast_2d(uv).shape[0], 1))], axis=1)
    p = uv1 @ np.asarray(H).T
    return p[:, :2] / p[:, 2:3]


def main():
    calib_path = Path(__file__).parent / "pid_calib.json"
    calib = json.load(open(calib_path))
    samples = calib["samples"]
    pen_uv = np.array(samples["pen_uv"])
    grasp = np.array(samples["grasp_pose"])
    approach = np.array(samples["approach_pose"])

    kin = SO101Kinematics()
    grasp_xyz = np.array([kin.fk(p)[:3, 3] for p in grasp])
    approach_xyz = np.array([kin.fk(p)[:3, 3] for p in approach])

    z = grasp_xyz[:, 2]
    print(f"grasp height z: median={np.median(z)*100:.1f}cm  spread q10..q90 = "
          f"{np.percentile(z,10)*100:.1f}..{np.percentile(z,90)*100:.1f}cm")
    print(f"grasp x range {grasp_xyz[:,0].min()*100:.0f}..{grasp_xyz[:,0].max()*100:.0f}cm, "
          f"y range {grasp_xyz[:,1].min()*100:.0f}..{grasp_xyz[:,1].max()*100:.0f}cm")
    if np.percentile(z, 90) - np.percentile(z, 10) > 0.05:
        print("WARNING: grasp z spread > 5cm — check the norm->deg conversion before trusting IK")

    H, rms = fit_homography(pen_uv, grasp_xyz[:, :2])
    print(f"pixel->table homography RMS: {rms*100:.2f}cm over {len(pen_uv)} grasps")

    # Median grasp orientation (rotation part), as the IK target orientation.
    Rs = np.stack([kin.fk(p)[:3, :3] for p in grasp])
    # Chordal mean + SVD projection back to SO(3).
    M = Rs.mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    R_mean = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt

    # Full camera pose via PnP (assumed webcam intrinsics): lets the runtime project
    # FK positions at ANY height into the image — the homography is only valid for
    # points on the table plane and would parallax-bias an elevated gripper.
    import cv2

    w, h = 640, 480
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    px = (pen_uv * [w, h]).astype(np.float64)
    # SQPNP is globally optimal and handles near-planar point sets; ITERATIVE
    # falls into a mirror minimum on this data.
    ok, rvec, tvec = cv2.solvePnP(grasp_xyz, px, K, None, flags=cv2.SOLVEPNP_SQPNP)
    rvec, tvec = cv2.solvePnPRefineLM(grasp_xyz, px, K, None, rvec, tvec)
    proj, _ = cv2.projectPoints(grasp_xyz, rvec, tvec, K, None)
    pnp_rms = float(np.sqrt(((proj[:, 0] - px) ** 2).sum(axis=1).mean()))
    print(f"PnP camera reprojection RMS: {pnp_rms:.1f}px")

    calib["table"] = {
        "H": H.tolist(),
        "rms_m": rms,
        "z_grasp": float(np.median(z)),
        "z_hover": float(np.median(approach_xyz[:, 2])),
        "R_grasp": R_mean.tolist(),
        "cam": {"K": K.tolist(), "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(),
                "size": [w, h], "rms_px": pnp_rms},
    }
    json.dump(calib, open(calib_path, "w"), indent=1)
    print(f"z_hover={calib['table']['z_hover']*100:.1f}cm; wrote table section to {calib_path}")


if __name__ == "__main__":
    main()

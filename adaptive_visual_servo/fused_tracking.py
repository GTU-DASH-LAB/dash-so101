"""Encoder+vision fused EE tracking: calibrate once, then track for free.

The arm's joint encoders + URDF forward kinematics already give the gripper's
3D position precisely and instantly. The only unknown between that and the
camera image is a FIXED projection (the camera never moves). So instead of
re-measuring the gripper visually on every servo step (blink = robot motion +
frames + settle time, the old bottleneck), we:

  1. during motor babbling, collect (FK(q) 3D, blink pixel 2D) pairs
  2. fit a 3x4 projection matrix P by normalized DLT (camera resectioning,
     Hartley & Zisserman) plus a constant 3D tool offset d absorbing the
     systematic blink-vs-TCP bias (the blink tracks the moving jaw, not the
     jaw-closure point)
  3. afterwards the EE pixel is P @ (FK(q) + d_tcp) -- microseconds, no robot
     motion, no frames. The image Jacobian at any pose is the numeric
     derivative of that composition, so Broyden updating (and its drift) goes
     away entirely.

Vision's remaining jobs: detect the OBJECT (nanodet/bg-sub, unchanged) and
provide the calibration observations. Same architecture as markerless
hand-eye calibration systems in the literature (Kalib arXiv:2408.10562;
Fanello et al., Frontiers Robotics & AI 2018 on iCub): proprioception is the
tracker, vision calibrates it.

Falls back gracefully: if the fit is poor (rms too high) or the babble
points are near-coplanar (DLT degenerate), run_episode keeps the old
blink-per-measurement mode.
"""

import numpy as np
from aruco_tracker import workspace_to_pixel


def _normalize_2d(x):
    c = x.mean(axis=0)
    s = np.sqrt(2) / (np.linalg.norm(x - c, axis=1).mean() + 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    xh = np.concatenate([x, np.ones((len(x), 1))], axis=1) @ T.T
    return xh, T


def _normalize_3d(X):
    c = X.mean(axis=0)
    s = np.sqrt(3) / (np.linalg.norm(X - c, axis=1).mean() + 1e-12)
    U = np.array([[s, 0, 0, -s * c[0]], [0, s, 0, -s * c[1]],
                  [0, 0, s, -s * c[2]], [0, 0, 0, 1]])
    Xh = np.concatenate([X, np.ones((len(X), 1))], axis=1) @ U.T
    return Xh, U


def fit_projection(X, px):
    """Normalized DLT camera resectioning: 3x4 P minimizing algebraic error
    over N>=6 (3D point, pixel) pairs. Returns (P, rms reprojection px)."""
    X, px = np.asarray(X, float), np.asarray(px, float)
    Xh, U = _normalize_3d(X)
    xh, T = _normalize_2d(px)
    rows = []
    for (Xi, xi) in zip(Xh, xh):
        rows.append(np.concatenate([np.zeros(4), -xi[2] * Xi, xi[1] * Xi]))
        rows.append(np.concatenate([xi[2] * Xi, np.zeros(4), -xi[0] * Xi]))
    _, _, Vt = np.linalg.svd(np.stack(rows))
    P = (np.linalg.inv(T) @ Vt[-1].reshape(3, 4) @ U)
    P /= np.linalg.norm(P[2, :3]) + 1e-12
    return P, reproj_rms(P, X, px)


def project(P, X):
    Xh = np.concatenate([np.atleast_2d(X), np.ones((np.atleast_2d(X).shape[0], 1))], axis=1)
    h = Xh @ P.T
    uv = h[:, :2] / h[:, 2:3]
    return uv[0] if np.ndim(X) == 1 else uv


def reproj_rms(P, X, px):
    e = project(P, np.asarray(X)) - np.asarray(px)
    return float(np.sqrt((e ** 2).sum(axis=1).mean()))


def spread_ok(X, min_thickness=0.002):
    """DLT needs non-coplanar points: reject clouds whose thinnest principal
    axis has < ~2mm std (truly flat, e.g. babbling that never varied height).
    Marginal thickness is fine -- fit() additionally cross-validates on
    held-out pairs, which is the real generalization gate."""
    X = np.asarray(X, float)
    if len(X) < 6:
        return False
    sv = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    return bool(sv[-1] / np.sqrt(len(X)) > min_thickness)


class FusedTracker:
    """P(FK(q) + d): fixed camera projection + constant world-frame tool
    offset, fit from babble observations.

    d soaks the blink-vs-TCP bias during FITTING (the observations track the
    moving jaw), but project()/jac() use d=0 -- the URDF's gripper frame IS
    the jaw-closure point we actually want to aim.
    # ponytail: d is world-frame constant (valid while the wrist keeps the
    # gripper pointing down, as in tabletop picks); gripper-frame d via full
    # FK poses if tasks ever vary wrist orientation a lot.
    """

    def __init__(self, fk_pos):
        self.fk = fk_pos          # q -> 3D position of the gripper frame
        self.P = None
        self.d = np.zeros(3)
        self.rms = np.inf
        self._X, self._px = [], []

    def add_pairs(self, qs, pxs):
        for q, px in zip(qs, pxs):
            self._X.append(self.fk(np.asarray(q, float)))
            self._px.append(np.asarray(px, float))

    @staticmethod
    def _fit_pd(X, px, gn_iters=3):
        d = np.zeros(3)
        P = None
        for _ in range(gn_iters):
            P, _ = fit_projection(X + d, px)
            # Gauss-Newton on the 3 tool-offset params
            r = (project(P, X + d) - px).ravel()
            J = np.zeros((len(r), 3))
            eps = 1e-4
            for k in range(3):
                dd = np.zeros(3); dd[k] = eps
                J[:, k] = ((project(P, X + d + dd) - px).ravel() - r) / eps
            step, *_ = np.linalg.lstsq(J, -r, rcond=None)
            d += step
        return P, d

    def fit(self, max_rms=10.0):
        """Fit P and d. The accept gate is CROSS-VALIDATED reprojection error
        (fit on even-indexed pairs, evaluate on odd) -- measures whether the
        model generalizes to poses it wasn't fit on, which is what tracking
        actually needs; in-sample rms alone can look great on a degenerate
        fit. Final model uses all pairs."""
        X, px = np.asarray(self._X), np.asarray(self._px)
        if not spread_ok(X):
            self.rms = np.inf
            return False
        if len(X) >= 10:
            P, d = self._fit_pd(X[::2], px[::2])
            val = reproj_rms(P, X[1::2] + d, px[1::2])
        else:
            P, d = self._fit_pd(X, px)
            val = reproj_rms(P, X + d, px)
        self.P, self.d = self._fit_pd(X, px)
        self.rms = val
        return val <= max_rms

    # ---- tracking (free: no robot motion, no frames) ----
    def ee_px(self, q):
        """Projection of the bare FK gripper frame (d=0)."""
        return project(self.P, self.fk(np.asarray(q, float)))

    def track_px(self, q):
        """Projection of the observed point (FK + fitted tool offset d) --
        the same physical point the blink measurements track, so aiming
        semantics match the proven blink mode exactly. run_episode adds a
        small image-space residual on top (re-anchored by occasional real
        blinks) to absorb the pose-dependent part of FK model error that a
        constant d can't."""
        return project(self.P, self.fk(np.asarray(q, float)) + self.d)

    def jac(self, q, eps=1e-4):
        """2xN image Jacobian of the fused model at q -- replaces babbled J
        and Broyden updates entirely (d is constant, so ee_px/track_px have
        the same derivative)."""
        q = np.asarray(q, float)
        J = np.zeros((2, len(q)))
        for i in range(len(q)):
            d = np.zeros(len(q)); d[i] = eps
            J[:, i] = (self.ee_px(q + d) - self.ee_px(q - d)) / (2 * eps)
        return J


class AnalyticalFusedTracker:
    """Encoder+vision fused tracking without motor babbling.
    Uses calibrated intrinsics + extrinsics (workspace) and the 3D translation
    offset from robot base to workspace coordinates to compute project/jacobian.

    The base offset t_robot_ws is refined ONLINE by a 3-state EKF: every real
    marker measurement the episode already takes (approach anchors, carry
    tracking) is a fresh (q, pixel) pair whose innovation updates t
    recursively -- the one-shot wrist-marker calibration is just the prior.
    EKF rather than plain RLS because the pixel is a NONLINEAR function of t
    (perspective + lens distortion), so each update linearizes h(t) =
    project(FK(q) + t) at the current estimate. A random-walk process noise
    (drift_std) keeps the filter from freezing, so t can also soak the slowly
    pose-dependent part of FK/marker-mount bias; the fast per-pose residual
    stays with control.py's image-space anchor correction c.
    """

    def __init__(self, fk_pos, camera_matrix, dist_coeffs, ws_rvec, ws_tvec, t_robot_ws,
                 offset_std=0.02, meas_px_std=4.0, drift_std=0.001, gate_px=80.0):
        self.fk = fk_pos
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ws_rvec = ws_rvec
        self.ws_tvec = ws_tvec
        self.t_robot_ws = np.asarray(t_robot_ws, float).copy()
        self.rms = 0.0  # Analytical, no fit residual
        # EKF over t_robot_ws: prior spread ~ one-shot calibration accuracy
        self.P_t = np.eye(3) * offset_std ** 2
        self.R_px = np.eye(2) * meas_px_std ** 2
        self.Q_t = np.eye(3) * drift_std ** 2
        self.gate_px = gate_px  # innovation gate: reject marker misdetections
        self.n_updates = 0

    def update(self, q, measured_px):
        """One EKF step on the base offset from a fresh (q, pixel) pair.
        Returns the innovation norm (px), or None if gated as an outlier."""
        pred = self.ee_px(q)
        nu = np.asarray(measured_px, float) - pred
        if not np.all(np.isfinite(nu)) or np.linalg.norm(nu) > self.gate_px:
            return None
        X_ws = self.fk(np.asarray(q, float)) + self.t_robot_ws
        H = np.zeros((2, 3))  # d(pixel)/d(t), finite differences
        eps = 1e-4
        for k in range(3):
            d = np.zeros(3); d[k] = eps
            H[:, k] = (workspace_to_pixel(X_ws + d, self.camera_matrix,
                                          self.dist_coeffs, self.ws_rvec,
                                          self.ws_tvec) - pred) / eps
        P = self.P_t + self.Q_t                    # predict: random-walk offset
        S = H @ P @ H.T + self.R_px
        K = P @ H.T @ np.linalg.inv(S)
        self.t_robot_ws = self.t_robot_ws + K @ nu
        self.P_t = (np.eye(3) - K @ H) @ P
        self.n_updates += 1
        return float(np.linalg.norm(nu))

    def ee_px(self, q):
        X_robot = self.fk(np.asarray(q, float))
        X_ws = X_robot + self.t_robot_ws
        # Project workspace 3D point to pixel
        return workspace_to_pixel(X_ws, self.camera_matrix, self.dist_coeffs, self.ws_rvec, self.ws_tvec)

    def track_px(self, q):
        return self.ee_px(q)

    def jac(self, q, eps=1e-4):
        q = np.asarray(q, float)
        J = np.zeros((2, len(q)))
        for i in range(len(q)):
            d = np.zeros(len(q)); d[i] = eps
            J[:, i] = (self.ee_px(q + d) - self.ee_px(q - d)) / (2 * eps)
        return J


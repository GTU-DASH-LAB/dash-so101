"""Adaptive uncalibrated visual servoing.

The 2x3 image Jacobian J (pixels per joint-unit) is the online-estimated
parameter set: motor babbling identifies it (least squares), Broyden updates
track it while working. No camera model, no accurate kinematics.

Robot interface (duck-typed; SimWorld and the real adapter both provide it):
  get_q() -> np.ndarray, set_q(q), set_gripper(g in [0,1]),
  gripper_contact() -> bool, read() -> BGR frame.
"""

import numpy as np

from perception import blink_locate


def damped_pinv(J, damping=1e-2):
    """Tikhonov-damped pseudo-inverse; damping is relative to J's scale so it
    keeps working whether J is px/rad (sim) or px/deg (real)."""
    JJt = J @ J.T
    mu = damping * np.trace(JJt) / JJt.shape[0] + 1e-9
    return J.T @ np.linalg.inv(JJt + mu * np.eye(JJt.shape[0]))


def broyden_update(J, dq, ds, beta=0.5, min_dq=0.004):
    """Rank-1 Broyden correction, gated: tiny steps carry mostly noise."""
    nq = float(dq @ dq)
    if np.sqrt(nq) < min_dq:
        return J
    return J + beta * np.outer(ds - J @ dq, dq) / nq


def babble(rig, scfg, rng):
    """Motor babbling: random-walk joint probes on a leash around the start
    pose; blink-localize the EE before/after each probe; least-squares fit of
    ds = J dq over all good pairs. Returns (J, s_ee)."""
    q0 = rig.get_q()
    s = blink_locate(rig, scfg)
    if s is None:
        raise RuntimeError("babble: EE blink not visible at start pose")
    amp = np.array(scfg.babble_step)
    q = q0.copy()
    dqs, dss = [], []
    for _ in range(scfg.babble_probes):
        q_target = q0 + rng.uniform(-1.5, 1.5, 3) * amp  # leash: stay near q0
        step = np.clip(q_target - q, -amp, amp)
        q_before = rig.get_q()
        rig.set_q(q + step)
        q = rig.get_q()  # post-clamp truth
        dq = q - q_before
        s_new = blink_locate(rig, scfg)
        if s_new is None or np.linalg.norm(s_new - s) > 120:
            continue  # EE occluded/off-frame or glitch: drop the pair
        dqs.append(dq)
        dss.append(s_new - s)
        s = s_new
    if len(dqs) < 6:
        raise RuntimeError(f"babble: only {len(dqs)} usable probes")
    Q, S = np.stack(dqs), np.stack(dss)
    if np.linalg.matrix_rank(Q, tol=1e-4) < 3:
        raise RuntimeError("babble: probes do not span joint space")
    Jt = np.linalg.solve(Q.T @ Q + 1e-9 * np.eye(3), Q.T @ S)
    return Jt.T, s


def servo_to(rig, scfg, J, s_ee, target_px, locate, tol_px):
    """Closed-loop image servo: drive the EE pixel onto target_px.

    `locate(predicted_px) -> px | None` measures the EE after each move (blink
    while the hand is free, patch tracker while carrying). J is Broyden-updated
    every step; a no-improvement watchdog aborts instead of oscillating.
    Returns (ok, J, s_ee).
    """
    target = np.asarray(target_px, float)
    best = np.inf
    stall = blind = 0
    for _ in range(scfg.max_steps):
        e = target - s_ee
        err = np.linalg.norm(e)
        if err <= tol_px and blind == 0:  # never declare success on prediction alone
            return True, J, s_ee
        if err < best - 1.0:
            best, stall = err, 0
        else:
            stall += 1
            if stall >= scfg.diverge_patience:
                return False, J, s_ee  # caller re-anchors / re-babbles
        dq = damped_pinv(J, scfg.damping) @ (scfg.lam * e)
        dq = np.clip(dq, -scfg.dq_max, scfg.dq_max)
        q_before = rig.get_q()
        rig.set_q(q_before + dq)
        dq_actual = rig.get_q() - q_before  # post-clamp truth
        s_pred = s_ee + J @ dq_actual
        s_new = locate(s_pred)
        # measurement gate: a blink/track wildly off the motion model is an
        # outlier (degenerate blink geometry, occlusion) — remeasure once,
        # then trust the prediction for this step rather than corrupting J
        if s_new is not None and np.linalg.norm(s_new - s_pred) > scfg.reject_px:
            s_new = locate(s_pred)
        if s_new is None or np.linalg.norm(s_new - s_pred) > scfg.reject_px:
            blind += 1
            if blind > scfg.max_blind:
                return False, J, s_ee
            s_ee = s_pred
            continue
        blind = 0
        J = broyden_update(J, dq_actual, s_new - s_ee,
                           scfg.broyden_beta, scfg.min_dq)
        s_ee = s_new
    return False, J, s_ee

"""Adaptive uncalibrated visual servoing.

The 2x3 image Jacobian J (pixels per joint-unit) is the online-estimated
parameter set: motor babbling identifies it (least squares), Broyden updates
track it while working. No camera model, no accurate kinematics.

Robot interface (duck-typed; SimWorld and the real adapter both provide it):
  get_q() -> np.ndarray, set_q(q), set_gripper(g in [0,1]),
  gripper_contact() -> bool, read() -> BGR frame.
"""

import numpy as np

from perception import (blink_locate, calibrate_grasp_frame, detect_objects,
                        detect_pad, locate_by_diff, locate_grasp_point,
                        pick_by_hue)


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
    n = len(q0)
    s = blink_locate(rig, scfg)
    if s is None:
        raise RuntimeError("babble: EE blink not visible at start pose")
    amp = np.full(n, scfg.babble_step)
    q = q0.copy()
    dqs, dss = [], []
    for _ in range(scfg.babble_probes):
        q_target = q0 + rng.uniform(-1.5, 1.5, n) * amp  # leash: stay near q0
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
    if len(dqs) < 2 * n:
        raise RuntimeError(f"babble: only {len(dqs)} usable probes")
    Q, S = np.stack(dqs), np.stack(dss)
    if np.linalg.matrix_rank(Q, tol=1e-4) < n:
        raise RuntimeError("babble: probes do not span joint space")
    Jt = np.linalg.solve(Q.T @ Q + 1e-9 * np.eye(n), Q.T @ S)
    return Jt.T, s


def servo_to(rig, scfg, J, s_ee, target_px, locate, tol_px, shape_dq=None):
    """Closed-loop image servo: drive the EE pixel onto target_px.

    `locate(predicted_px) -> px | None` measures the EE after each move (blink
    while the hand is free, patch tracker while carrying). J is Broyden-updated
    every step; a no-improvement watchdog aborts instead of oscillating.

    `shape_dq(q) -> dq_extra` adds a correction to every step — used to hold
    height with the nominal model, because a pixel target is a whole 3D camera
    ray: without it the null space lets the EE slide down the ray into the
    table. Broyden stays consistent (it sees the executed dq).
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
        if shape_dq is not None:
            dq = np.clip(dq + shape_dq(q_before), -0.2, 0.2)
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


def nominal_z_to(rig, model, z_target, scfg, after_step=None):
    """Open-loop height change using the (deliberately imperfect) nominal
    model — the only place any kinematic model is used. XY stays visually
    closed elsewhere. Returns False if jammed (joint limits)."""
    for _ in range(30):
        q = rig.get_q()
        dz = z_target - model.ee(q)[2]
        if abs(dz) < 0.004:
            return True
        dq = model.step_dz(q, float(np.clip(dz, -scfg.approach_dz, scfg.approach_dz)))
        rig.set_q(q + np.clip(dq, -0.2, 0.2))
        dq_actual = rig.get_q() - q
        if np.linalg.norm(dq_actual) < 1e-5:
            return False
        if after_step is not None:
            after_step(dq_actual)
    return False


def run_episode(rig, model, scfg, background, rng,
                target_hue=None, detector=None, J=None):
    """One pick-and-place: DETECT -> (BABBLE) -> SERVO_XY -> interleaved
    DESCEND -> GRASP -> LIFT -> TRANSPORT -> lower+re-servo -> RELEASE -> HOME.

    `background`: the one-time empty-workspace photo.
    `target_hue`: optional 'pick the <color> one' selector (OpenCV hue).
    `detector`: optional plug-in `f(frame) -> [Blob]` (e.g. a learned model)
    replacing background subtraction.
    `J`: reuse a Jacobian from a previous episode; babbles fresh if None.
    Returns dict(ok, reason, J).
    """
    home_q = rig.get_q()

    def fail(reason):
        rig.set_gripper(1.0)
        return dict(ok=False, reason=reason, J=J)

    frame = rig.read()
    pad_px = detect_pad(frame, scfg.pad_hue, scfg.hue_tol)
    if pad_px is None:
        return fail("drop pad not found")

    def find_objects(fr, extra_exclude=()):
        blobs = (detector(fr) if detector is not None else
                 detect_objects(fr, background, scfg.bg_thresh, scfg.obj_min_area,
                                max_area=scfg.obj_max_area))
        excl = [pad_px, *extra_exclude]
        return [b for b in blobs
                if all(np.linalg.norm(b.center - np.asarray(p)) > 60 for p in excl)]

    blobs = find_objects(frame)
    if not blobs:
        return fail("no objects detected")
    tgt = (pick_by_hue(blobs, target_hue, scfg.hue_tol)
           if target_hue is not None else blobs[0])
    if tgt is None:
        return fail("no object matches requested hue")
    obj_px = np.asarray(tgt.center)

    try:
        if J is None:
            J, s = babble(rig, scfg, rng)
        else:
            s = blink_locate(rig, scfg)
            if s is None:
                J, s = babble(rig, scfg, rng)
    except RuntimeError as e:
        return fail(f"babble failed: {e}")

    # self-calibrate the jaw-closure point (single-moving-jaw grippers aim
    # 25-48px off if the servo tracks the raw blink centroid = the moving jaw;
    # see perception.calibrate_grasp_frame). Free air here: hand is at the
    # babble/travel pose, not near objects. Falls back to the raw blink if
    # calibration fails (symmetric grippers return (0,0) and behave as before).
    grasp_ab = calibrate_grasp_frame(rig, scfg)

    def bl(pred):
        if grasp_ab is not None:
            return locate_grasp_point(rig, scfg, grasp_ab)
        return blink_locate(rig, scfg)

    s2 = bl(None)  # re-anchor: babble measured the raw blink point, not this one
    if s2 is not None:
        s = s2

    def z_hold(z_ref):
        """Per-step correction keeping nominal height at z_ref while the
        image loop owns XY (see servo_to.shape_dq). Clamped to the servo's
        own per-joint authority (dq_max): near full arm extension the IK
        returns huge joint deltas for small dz, and an uncapped hold term
        was observed dragging the EE monotonically AWAY from a nearly
        converged target for 10+ steps (servo clamped at dq_max, hold not)."""
        def f(q):
            dz = float(np.clip(0.6 * (z_ref - model.ee(q)[2]),
                               -scfg.approach_dz, scfg.approach_dz))
            return np.clip(model.step_dz(q, dz), -scfg.dq_max, scfg.dq_max)
        return f

    def servo_recover(target, tol, hold):
        """Servo with escalating recovery: re-anchor blink, then re-babble."""
        nonlocal J, s
        ok, J, s = servo_to(rig, scfg, J, s, target, bl, tol, shape_dq=hold)
        if ok:
            return True
        s2 = bl(None)
        if s2 is not None:
            s = s2
        ok, J, s = servo_to(rig, scfg, J, s, target, bl, tol, shape_dq=hold)
        if ok:
            return True
        try:
            J, _ = babble(rig, scfg, rng)
        except RuntimeError:
            return False  # e.g. EE not visible right now -- recovery failed, not a crash
        s2 = bl(None)  # babble tracks the raw blink point; re-anchor to ours
        if s2 is not None:
            s = s2
        ok, J, s = servo_to(rig, scfg, J, s, target, bl, tol, shape_dq=hold)
        return ok

    # ---- approach + descend + grasp, with retries ----
    grasped = False
    for attempt in range(scfg.retries + 1):
        nominal_z_to(rig, model, scfg.approach_z, scfg)
        s_up = bl(None)
        if s_up is not None:
            s = s_up
        if not servo_recover(obj_px, scfg.tol_coarse_px, z_hold(scfg.approach_z)):
            return fail("approach servo failed")
        # interleaved descend: parallax shrinks as height drops. Capped, not
        # `while True` -- a real IK backend (unlike the toy sim's always-
        # solvable 3-DOF analytic model) can plateau at some pose and make no
        # further Z progress, which an unbounded loop would spin on forever.
        ok_descend = True
        max_stages = 3 * int(np.ceil((scfg.approach_z - scfg.grasp_z) / scfg.approach_dz)) + 5
        for _ in range(max_stages):
            z = model.ee(rig.get_q())[2]
            if z <= scfg.grasp_z + 0.004:
                break
            stage = max(scfg.grasp_z, z - scfg.approach_dz)
            nominal_z_to(rig, model, stage, scfg)
            s2 = bl(None)
            if s2 is not None:
                s = s2
            if not servo_recover(obj_px, scfg.tol_fine_px, z_hold(stage)):
                ok_descend = False
                break
        else:
            ok_descend = False  # never reached grasp_z within the stage budget
        if ok_descend:
            rig.set_gripper(0.0)
            if rig.gripper_contact():
                grasped = True
                break
        # retry: reopen, rise, re-find the object (it may have been nudged)
        rig.set_gripper(1.0)
        nominal_z_to(rig, model, scfg.approach_z, scfg)
        blobs = find_objects(rig.read(), extra_exclude=(s,))
        if blobs:
            tgt = (pick_by_hue(blobs, target_hue, scfg.hue_tol)
                   if target_hue is not None else
                   min(blobs, key=lambda b: np.linalg.norm(b.center - obj_px)))
            if tgt is not None:
                obj_px = np.asarray(tgt.center)
    if not grasped:
        return fail("grasp failed after retries")

    # ---- lift, tracking the hand+object via bg-diff (no blinking while holding) ----
    def track_step(dq):
        nonlocal s
        pred = s + J @ dq
        new_s = locate_by_diff(rig.read(), background, pred, scfg.track_roi,
                               scfg.bg_thresh, scfg.obj_min_area, scfg.track_max_jump)
        s = new_s if new_s is not None else pred

    nominal_z_to(rig, model, scfg.lift_z, scfg, after_step=track_step)
    if not rig.gripper_contact():
        return fail("object dropped during lift")

    def tr(pred):
        return locate_by_diff(rig.read(), background, pred, scfg.track_roi,
                              scfg.bg_thresh, scfg.obj_min_area, scfg.track_max_jump)

    ok, J, s = servo_to(rig, scfg, J, s, pad_px, tr, scfg.tol_coarse_px,
                        shape_dq=z_hold(scfg.lift_z))
    if not ok:
        return fail("transport servo failed")
    # lower over the pad and re-servo: kills the remaining parallax offset
    nominal_z_to(rig, model, scfg.release_z, scfg, after_step=track_step)
    ok, J, s = servo_to(rig, scfg, J, s, pad_px, tr, scfg.tol_coarse_px,
                        shape_dq=z_hold(scfg.release_z))
    rig.set_gripper(1.0)
    nominal_z_to(rig, model, scfg.lift_z, scfg)
    q = rig.get_q()
    for a in np.linspace(0.2, 1.0, 5):
        rig.set_q(q + a * (home_q - q))
    return dict(ok=True, reason="released over pad", J=J)

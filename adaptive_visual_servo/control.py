"""Adaptive uncalibrated visual servoing.

The 2x3 image Jacobian J (pixels per joint-unit) is the online-estimated
parameter set: motor babbling identifies it (least squares), Broyden updates
track it while working. No camera model, no accurate kinematics.

Robot interface (duck-typed; SimWorld and the real adapter both provide it):
  get_q() -> np.ndarray, set_q(q), set_gripper(g in [0,1]),
  gripper_contact() -> bool, read() -> BGR frame.
"""

from dataclasses import replace

import time
import numpy as np

from fused_tracking import FusedTracker
from perception import (blink_locate, calibrate_grasp_frame, detect_objects,
                        detect_pad, filter_by_background, locate_by_diff,
                        locate_grasp_point, pick_by_hue, locate_by_marker,
                        locate_by_aruco)


_LAST_GRIPPER_WRIST_OFFSET = None


def locate_gripper(rig, scfg, predicted_px=None):
    global _LAST_GRIPPER_WRIST_OFFSET
    if getattr(scfg, "use_aruco", False):
        frame = rig.read()
        g_px = locate_by_aruco(frame, scfg.aruco_id, predicted_px=predicted_px,
                               max_dist=scfg.gripper_search_radius)
        w_id = getattr(scfg, "aruco_wrist_id", 49)
        w_px = locate_by_aruco(frame, w_id, predicted_px=predicted_px,
                               max_dist=scfg.gripper_search_radius)
        
        if g_px is not None and w_px is not None:
            _LAST_GRIPPER_WRIST_OFFSET = g_px - w_px
            return g_px
        elif g_px is not None:
            return g_px
        elif w_px is not None:
            if _LAST_GRIPPER_WRIST_OFFSET is not None:
                return w_px + _LAST_GRIPPER_WRIST_OFFSET
            else:
                return w_px
        return None
    elif getattr(scfg, "gripper_marker_hue", None) is not None:
        return locate_by_marker(rig.read(), scfg.gripper_marker_hue,
                                scfg.hue_tol, scfg.gripper_marker_sat_min,
                                scfg.gripper_marker_val_min,
                                scfg.gripper_marker_area_min,
                                scfg.gripper_marker_area_max,
                                predicted_px=predicted_px, max_dist=scfg.gripper_search_radius)
    else:
        return blink_locate(rig, scfg, predicted_px=predicted_px, max_dist=scfg.gripper_search_radius)


def scan_wrist_for_marker(rig, scfg, max_scan_deg=25.0, debug=None):
    """Scan wrist flex angles (previous joint before wrist roll) and cycle gripper
    open/close to bring the ArUco marker into view of the camera."""
    debug = debug or (lambda ev: None)
    q = rig.get_q()
    flex_idx = 3  # wrist_flex is joint index 3

    if locate_gripper(rig, scfg) is not None:
        return True

    debug(dict(msg="  [scan_wrist] Marker not visible -- scanning wrist "
                   f"±{max_scan_deg:.0f}° + gripper cycles (can take ~30s)..."))
    orig_flex = q[flex_idx]

    # Try cycling the gripper open/closed first at current pose
    for g_val in [0.0, 1.0, 0.5]:
        rig.set_gripper(g_val)
        time.sleep(0.3)
        if locate_gripper(rig, scfg) is not None:
            debug(dict(msg="  [scan_wrist] Marker recovered by gripper cycle."))
            return True

    # Scan wrist_flex (tilting the hand up/down) to face the camera
    scan_steps = np.linspace(-np.radians(max_scan_deg), np.radians(max_scan_deg), 7)
    for step in scan_steps:
        q_new = q.copy()
        q_new[flex_idx] = orig_flex + step
        rig.set_q(q_new)
        time.sleep(0.3)
        for g_val in [0.0, 1.0]:
            rig.set_gripper(g_val)
            time.sleep(0.3)
            if locate_gripper(rig, scfg) is not None:
                debug(dict(msg=f"  [scan_wrist] Marker recovered at flex offset "
                               f"{np.degrees(step):+.0f}°."))
                return True

    # Restore original configurations if not found
    q_restore = q.copy()
    q_restore[flex_idx] = orig_flex
    rig.set_q(q_restore)
    rig.set_gripper(0.0)
    debug(dict(msg="  [scan_wrist] Marker NOT found after full scan."))
    return False


def damped_pinv(J, damping=1e-2):
    """Tikhonov-damped pseudo-inverse; damping is relative to J's scale so it
    keeps working whether J is px/rad (sim) or px/deg (real)."""
    JJt = J @ J.T
    mu = damping * np.trace(JJt) / JJt.shape[0] + 1e-9
    return J.T @ np.linalg.inv(JJt + mu * np.eye(JJt.shape[0]))


def broyden_update(J, dq, ds, beta=0.5, min_dq=0.004, adaptive=False):
    """Rank-1 Broyden correction, gated: tiny steps carry mostly noise."""
    nq = float(dq @ dq)
    if np.sqrt(nq) < min_dq:
        return J
    if adaptive:
        pred = J @ dq
        err = np.linalg.norm(ds - pred)
        mag = np.linalg.norm(ds) + 1e-5
        rel_err = err / mag
        # Scale beta between beta*0.5 (low error, noise filter) and beta*1.6 (high error, fast learning)
        adaptive_beta = float(np.clip(beta * (0.5 + 1.0 * rel_err), beta * 0.5, min(beta * 1.6, 0.95)))
    else:
        adaptive_beta = beta
    return J + adaptive_beta * np.outer(ds - J @ dq, dq) / nq


def make_null_fn(scfg, q_vis_ref):
    """Secondary-objective joint velocity for servo_to's null-space term:

      dq_null = joint-limit repulsion (potential field) + wrist posture
                attraction toward the last marker-visible pose

    servo_to projects it through (I - J+J), so it never disturbs the image
    task to first order. `q_vis_ref` is a dict whose "q" run_episode updates
    on every real marker sighting -- pulling the wrist back toward a pose
    where the marker demonstrably faced the camera is what keeps it visible
    without ever commanding the EE pixel off target.

    Limit potential: U = sum over joints of max(0, |u|-dz)^2 with
    u = (q-mid)/half in [-1, 1]; dq = -k_lim * dU/dq stays exactly zero
    inside the deadzone so mid-range motion is never biased.
    Returns None when neither objective is configured."""
    lim_on = getattr(scfg, "q_lo", None) is not None and scfg.null_k_lim > 0
    vis_on = scfg.null_k_vis > 0
    if not lim_on and not vis_on:
        return None
    if lim_on:
        lo, hi = np.asarray(scfg.q_lo, float), np.asarray(scfg.q_hi, float)
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    dz = scfg.null_deadzone

    def null_fn(q):
        dq = np.zeros_like(q)
        if lim_on:
            u = (q - mid) / half                     # -1 at q_lo .. +1 at q_hi
            over = (np.abs(u) - dz) / (1.0 - dz)
            push = np.where(over > 0, over ** 2, 0.0)
            dq -= scfg.null_k_lim * np.sign(u) * push
        q_vis = q_vis_ref.get("q")
        if vis_on and q_vis is not None and len(q) > max(scfg.wrist_joints):
            for j in scfg.wrist_joints:
                dq[j] += scfg.null_k_vis * (q_vis[j] - q[j])
        return dq

    return null_fn


def babble(rig, scfg, rng, pair_sink=None):
    """Motor babbling: random-walk joint probes on a leash around the start
    pose; blink-localize the EE before/after each probe; least-squares fit of
    ds = J dq over all good pairs. Returns (J, s_ee).

    `pair_sink`: optional list collecting (q, blink_px) calibration pairs for
    fused_tracking's camera fit -- the same probes serve both purposes."""
    q0 = rig.get_q()
    n = len(q0)
    s = locate_gripper(rig, scfg)
    if s is None:
        raise RuntimeError("babble: EE blink not visible at start pose")
    if pair_sink is not None:
        pair_sink.append((q0.copy(), s.copy()))
    amp = np.full(n, scfg.babble_step)
    q = q0.copy()
    dqs, dss = [], []
    n_none = n_glitch = 0
    for _ in range(scfg.babble_probes):
        q_target = q0 + rng.uniform(-1.5, 1.5, n) * amp  # leash: stay near q0
        step = np.clip(q_target - q, -amp, amp)
        q_before = rig.get_q()
        rig.set_q(q + step)
        q = rig.get_q()  # post-clamp truth
        dq = q - q_before
        s_new = locate_gripper(rig, scfg, predicted_px=s)
        if s_new is None:
            n_none += 1
            continue  # blink found no gripper motion in the image
        if np.linalg.norm(s_new - s) > 120:
            n_glitch += 1
            continue  # jumped implausibly far: occlusion/misdetection glitch
        dqs.append(dq)
        dss.append(s_new - s)
        s = s_new
        if pair_sink is not None:
            pair_sink.append((q.copy(), s.copy()))
    if len(dqs) < 2 * n:
        raise RuntimeError(
            f"babble: only {len(dqs)}/{scfg.babble_probes} usable probes "
            f"({n_none} blinks saw no gripper motion, {n_glitch} jumped >120px). "
            "On real hardware, usual causes in order: gripper_open_pos/"
            "gripper_closed_pos miscalibrated so the fingers barely move "
            "between blink positions (run --calibrate-gripper, then use the "
            "UI's Test Blink), camera auto-exposure flicker, or the gripper "
            "out of the camera's view at the babble pose.")
    Q, S = np.stack(dqs), np.stack(dss)
    if np.linalg.matrix_rank(Q, tol=1e-4) < n:
        raise RuntimeError("babble: probes do not span joint space")
    Jt = np.linalg.solve(Q.T @ Q + 1e-9 * np.eye(n), Q.T @ S)
    return Jt.T, s


def servo_to(rig, scfg, J, s_ee, target_px, locate, tol_px, shape_dq=None,
             jac_fn=None, null_fn=None, debug=None):
    """Closed-loop image servo: drive the EE pixel onto target_px.

    `locate(predicted_px) -> px | None` measures the EE (blink while the hand
    is free, bg-diff while carrying). Measuring costs real robot motion
    (gripper blinks), so it happens only every `scfg.measure_every` steps —
    in between, the estimated Jacobian dead-reckons (that IS the adaptive
    model earning its keep). Broyden updates use the accumulated dq since the
    last good measurement. Success is only ever declared on a fresh
    measurement, never on prediction. A no-improvement watchdog (counted on
    measurements) aborts instead of oscillating.

    `shape_dq(q) -> dq_extra` adds a correction to every step — used to hold
    height with the nominal model, because a pixel target is a whole 3D camera
    ray: without it the null space lets the EE slide down the ray into the
    table. Broyden stays consistent (it sees the executed dq).

    `jac_fn(q) -> 2xN`: fused-tracking mode — the image Jacobian comes from
    the calibrated camera+FK model at every step (fresh, pose-exact), so
    Broyden updating is skipped entirely.

    `null_fn(q) -> dq_null` (see make_null_fn): secondary objectives (joint-
    limit repulsion, marker-visibility posture) added through the null-space
    projector: dq = J+(lam*e) + (I - J+J) dq_null.
    Returns (ok, J, s_ee).
    """
    target = np.asarray(target_px, float)
    best = np.inf
    stall = blind = since = 0
    s_base = s_ee.copy()                    # s at last good measurement
    dq_acc = np.zeros(len(rig.get_q()))     # executed dq since then
    fresh = True                            # s_ee comes from a measurement
    debug = debug or (lambda ev: None)
    for step in range(scfg.max_steps):
        e = target - s_ee
        err_norm = np.linalg.norm(e)
        if step % 5 == 0 or err_norm <= tol_px:
            debug(dict(msg=f"  [servo_to] step {step}/{scfg.max_steps}: err={err_norm:.1f}px (tol={tol_px}px) stall={stall}"))
        if err_norm <= tol_px and fresh and blind == 0:
            debug(dict(msg=f"  [servo_to] Success: converged in {step} steps!"))
            return True, J, s_ee
        q_before = rig.get_q()
        if jac_fn is not None:
            J = jac_fn(q_before)
        err_norm = np.linalg.norm(e)
        if getattr(scfg, "stiction_comp", False):
            # Smoothly boost lam from scfg.lam (large error) towards 0.95 (small error) to overcome stiction
            adaptive_lam = float(scfg.lam + (0.95 - scfg.lam) * np.exp(-err_norm / 15.0))
            # tracking-rate term: measured progress has stalled (stiction /
            # under-modeled J eating the command) -> push proportionally harder
            adaptive_lam = min(0.95, adaptive_lam * (1.0 + 0.15 * stall))
        else:
            adaptive_lam = scfg.lam
        Jp = damped_pinv(J, scfg.damping)
        dq = np.clip(Jp @ (adaptive_lam * e), -scfg.dq_max, scfg.dq_max)
        if null_fn is not None:
            # secondary objectives ride the null space: zero first-order pixel
            # motion, so the image task never sees them
            dq_ns = (np.eye(len(q_before)) - Jp @ J) @ null_fn(q_before)
            dq = np.clip(dq + dq_ns, -scfg.dq_max, scfg.dq_max)
        if shape_dq is not None:
            dq = np.clip(dq + shape_dq(q_before), -0.2, 0.2)
        rig.set_q(q_before + dq)
        dq_actual = rig.get_q() - q_before  # post-clamp truth
        dq_acc += dq_actual
        s_pred = s_ee + J @ dq_actual
        since += 1
        # dead-reckon between measurements — unless prediction says we're at
        # the target, which must be confirmed by a real measurement
        if since < scfg.measure_every and np.linalg.norm(target - s_pred) > tol_px:
            s_ee, fresh = s_pred, False
            continue
        since = 0
        s_new = locate(s_pred)
        # measurement gate: a blink/track wildly off the motion model is an
        # outlier (degenerate blink geometry, occlusion) — remeasure once,
        # then trust the prediction for this step rather than corrupting J.
        # In fused mode (jac_fn set) the "measurement" is the calibrated FK
        # model — deterministic, no outliers, no occlusion — so the gate is
        # skipped entirely. The gate was the primary cause of false approach
        # failures in fused mode: near joint limits, set_q clips the command,
        # so the linear prediction s_pred = s + J·dq diverges from the
        # nonlinear FK recomputation, tripping the reject threshold.
        if jac_fn is None:  # blink/bg-diff mode: real visual measurement
            if s_new is not None and np.linalg.norm(s_new - s_pred) > scfg.reject_px:
                s_new = locate(s_pred)
            if s_new is None or np.linalg.norm(s_new - s_pred) > scfg.reject_px:
                blind += 1
                if blind > scfg.max_blind:
                    return False, J, s_ee
                s_ee, fresh = s_pred, False
                continue
        blind = 0
        if jac_fn is None:  # fused mode: J is model-derived fresh each step
            J = broyden_update(J, dq_acc, s_new - s_base,
                               scfg.broyden_beta, scfg.min_dq,
                               adaptive=getattr(scfg, "stiction_comp", False))
        s_ee, s_base, fresh = s_new, s_new.copy(), True
        dq_acc = np.zeros_like(dq_acc)
        err = np.linalg.norm(target - s_new)
        # fractional improvement threshold: in blink mode, noise jitters
        # the error by several px per measurement, so 1px is a genuine
        # "no improvement" signal. In fused mode (noiseless FK), the servo
        # converges smoothly at <1px/step near the target — legitimate
        # progress that a fixed 1px gate mistakes for divergence.
        if jac_fn is not None:
            if err < best:
                best, stall = err, 0
            else:
                stall += 1
        else:
            improvement = 1.0
            if err < best - improvement:
                best, stall = err, 0
            else:
                stall += 1
        if stall >= scfg.diverge_patience:
            debug(dict(msg=f"  [servo_to] Aborted: stall limit ({scfg.diverge_patience}) reached!"))
            return False, J, s_ee  # caller re-anchors / re-babbles
    debug(dict(msg=f"  [servo_to] Aborted: maximum steps ({scfg.max_steps}) reached without convergence."))
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
                target_hue=None, detector=None, J=None, debug=None,
                manual_target_px=None, manual_pad_px=None, tracker=None):
    """One pick-and-place: DETECT -> (BABBLE) -> SERVO_XY -> interleaved
    DESCEND -> GRASP -> LIFT -> TRANSPORT -> lower+re-servo -> RELEASE -> HOME.

    `background`: the one-time empty-workspace photo.
    `target_hue`: optional 'pick the <color> one' selector (OpenCV hue).
    `detector`: optional plug-in `f(frame) -> [Blob]` (e.g. a learned model)
    replacing background subtraction.
    `J`: reuse a Jacobian from a previous episode; babbles fresh if None.
    `tracker`: reuse a fitted FusedTracker from a previous episode (returned
    in the result dict) -- the camera is fixed, so episode 2+ skips babbling
    entirely: EE tracking is pure FK projection from the first frame.
    `debug`: optional callable(dict) fed detection/target/EE-estimate events,
    for live visualization (see run_pb_sim.py's controller-view window).
    `manual_target_px`/`manual_pad_px`: skip detection entirely and servo to
    a human-picked pixel instead -- e.g. clicked in a UI. Manual XY is exactly
    as valid a target as a detected one: the servo only ever needs a pixel to
    aim at, never what put it there. Still adaptive control end to end (still
    babbles/Broyden-updates the Jacobian, still visually closes the loop) --
    the only thing skipped is *finding* the pixel, not *reaching* it.
    Returns dict(ok, reason, J).
    """
    home_q = rig.get_q()
    rig.set_gripper(0.0) # Close gripper during initial transit and XY servo
    debug = debug or (lambda ev: None)
    fused = tracker if (tracker is not None and scfg.fuse) else None

    # marker-visibility reference for the null-space controller: updated at
    # every REAL marker sighting, so the wrist is continuously pulled back
    # toward a camera-facing pose instead of drifting until a ~30s scan
    q_vis = {"q": None}

    def note_marker_seen(q_now=None):
        if getattr(scfg, "use_aruco", False):
            q_vis["q"] = (rig.get_q() if q_now is None else q_now).copy()

    null_fn = make_null_fn(scfg, q_vis)

    def fail(reason):
        debug(dict(msg=f"[episode] FAILED: {reason} -- raising arm and returning home."))
        rig.set_gripper(1.0)
        # Safely raise Z to travel height first
        nominal_z_to(rig, model, scfg.lift_z, scfg)
        # Smoothly return the arm to home position
        rig.set_q(home_q)
        return dict(ok=False, reason=reason, J=J, tracker=fused,
                    fused_rms=(fused.rms if fused is not None else None))

    frame = rig.read()
    if manual_pad_px is not None:
        pad_px = np.asarray(manual_pad_px, float)
    else:
        pad_px = detect_pad(frame, scfg.pad_hue, scfg.hue_tol)
        if pad_px is None:
            return fail("drop pad not found")

    def find_objects(fr, extra_exclude=()):
        if detector is not None:
            # cross-check learned detections against background subtraction:
            # a pickable object must also differ from the empty-workspace
            # photo. This drops detections of the ROBOT ARM itself (parked at
            # home in that photo, so bg-sub never sees it) -- observed live:
            # nanodet boxed the whole arm as its largest detection and the
            # servo chased its own arm instead of the cube.
            blobs = filter_by_background(detector(fr), fr, background,
                                         scfg.bg_thresh, scfg.obj_min_area)
        else:
            blobs = detect_objects(fr, background, scfg.bg_thresh, scfg.obj_min_area,
                                   max_area=scfg.obj_max_area)
        excl = [pad_px, *extra_exclude]
        return [b for b in blobs
                if all(np.linalg.norm(b.center - np.asarray(p)) > 60 for p in excl)]

    if manual_target_px is not None:
        obj_px = np.asarray(manual_target_px, float)
    else:
        blobs = find_objects(frame)
        if not blobs:
            return fail("no objects detected")
        tgt = (pick_by_hue(blobs, target_hue, scfg.hue_tol)
               if target_hue is not None else blobs[0])
        if tgt is None:
            return fail("no object matches requested hue")
        obj_px = np.asarray(tgt.center)
    debug(dict(pad_px=pad_px, target=obj_px))

    try:
        if fused is not None:
            s = fused.track_px(rig.get_q())
            J = fused.jac(rig.get_q())
        elif scfg.fuse and model is not None:
            pairs = []
            J, s = babble(rig, scfg, rng, pair_sink=pairs)
            t = FusedTracker(model.ee)
            t.add_pairs([p[0] for p in pairs], [p[1] for p in pairs])
            if t.fit(scfg.fuse_max_rms):
                fused = t
            # else: fit didn't generalize -- keep the babbled J + blink mode
        elif J is None:
            J, s = babble(rig, scfg, rng)
        else:
            s = locate_gripper(rig, scfg)
            if s is None:
                J, s = babble(rig, scfg, rng)
    except RuntimeError as e:
        return fail(f"babble failed: {e}")

    if fused is not None:
        # encoder+vision fusion: per-STEP tracking is P(FK(encoders)+d)+c --
        # free, instant, noiseless -- and the model Jacobian replaces Broyden.
        # c is an image-space residual re-anchored by ONE real visual
        # measurement at each phase transition (approach start, each descend
        # stage): it absorbs the pose-dependent part of FK model error that
        # the constant fitted offset d can't. Net: a handful of blinks per
        # episode instead of one per servo step, with locally-exact aim.
        scfg = replace(scfg, measure_every=1)
        jac_fn = fused.jac
        c = np.zeros(2)

        def loc_coarse(pred):
            r = fused.track_px(rig.get_q()) + c
            debug(dict(s=r))
            return r

        loc_fine = loc_coarse

        grasp_ab = getattr(fused, "grasp_ab", None)  # closure offset reused across episodes
        if grasp_ab is None:
            grasp_ab = calibrate_grasp_frame(rig, scfg, predicted_px=s)
            fused.grasp_ab = grasp_ab

        # AnalyticalFusedTracker: recursive (EKF) base-offset refinement from
        # the same real marker measurements the anchors already take. Only
        # marker-point measurements feed it -- anchor_fine measures the jaw
        # CLOSURE point, a different physical point that would bias the offset.
        kf_update = getattr(fused, "update", None)

        def anchor_coarse():
            q_now = rig.get_q()
            pred = fused.track_px(q_now)
            m = locate_gripper(rig, scfg, predicted_px=pred)
            if m is not None:
                if kf_update is not None:
                    kf_update(q_now, m)
                c[:] = m - fused.track_px(q_now)  # residual on the UPDATED model
                note_marker_seen(q_now)
            return loc_coarse(None)

        def anchor_fine():
            pred = fused.track_px(rig.get_q())
            m = (locate_grasp_point(rig, scfg, grasp_ab, predicted_px=pred) if grasp_ab is not None
                 else locate_gripper(rig, scfg, predicted_px=pred))
            if m is not None:
                c[:] = m - pred
            return loc_coarse(None)

        J = fused.jac(rig.get_q())
        s = loc_coarse(None)
    else:
        jac_fn = None
        # self-calibrate the jaw-closure point (single-moving-jaw grippers aim
        # 25-48px off if the servo tracks the raw blink centroid = the moving
        # jaw; see perception.calibrate_grasp_frame). Free air here: hand is at
        # the babble/travel pose, not near objects. Falls back to the raw blink
        # if calibration fails (symmetric grippers return (0,0) as before).
        grasp_ab = calibrate_grasp_frame(rig, scfg, predicted_px=s)

        def loc_coarse(pred):
            """Raw blink: one gripper wiggle. Cheap; tracks the moving jaw --
            fine for the coarse traverse, the ~40px jaw offset doesn't matter
            until we're lining up the grasp."""
            r = locate_gripper(rig, scfg, predicted_px=pred)
            if r is not None:
                note_marker_seen()
            debug(dict(s=r))
            return r

        def loc_fine(pred):
            """Closure-point locator: two blinks + calibrated offset. Used for
            descend/grasp alignment where the aim point must be the real grasp
            point, not the moving jaw."""
            if grasp_ab is None:
                return loc_coarse(pred)
            r = locate_grasp_point(rig, scfg, grasp_ab, predicted_px=pred)
            debug(dict(s=r))
            return r

        def anchor_coarse():
            return loc_coarse(None)

        def anchor_fine():
            return loc_fine(None)

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

    def servo_recover(target, tol, hold, loc):
        """Servo with escalating recovery: re-anchor, then re-babble (in
        fused mode the re-babble's blinks become fresh calibration pairs)."""
        nonlocal J, s
        debug(dict(msg=f"[servo_recover] Phase 1: Servoing to target with current Jacobian..."))
        ok, J, s = servo_to(rig, scfg, J, s, target, loc, tol, shape_dq=hold,
                            jac_fn=jac_fn, null_fn=null_fn, debug=debug)
        if ok:
            return True

        debug(dict(msg=f"[servo_recover] Phase 1 failed. Phase 2: Re-anchoring position..."))
        s2 = loc(None)
        if s2 is not None:
            s = s2
        ok, J, s = servo_to(rig, scfg, J, s, target, loc, tol, shape_dq=hold,
                            jac_fn=jac_fn, null_fn=null_fn, debug=debug)
        if ok:
            return True
            
        debug(dict(msg=f"[servo_recover] Phase 2 failed. Phase 3: Escalating to motor babbling to update model parameters..."))
        try:
            pairs = [] if fused is not None else None
            J2, _ = babble(rig, scfg, rng, pair_sink=pairs)
            if fused is not None:
                debug(dict(msg="[servo_recover] Re-fitting FusedTracker with new calibration pairs..."))
                fused.add_pairs([p[0] for p in pairs], [p[1] for p in pairs])
                fused.fit(scfg.fuse_max_rms)  # refit with more data
            else:
                J = J2
        except RuntimeError as e:
            debug(dict(msg=f"[servo_recover] Babble failed: {e}"))
            return False  # e.g. EE not visible right now -- recovery failed, not a crash
            
        s2 = loc(None)  # babble tracks the raw blink point; re-anchor to ours
        if s2 is not None:
            s = s2
        ok, J, s = servo_to(rig, scfg, J, s, target, loc, tol, shape_dq=hold,
                            jac_fn=jac_fn, null_fn=null_fn, debug=debug)
        return ok

    # ---- approach + descend + grasp, with retries ----
    grasped = False
    for attempt in range(scfg.retries + 1):
        debug(dict(msg=f"[episode] Approach attempt {attempt + 1}/{scfg.retries + 1}: "
                       f"rising to z={scfg.approach_z:.3f}m..."))
        nominal_z_to(rig, model, scfg.approach_z, scfg)
        if getattr(scfg, "use_aruco", False):
            scan_wrist_for_marker(rig, scfg, debug=debug)
        s_up = anchor_coarse()  # fused mode: one real blink re-grounds the model here
        if s_up is not None:
            s = s_up
        if not servo_recover(obj_px, scfg.tol_coarse_px, z_hold(scfg.approach_z),
                             loc_coarse):
            return fail("approach servo failed")

        # Open gripper right before descending, near the object
        rig.set_gripper(1.0)
        time.sleep(0.5)

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
            debug(dict(msg=f"[episode] Descend stage: z={z:.3f} -> {stage:.3f}m "
                           f"(grasp at {scfg.grasp_z:.3f}m)"))
            nominal_z_to(rig, model, stage, scfg)
            s2 = anchor_fine()  # fused: one closure-point blink per stage, not per step
            if s2 is None and getattr(scfg, "use_aruco", False):
                scan_wrist_for_marker(rig, scfg, debug=debug)
                s2 = anchor_fine()
            if s2 is not None:
                s = s2
            if not servo_recover(obj_px, scfg.tol_fine_px, z_hold(stage), loc_fine):
                ok_descend = False
                break
        else:
            ok_descend = False  # never reached grasp_z within the stage budget
        if ok_descend:
            debug(dict(msg="[episode] At grasp height -- closing gripper..."))
            rig.set_gripper(0.0)
            if rig.gripper_contact():
                debug(dict(msg="[episode] Grasp confirmed (jaw stalled on object)."))
                grasped = True
                break
            debug(dict(msg="[episode] Grasp check failed (no contact) -- retrying."))
        # retry: reopen, rise, re-find the object (it may have been nudged).
        # Manual target: nothing to re-detect, just retry at the same pixel.
        rig.set_gripper(1.0)
        nominal_z_to(rig, model, scfg.approach_z, scfg)
        if manual_target_px is None:
            blobs = find_objects(rig.read(), extra_exclude=(s,))
            if blobs:
                tgt = (pick_by_hue(blobs, target_hue, scfg.hue_tol)
                       if target_hue is not None else
                       min(blobs, key=lambda b: np.linalg.norm(b.center - obj_px)))
                if tgt is not None:
                    obj_px = np.asarray(tgt.center)
    if not grasped:
        return fail("grasp failed after retries")

    # ---- lift + carry tracking ----
    if fused is not None:
        # can't blink while holding, and the model's residual c goes stale
        # across a long traverse (FK bias is pose-dependent -- observed:
        # grasp landed 4mm accurate, then transport failed on model-only
        # tracking). So the carry phase fuses both: the model predicts (free,
        # exact prior), bg-diff CONFIRMS with a real 1-frame measurement (no
        # robot motion), and each accepted measurement re-anchors c.
        def _carry_meas():
            q_now = rig.get_q()
            pred = fused.track_px(q_now) + c
            if getattr(scfg, "use_aruco", False):
                m = locate_by_aruco(rig.read(), scfg.aruco_id, predicted_px=pred,
                                   max_dist=scfg.gripper_search_radius)
                if m is not None:
                    if kf_update is not None:
                        kf_update(q_now, m)  # marker semantics: safe to refine offset
                    note_marker_seen(q_now)
            else:
                m = locate_by_diff(rig.read(), background, pred, scfg.track_roi,
                                   scfg.bg_thresh, scfg.obj_min_area, scfg.track_max_jump)
            if m is not None:
                c[:] = m - fused.track_px(q_now)
                return m
            return pred

        def track_step(dq):
            nonlocal s
            s = _carry_meas()

        def tr(pred):
            r = _carry_meas()
            debug(dict(s=r))
            return r
    else:
        def track_step(dq):
            nonlocal s
            pred = s + J @ dq
            if getattr(scfg, "use_aruco", False):
                new_s = locate_by_aruco(rig.read(), scfg.aruco_id, predicted_px=pred,
                                       max_dist=scfg.gripper_search_radius)
            else:
                new_s = locate_by_diff(rig.read(), background, pred, scfg.track_roi,
                                       scfg.bg_thresh, scfg.obj_min_area, scfg.track_max_jump)
            s = new_s if new_s is not None else pred

        def tr(pred):
            if getattr(scfg, "use_aruco", False):
                r = locate_by_aruco(rig.read(), scfg.aruco_id, predicted_px=pred,
                                   max_dist=scfg.gripper_search_radius)
                if r is not None:
                    note_marker_seen()
            else:
                r = locate_by_diff(rig.read(), background, pred, scfg.track_roi,
                                   scfg.bg_thresh, scfg.obj_min_area, scfg.track_max_jump)
            debug(dict(s=r))
            return r

    debug(dict(msg=f"[episode] Lifting to z={scfg.lift_z:.3f}m..."))
    nominal_z_to(rig, model, scfg.lift_z, scfg, after_step=track_step)
    if not rig.gripper_contact():
        return fail("object dropped during lift")

    debug(dict(target=pad_px))  # carrying: the goal is the pad now
    debug(dict(msg="[episode] Transporting to drop target..."))
    ok, J, s = servo_to(rig, scfg, J, s, pad_px, tr, scfg.tol_coarse_px,
                        shape_dq=z_hold(scfg.lift_z), jac_fn=jac_fn,
                        null_fn=null_fn, debug=debug)
    if not ok:
        return fail("transport servo failed")
    # lower over the pad and re-servo: kills the remaining parallax offset
    debug(dict(msg=f"[episode] Lowering to z={scfg.release_z:.3f}m and re-servoing..."))
    nominal_z_to(rig, model, scfg.release_z, scfg, after_step=track_step)
    ok, J, s = servo_to(rig, scfg, J, s, pad_px, tr, scfg.tol_coarse_px,
                        shape_dq=z_hold(scfg.release_z), jac_fn=jac_fn,
                        null_fn=null_fn, debug=debug)
    debug(dict(msg="[episode] Releasing and homing..."))
    rig.set_gripper(1.0)
    nominal_z_to(rig, model, scfg.lift_z, scfg)
    q = rig.get_q()
    for a in np.linspace(0.2, 1.0, 5):
        rig.set_q(q + a * (home_q - q))
    return dict(ok=True, reason="released over pad", J=J, tracker=fused,
                fused_rms=(fused.rms if fused is not None else None))

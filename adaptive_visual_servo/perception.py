"""Vision primitives: diff/blob analysis, background-subtraction object
detection (any color/shape), HSV selection, gripper-blink EE localization,
and a small template tracker for the carry phase.

Everything works on plain BGR frames — same code for sim and real camera.
"""

from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass
class Blob:
    center: np.ndarray  # (u, v)
    area: int
    bbox: tuple         # x, y, w, h
    color: tuple        # mean BGR


def diff_mask(a, b, thresh):
    d = cv2.absdiff(a, b).max(axis=2)
    d = cv2.GaussianBlur(d, (5, 5), 0)
    return (d > thresh).astype(np.uint8)


def find_blobs(mask, min_area, frame=None):
    """Connected components >= min_area, largest first."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        color = (0, 0, 0)
        if frame is not None:
            color = cv2.mean(frame, mask=(labels == i).astype(np.uint8))[:3]
        out.append(Blob(cents[i].astype(float), area, tuple(stats[i, :4]), color))
    out.sort(key=lambda b: -b.area)
    return out


def detect_objects(frame, background, thresh=28, min_area=80, exclude=(),
                   exclude_r=30, max_area=None):
    """Any new blob vs. the empty-workspace photo = an object (color-agnostic).

    `exclude`: pixel points (e.g. current EE, the pad) whose vicinity is ignored.
    `max_area` filters out arm-sized blobs when detecting with the arm in view.
    """
    blobs = find_blobs(diff_mask(frame, background, thresh), min_area, frame)
    return [b for b in blobs
            if (max_area is None or b.area <= max_area)
            and all(np.linalg.norm(b.center - np.asarray(p)) > exclude_r
                    for p in exclude)]


def _hue_dist(h1, h2):
    d = abs(int(h1) - int(h2)) % 180
    return min(d, 180 - d)


def pick_by_hue(blobs, hue, tol=14, min_sat=60):
    """Select the detected object matching a requested color, or None."""
    best, best_d = None, tol + 1
    for b in blobs:
        px = np.uint8([[b.color]])
        h, s, _ = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
        d = _hue_dist(h, hue)
        if s >= min_sat and d < best_d:
            best, best_d = b, d
    return best


def detect_pad(frame, hue, tol=14, min_area=300):
    """The drop pad is ours, so its color is known: plain HSV mask, largest blob."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo, hi = int(hue) - tol, int(hue) + tol
    if lo < 0:
        mask = cv2.inRange(hsv, (0, 80, 60), (hi, 255, 255)) | \
               cv2.inRange(hsv, (180 + lo, 80, 60), (179, 255, 255))
    elif hi > 179:
        mask = cv2.inRange(hsv, (lo, 80, 60), (179, 255, 255)) | \
               cv2.inRange(hsv, (0, 80, 60), (hi - 180, 255, 255))
    else:
        mask = cv2.inRange(hsv, (lo, 80, 60), (hi, 255, 255))
    blobs = find_blobs((mask > 0).astype(np.uint8), min_area)
    return blobs[0].center if blobs else None


def blink_locate(rig, scfg):
    """Markerless EE localization: wiggle only the gripper fingers between two
    frames — the diff blob is exactly the fingers. Returns (u, v) or None.
    Never call while holding an object."""
    g_open, g_mid = scfg.blink_dg
    rig.set_gripper(g_open)
    a = rig.read()
    rig.set_gripper(g_mid)
    b = rig.read()
    rig.set_gripper(g_open)
    blobs = find_blobs(diff_mask(a, b, scfg.diff_thresh), scfg.min_blob)
    if not blobs:
        return None
    # each finger sweep is its own component: area-weighted mean = gripper center
    weights = np.array([b.area for b in blobs], float)
    centers = np.stack([b.center for b in blobs])
    return (weights @ centers) / weights.sum()


def _blink_pair(rig, scfg, pair):
    return blink_locate(rig, replace(scfg, blink_dg=pair))


def _perp(v):
    return np.array([-v[1], v[0]])


# safe blink pairs: gripper never goes below 0.4 (no grasp trigger, no chance
# of striking an object under the jaws); the near-closure pair is FREE AIR ONLY
BLINK_HI = (1.0, 0.8)      # jaw position ~ g=0.9
BLINK_LO = (0.6, 0.4)      # jaw position ~ g=0.5
BLINK_CLOSURE = (0.12, 0.0)


def calibrate_grasp_frame(rig, scfg):
    """One-time (per episode) self-calibration of where the jaws actually
    close, for single-moving-jaw grippers like the SO-101.

    The plain blink centroid tracks the MOVING jaw, which is offset from the
    real grasp point (where the jaws meet) by up to the jaw gap -- measured
    25-48px of aim error on the real URDF mesh. Fix: in free air, blink once
    near closure (measuring the true grasp point directly), and express its
    offset from the safe mid-range blink in the jaw-sweep direction frame
    (component along the sweep d and along d-perpendicular). That 2-component
    offset transfers across arm poses (rotation/scale ride along with d):
    measured 2-10px residual at other poses vs ~25-48px uncorrected.

    Call ONLY with the hand in free air (e.g. right after babbling). Returns
    (a, b), or (0, 0) for symmetric grippers (both jaws move -> no offset),
    or None if the blinks failed."""
    c1 = _blink_pair(rig, scfg, BLINK_HI)
    c2 = _blink_pair(rig, scfg, BLINK_LO)
    if c1 is None or c2 is None:
        rig.set_gripper(1.0)
        return None
    d = c2 - c1
    if np.linalg.norm(d) < 3.0:
        rig.set_gripper(1.0)
        return (0.0, 0.0)  # symmetric gripper: blink centroid is already the grasp point
    closure = _blink_pair(rig, scfg, BLINK_CLOSURE)
    rig.set_gripper(1.0)  # blink restores its pair's first value; travel open
    if closure is None:
        return None
    M = np.stack([d, _perp(d)], axis=1)
    a, b = np.linalg.solve(M, closure - c2)
    return (float(a), float(b))


def locate_grasp_point(rig, scfg, ab):
    """Grasp-point (jaw-closure) locator using only safe open-range blinks
    plus the calibrated sweep-frame offset from calibrate_grasp_frame."""
    c1 = _blink_pair(rig, scfg, BLINK_HI)
    c2 = _blink_pair(rig, scfg, BLINK_LO)
    rig.set_gripper(1.0)
    if c1 is None or c2 is None:
        return None
    d = c2 - c1
    if np.linalg.norm(d) < 3.0:
        return c2
    return c2 + ab[0] * d + ab[1] * _perp(d)


def locate_by_diff(frame, background, center, roi=80, thresh=28, min_area=30, max_jump=25.0):
    """Background-subtraction localization of the carried gripper+object in a
    local ROI around `center` (the kinematic prediction s + J@dq) -- for the
    carry phase, when the gripper can't blink to re-anchor.

    Unlike template matching, this is invariant to the rotation/viewing-angle
    appearance changes a long transport puts the carried object through: it
    only asks "did this pixel change from the known-empty background", never
    what the object looks like, so there's no template to drift or go stale.
    Returns None (not a stale guess) when nothing plausible is found, so the
    caller's own blind-step handling (servo_to) decides how much of that to
    tolerate. `max_jump` rejects a distractor blob (e.g. another not-yet-picked
    object) that happens to land inside the ROI -- a rigidly-held object's
    per-step motion is bounded by the joint-step clamp, so the true target is
    always the blob nearest the prediction, not just any new blob.
    """
    h, w = frame.shape[:2]
    r = roi // 2
    x0 = int(np.clip(center[0] - r, 0, w - roi))
    y0 = int(np.clip(center[1] - r, 0, h - roi))
    blobs = find_blobs(diff_mask(frame[y0:y0 + roi, x0:x0 + roi],
                                 background[y0:y0 + roi, x0:x0 + roi], thresh), min_area)
    if not blobs:
        return None
    c = np.array([roi / 2.0, roi / 2.0])
    best = min(blobs, key=lambda b: np.linalg.norm(b.center - c))
    if np.linalg.norm(best.center - c) > max_jump:
        return None
    return best.center + np.array([x0, y0])

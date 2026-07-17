"""Vision primitives: diff/blob analysis, background-subtraction object
detection (any color/shape), HSV selection, gripper-blink EE localization,
and a small template tracker for the carry phase.

Everything works on plain BGR frames — same code for sim and real camera.
"""

import time
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
    # cancel global exposure/white-balance shifts between the two frames:
    # real webcams auto-adjust between captures, which puts a uniform pedestal
    # under the whole diff -- the median estimates that pedestal (real motion
    # covers few pixels, so it barely affects the median) and removing it
    # keeps thresholding about MOTION, not lighting
    d = cv2.subtract(d, int(np.median(d)))
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


def blink_measure(rig, scfg, predicted_px=None, max_dist=120.0):
    """Markerless EE localization by synchronous detection: the gripper is
    the only thing in the scene that changes exactly WHEN commanded, in BOTH
    directions, at the SAME place.

    A single before/after diff (the naive version) fails on real cameras:
    compression shimmer, specular surfaces, and passers-by produce dozens of
    diff blobs per frame pair and the centroid lands anywhere (observed live:
    13-33 blobs, position jumping across the whole frame). Instead:

      1. null pair (a1, a2): two frames with NO command between them --
         anything that differs is in-place scene flicker (monitor, specular
         shimmer), dilated into an exclusion mask
      2. two INDEPENDENT diff pairs that share no frame: (a2 open vs b1 mid)
         AND (b2 mid vs c open). The fingers moved away and back, so they
         mark the same pixels in both pairs. A one-frame transient (codec
         speckle) can't repeat across two disjoint pairs -- note a single
         mid-frame shared by both diffs WOULD leak its own speckles through
         the AND, which is why b is captured twice. A continuously moving
         person marks leading/trailing edges at different places in each
         pair, so their AND is empty too.

    Returns (px | None, info dict) -- info carries diagnostics for the UI's
    Test Blink. Never call while holding an object."""
    g_open, g_mid = scfg.blink_dg

    def read_pair():
        f1 = rig.read()
        if scfg.blink_null_gap_s > 0:  # real cameras: avoid the same buffered frame
            time.sleep(scfg.blink_null_gap_s)
        return f1, rig.read()

    rig.set_gripper(g_open)
    a1, a2 = read_pair()
    rig.set_gripper(g_mid)
    b1, b2 = read_pair()
    rig.set_gripper(g_open)
    c = rig.read()

    m_null = diff_mask(a1, a2, scfg.diff_thresh)
    m = cv2.bitwise_and(diff_mask(a2, b1, scfg.diff_thresh),
                        diff_mask(b2, c, scfg.diff_thresh))
    m[cv2.dilate(m_null, np.ones((9, 9), np.uint8)) > 0] = 0

    blobs = find_blobs(m, scfg.min_blob)
    kept = [x for x in blobs if x.area <= scfg.blink_max_blob]

    # Prediction Gating: filter out blobs far from predicted position (e.g. hands)
    if predicted_px is not None:
        kept = [x for x in kept if np.linalg.norm(x.center - np.asarray(predicted_px)) <= max_dist]

    info = dict(n_raw=len(blobs), n_kept=len(kept),
                areas=[x.area for x in kept],
                noise_px=int(m_null.sum()))
    if not kept:
        return None, info
    # each finger sweep is its own component: area-weighted mean = gripper center
    weights = np.array([x.area for x in kept], float)
    centers = np.stack([x.center for x in kept])
    return (weights @ centers) / weights.sum(), info


def blink_locate(rig, scfg, predicted_px=None, max_dist=120.0):
    """blink_measure without the diagnostics -- the control pipeline's view."""
    px, _ = blink_measure(rig, scfg, predicted_px=predicted_px, max_dist=max_dist)
    return px


def filter_by_background(det_blobs, frame, background, thresh=28, min_area=80,
                         match_r=45.0):
    """Keep only learned-detector blobs corroborated by background
    subtraction: a pickable object was placed AFTER the empty-workspace photo,
    so it must show up in the bg-diff too. Detections with no bg-diff blob
    nearby are things that were already in the photo -- most importantly the
    robot arm itself, which a COCO detector will happily box (observed:
    nanodet's largest detection was the arm, and the servo chased it)."""
    evidence = find_blobs(diff_mask(frame, background, thresh), min_area)
    return [b for b in det_blobs
            if any(np.linalg.norm(b.center - e.center) < match_r for e in evidence)]


def _blink_pair(rig, scfg, pair, predicted_px=None, max_dist=120.0):
    return blink_locate(rig, replace(scfg, blink_dg=pair), predicted_px=predicted_px, max_dist=max_dist)


def _perp(v):
    return np.array([-v[1], v[0]])


# safe blink pairs: gripper never goes below 0.4 (no grasp trigger, no chance
# of striking an object under the jaws); the near-closure pair is FREE AIR ONLY
BLINK_HI = (1.0, 0.8)      # jaw position ~ g=0.9
BLINK_LO = (0.6, 0.4)      # jaw position ~ g=0.5
BLINK_CLOSURE = (0.12, 0.0)


def locate_by_marker(frame, hue, tol=14, min_sat=60, min_val=50, min_area=15, max_area=5000,
                     predicted_px=None, max_dist=120.0):
    """Detects gripper position by color marker tip.
    Passive and instantaneous; no wiggling needed."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo, hi = int(hue) - tol, int(hue) + tol
    if lo < 0:
        mask = cv2.inRange(hsv, (0, min_sat, min_val), (hi, 255, 255)) | \
               cv2.inRange(hsv, (180 + lo, min_sat, min_val), (179, 255, 255))
    elif hi > 179:
        mask = cv2.inRange(hsv, (lo, min_sat, min_val), (179, 255, 255)) | \
               cv2.inRange(hsv, (0, min_sat, min_val), (hi - 180, 255, 255))
    else:
        mask = cv2.inRange(hsv, (lo, min_sat, min_val), (hi, 255, 255))

    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))

    blobs = find_blobs(mask, min_area)
    blobs = [b for b in blobs if b.area <= max_area]

    if predicted_px is not None:
        blobs = [b for b in blobs if np.linalg.norm(b.center - np.asarray(predicted_px)) <= max_dist]

    return blobs[0].center if blobs else None


def calibrate_grasp_frame(rig, scfg, predicted_px=None):
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
    if getattr(scfg, "gripper_marker_hue", None) is not None or getattr(scfg, "use_aruco", False):
        return (0.0, 0.0)  # Passive trackers don't wiggle; no jaw-offset calibration needed.

    c1 = _blink_pair(rig, scfg, BLINK_HI, predicted_px=predicted_px, max_dist=scfg.gripper_search_radius)
    c2 = _blink_pair(rig, scfg, BLINK_LO, predicted_px=predicted_px, max_dist=scfg.gripper_search_radius)
    if c1 is None or c2 is None:
        rig.set_gripper(1.0)
        return None
    d = c2 - c1
    if np.linalg.norm(d) < 3.0:
        rig.set_gripper(1.0)
        return (0.0, 0.0)  # symmetric gripper: blink centroid is already the grasp point
    closure = _blink_pair(rig, scfg, BLINK_CLOSURE, predicted_px=predicted_px, max_dist=scfg.gripper_search_radius)
    rig.set_gripper(1.0)  # blink restores its pair's first value; travel open
    if closure is None:
        return None
    M = np.stack([d, _perp(d)], axis=1)
    a, b = np.linalg.solve(M, closure - c2)
    return (float(a), float(b))


def locate_grasp_point(rig, scfg, ab, predicted_px=None):
    """Grasp-point (jaw-closure) locator using only safe open-range blinks
    plus the calibrated sweep-frame offset from calibrate_grasp_frame."""
    if getattr(scfg, "use_aruco", False):
        return locate_by_aruco(rig.read(), scfg.aruco_id, predicted_px=predicted_px,
                               max_dist=scfg.gripper_search_radius)

    if getattr(scfg, "gripper_marker_hue", None) is not None:
        # Marker mode: passive and instantaneous color tracking directly
        return locate_by_marker(rig.read(), scfg.gripper_marker_hue, scfg.hue_tol,
                                scfg.gripper_marker_sat_min, scfg.gripper_marker_val_min,
                                scfg.gripper_marker_area_min, scfg.gripper_marker_area_max,
                                predicted_px=predicted_px, max_dist=scfg.gripper_search_radius)

    c1 = _blink_pair(rig, scfg, BLINK_HI, predicted_px=predicted_px, max_dist=scfg.gripper_search_radius)
    c2 = _blink_pair(rig, scfg, BLINK_LO, predicted_px=predicted_px, max_dist=scfg.gripper_search_radius)
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


_DETECTOR_CACHE = {}


def get_cached_detector(dictionary_id):
    from aruco_tracker import ArucoDetector
    if dictionary_id not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[dictionary_id] = ArucoDetector(dictionary_id)
    return _DETECTOR_CACHE[dictionary_id]


def locate_by_aruco(frame, marker_id=0, dictionary_id=None, predicted_px=None, max_dist=120.0):
    """Detects gripper position by AruCo marker using cached high-sensitivity detector."""
    if dictionary_id is None:
        try:
            dictionary_id = cv2.aruco.DICT_4X4_50
        except AttributeError:
            return None

    try:
        detector = get_cached_detector(dictionary_id)
        corners, ids, _ = detector.detect(frame)

        if ids is not None:
            for idx, m_id in enumerate(ids.flatten()):
                if m_id == marker_id:
                    c = corners[idx][0]
                    center = np.mean(c, axis=0)
                    # search-radius gate vs the motion-model prediction: a
                    # detection implausibly far away is a misdetection (was
                    # silently ignored before -- the params existed unused)
                    if (predicted_px is not None and
                            np.linalg.norm(center - np.asarray(predicted_px)) > max_dist):
                        return None
                    return center
    except Exception:
        pass
    return None

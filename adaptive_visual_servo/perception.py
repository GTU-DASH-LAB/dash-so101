"""Vision primitives: diff/blob analysis, background-subtraction object
detection (any color/shape), HSV selection, gripper-blink EE localization,
and a small template tracker for the carry phase.

Everything works on plain BGR frames — same code for sim and real camera.
"""

from dataclasses import dataclass

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


def detect_objects(frame, background, thresh=28, min_area=80, exclude=(), exclude_r=30):
    """Any new blob vs. the empty-workspace photo = an object (color-agnostic).

    `exclude`: pixel points (e.g. current EE) whose vicinity is ignored.
    """
    blobs = find_blobs(diff_mask(frame, background, thresh), min_area, frame)
    return [b for b in blobs
            if all(np.linalg.norm(b.center - np.asarray(p)) > exclude_r for p in exclude)]


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


class PatchTracker:
    """Grayscale template tracking in a local ROI, for the carry phase (when the
    gripper can't blink). Falls back to the caller's motion prediction when the
    match is weak; template is refreshed after every good match."""

    def __init__(self, frame, center, patch=42, roi=140, match_min=0.35):
        self.patch, self.roi, self.match_min = patch, roi, match_min
        self.pos = np.asarray(center, float)
        self.tmpl = self._crop(frame, self.pos, patch)

    @staticmethod
    def _crop(frame, center, size):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        x0 = int(np.clip(center[0] - size / 2, 0, w - size))
        y0 = int(np.clip(center[1] - size / 2, 0, h - size))
        return g[y0:y0 + size, x0:x0 + size]

    def update(self, frame, predicted=None):
        base = np.asarray(predicted, float) if predicted is not None else self.pos
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        x0 = int(np.clip(base[0] - self.roi / 2, 0, w - self.roi))
        y0 = int(np.clip(base[1] - self.roi / 2, 0, h - self.roi))
        res = cv2.matchTemplate(g[y0:y0 + self.roi, x0:x0 + self.roi],
                                self.tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score >= self.match_min:
            self.pos = np.array([x0 + loc[0] + self.patch / 2,
                                 y0 + loc[1] + self.patch / 2], float)
            self.tmpl = self._crop(frame, self.pos, self.patch)
        else:
            self.pos = base  # trust the motion model this step
        return self.pos.copy(), float(score)

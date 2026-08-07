"""Validates the hand-written NanoDet-Plus DFL decode against a synthetic
head output with a planted detection (ground truth we control), since the
sim's abstract painted shapes won't trigger real COCO detections the way a
photo of an actual ball/bottle/cup would -- this checks the decode math is
correct independent of what's in front of the camera."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanodet_detector import (_CENTERS, COCO_CLASSES, REG_MAX, NanodetDetector,
                              decode_nanodet_output)


def _plant_box(output, row, cls_id, l_bin, t_bin, r_bin, b_bin, num_classes=80):
    """Write scores into `row` that decode to an exact box, given ltrb as
    integer DFL bins (0..REG_MAX) -- a one-hot bin decodes to exactly that
    integer, so the recovered box is exact, no rounding ambiguity.
    Classification column is a probability (this ONNX export already applies
    sigmoid internally), not a raw logit."""
    output[row, cls_id] = 0.95
    for k, b in enumerate((l_bin, t_bin, r_bin, b_bin)):
        logits = np.full(REG_MAX + 1, -10.0)  # near one-hot at the exact DFL bin
        logits[b] = 10.0
        base = num_classes + k * (REG_MAX + 1)
        output[row, base:base + REG_MAX + 1] = logits


def test_decode_recovers_planted_box():
    num_classes = 80
    output = np.zeros((_CENTERS.shape[0], num_classes + 4 * (REG_MAX + 1)), np.float32)
    output[:, :num_classes] = 0.001  # every row starts "no class" (near-zero probability)
    # pick a mid-grid row on the stride-16 level so the box fits comfortably
    row = int(np.where(_CENTERS[:, 2] == 16)[0][50])
    cx, cy, stride = _CENTERS[row]
    l_bin, t_bin, r_bin, b_bin = 3, 2, 2, 3  # exact DFL bins, no rounding ambiguity
    true_box = (cx - l_bin * stride, cy - t_bin * stride,
               cx + r_bin * stride, cy + b_bin * stride)
    _plant_box(output, row, 32, l_bin, t_bin, r_bin, b_bin)  # 32 = "sports ball"

    dets = decode_nanodet_output(output, score_thresh=0.3)
    assert len(dets) == 1, f"expected exactly 1 detection, got {len(dets)}"
    x0, y0, x1, y1, score, cid = dets[0]
    assert cid == 32 and score > 0.9
    for got, want in zip((x0, y0, x1, y1), true_box):
        assert abs(got - want) < 2.0, f"decoded box off: {(x0,y0,x1,y1)} vs {true_box}"


def test_decode_empty_below_threshold():
    num_classes = 80
    output = np.full((_CENTERS.shape[0], num_classes + 4 * (REG_MAX + 1)), 0.001, np.float32)
    assert decode_nanodet_output(output, score_thresh=0.3) == []


def test_class_allowlist_filters_blobs(monkeypatch):
    """NanodetDetector.__call__ drops detections outside class_names without
    needing a real photo -- stub the ONNX session + decode to isolate the
    filtering logic."""
    det = NanodetDetector.__new__(NanodetDetector)  # skip ONNX session load
    det.allowed = {COCO_CLASSES.index("bottle")}
    det.score_thresh, det.nms_thresh = 0.3, 0.5

    class FakeSession:
        def run(self, _out, feed):
            num_classes = 80
            out = np.zeros((1, _CENTERS.shape[0], num_classes + 4 * (REG_MAX + 1)), np.float32)
            out[0, :, :num_classes] = 0.001
            row_bottle = int(np.where(_CENTERS[:, 2] == 16)[0][10])
            row_cup = int(np.where(_CENTERS[:, 2] == 16)[0][60])
            _plant_box(out[0], row_bottle, COCO_CLASSES.index("bottle"), 2, 2, 2, 2)
            _plant_box(out[0], row_cup, COCO_CLASSES.index("cup"), 2, 2, 2, 2)
            return (out,)

    det.session = FakeSession()
    frame = np.zeros((320, 320, 3), np.uint8)
    blobs = det(frame)
    assert len(blobs) == 1, f"expected only the allowlisted class, got {len(blobs)}"


if __name__ == "__main__":
    for fn in [test_decode_recovers_planted_box, test_decode_empty_below_threshold]:
        fn()
        print(f"ok {fn.__name__}")
    test_class_allowlist_filters_blobs(None)
    print("ok test_class_allowlist_filters_blobs")
    print("ALL OK")

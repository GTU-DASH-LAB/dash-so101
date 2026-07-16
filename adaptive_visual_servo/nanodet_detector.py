"""NanoDet-Plus object detector (github.com/RangiLyu/nanodet), ONNX inference
only -- no PyTorch/mmcv training deps, just onnxruntime + the ~5MB exported
checkpoint (assets/nanodet/nanodet-plus-m_320.onnx).

Drop-in replacement for perception.detect_objects's bg-sub detector via
control.run_episode's `detector` plug point: `detector(frame) -> [Blob]`.
Real, COCO-trained recognition (best for real-camera runs with everyday
objects -- balls, bottles, cups, fruit) rather than the sim's abstract
painted shapes, which don't resemble any COCO class closely enough to detect
reliably -- background subtraction stays the sim's default detector.
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

from perception import Blob

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "nanodet")
MODEL_PATH = os.path.join(ASSETS_DIR, "nanodet-plus-m_320.onnx")

INPUT_SIZE = 320
STRIDES = (8, 16, 32, 64)
REG_MAX = 7
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)  # BGR
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# sensible tabletop pick-and-place default -- override with class_names=None for all 80
TABLETOP_CLASSES = ("sports ball", "bottle", "cup", "apple", "banana", "orange",
                    "bowl", "cell phone", "scissors", "remote")


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _grid_centers():
    """(N,3) rows of (cx, cy, stride) in 320x320 model-input pixels, in the
    same stride-major, row-major order the exported head flattens outputs."""
    rows = []
    for s in STRIDES:
        g = INPUT_SIZE // s
        ys, xs = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        cx = (xs.ravel() + 0.5) * s
        cy = (ys.ravel() + 0.5) * s
        rows.append(np.stack([cx, cy, np.full(cx.shape, s, dtype=np.float32)], axis=1))
    return np.concatenate(rows, axis=0)


_CENTERS = _grid_centers()


def decode_nanodet_output(output, num_classes=80, score_thresh=0.35, nms_thresh=0.5):
    """output: (N, num_classes + 4*(REG_MAX+1)) raw head tensor -> list of
    (x0, y0, x1, y1, score, class_id) in 320x320 model-input pixel space."""
    assert output.shape[0] == _CENTERS.shape[0], "grid/output size mismatch"
    scores = output[:, :num_classes]  # this export already applies sigmoid internally
    cls_id = scores.argmax(axis=1)
    cls_score = scores[np.arange(len(scores)), cls_id]
    keep = cls_score > score_thresh
    if not np.any(keep):
        return []

    dist = output[keep, num_classes:].reshape(-1, 4, REG_MAX + 1)
    dist = (_softmax(dist, axis=-1) * np.arange(REG_MAX + 1)).sum(-1)  # (M,4) in stride units
    centers = _CENTERS[keep]
    dist = dist * centers[:, 2:3]
    x0 = centers[:, 0] - dist[:, 0]
    y0 = centers[:, 1] - dist[:, 1]
    x1 = centers[:, 0] + dist[:, 2]
    y1 = centers[:, 1] + dist[:, 3]
    boxes = np.stack([x0, y0, x1 - x0, y1 - y0], axis=1)  # cv2.NMS wants xywh

    idx = cv2.dnn.NMSBoxes(boxes.tolist(), cls_score[keep].tolist(), score_thresh, nms_thresh)
    idx = np.asarray(idx).reshape(-1)
    out = []
    for i in idx:
        x, y, w, h = boxes[i]
        out.append((x, y, x + w, y + h, float(cls_score[keep][i]), int(cls_id[keep][i])))
    return out


class NanodetDetector:
    """Callable: frame(BGR) -> list[Blob], matching control.run_episode's
    `detector` plug point."""

    def __init__(self, class_names=TABLETOP_CLASSES, score_thresh=0.35, nms_thresh=0.5):
        so = ort.SessionOptions()
        so.log_severity_level = 3  # ERROR only -- this export's initializers-as-
                                    # graph-inputs quirk otherwise logs a WARNING
                                    # per weight tensor on every session load
        self.session = ort.InferenceSession(MODEL_PATH, sess_options=so,
                                            providers=["CPUExecutionProvider"])
        self.allowed = (None if class_names is None
                        else {COCO_CLASSES.index(c) for c in class_names})
        self.score_thresh, self.nms_thresh = score_thresh, nms_thresh

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        rh, rw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame, (rw, rh))
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
        canvas[:rh, :rw] = resized
        x = ((canvas.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        return x.astype(np.float32), scale

    def _detect(self, frame):
        """Raw detections in original image coords, allowlist-filtered:
        [(x0, y0, x1, y1, score, class_id), ...], largest box first."""
        x, scale = self._preprocess(frame)
        (output,) = self.session.run(None, {"data": x})
        dets = decode_nanodet_output(output[0], score_thresh=self.score_thresh,
                                     nms_thresh=self.nms_thresh)
        out = []
        for x0, y0, x1, y1, score, cid in dets:
            if self.allowed is not None and cid not in self.allowed:
                continue
            out.append((x0 / scale, y0 / scale, x1 / scale, y1 / scale, score, cid))
        out.sort(key=lambda d: -(d[2] - d[0]) * (d[3] - d[1]))
        return out

    def __call__(self, frame):
        """frame(BGR) -> list[Blob], matching control.run_episode's
        `detector` plug point."""
        h, w = frame.shape[:2]
        blobs = []
        for x0, y0, x1, y1, score, cid in self._detect(frame):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            # mean BGR of the box interior, so pick_by_hue ("pick the red one")
            # works the same as with the bg-sub detector's blobs
            ix0, iy0 = int(np.clip(x0, 0, w - 1)), int(np.clip(y0, 0, h - 1))
            ix1, iy1 = int(np.clip(x1, ix0 + 1, w)), int(np.clip(y1, iy0 + 1, h))
            color = tuple(frame[iy0:iy1, ix0:ix1].reshape(-1, 3).mean(axis=0))
            blobs.append(Blob(np.array([cx, cy]), int((x1 - x0) * (y1 - y0)),
                              (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                              color))
        return blobs

    def detect_labeled(self, frame):
        """Like __call__ but keeps score/class_name -- for anything that
        wants to display what nanodet actually saw (camera_nanodet_tester.py),
        not just the bare Blob the control pipeline consumes."""
        out = []
        for x0, y0, x1, y1, score, cid in self._detect(frame):
            out.append(dict(bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                            center=np.array([(x0 + x1) / 2, (y0 + y1) / 2]),
                            score=score, class_id=cid, class_name=COCO_CLASSES[cid]))
        return out

"""Zero-shot open-vocabulary grounding: free-form object phrase -> image point.

Uses OWLv2 (google/owlv2-base-patch16-ensemble, frozen). It was trained on
web-scale data, so it locates objects the robot policy never saw in any demo —
this is where the stack's zero-shot object generalization comes from.
"""

import numpy as np

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from transformers import pipeline

        _detector = pipeline(
            "zero-shot-object-detection", model="google/owlv2-base-patch16-ensemble"
        )
    return _detector


def locate(image: "np.ndarray | str", phrase: str, threshold: float = 0.1) -> tuple[float, float, float]:
    """Find `phrase` in `image` (RGB array or path). Returns (u, v, score), with
    (u, v) the normalized [0,1] center of the best matching box.

    Raises LookupError if nothing scores above `threshold`.
    """
    from PIL import Image

    pil = Image.open(image).convert("RGB") if isinstance(image, str) else Image.fromarray(image)
    detections = _get_detector()(pil, candidate_labels=[phrase], threshold=threshold)
    if not detections:
        raise LookupError(f"could not find '{phrase}' in the image")
    best = max(detections, key=lambda d: d["score"])
    box = best["box"]
    u = (box["xmin"] + box["xmax"]) / 2 / pil.width
    v = (box["ymin"] + box["ymax"]) / 2 / pil.height
    return u, v, best["score"]


if __name__ == "__main__":
    import sys

    image_path, phrase = sys.argv[1], sys.argv[2]
    u, v, score = locate(image_path, phrase)
    print(f"'{phrase}' -> u={u:.3f} v={v:.3f} (score {score:.3f})")

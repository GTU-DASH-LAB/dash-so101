"""Zero-shot open-vocabulary grounding: free-form object phrase -> image point.

Two pluggable backends, selected with SO_BRAIN_GROUNDER (models download on first
use — run heavy backends on the GX10, not the laptop):

  owlv2  (default) google/owlv2-base-patch16-ensemble, ~150M params, ~600MB.
         Fast, light, good at plain noun phrases ("a red pen").
  nvidia nvidia/LocateAnything-3B (Moon-ViT + Qwen2.5, Parallel Box Decoding).
         3B params. Understands referring expressions ("the pen closest to the
         robot arm"). License: NVIDIA non-commercial research only.
"""

import os
import re

import numpy as np

BACKEND = os.environ.get("SO_BRAIN_GROUNDER", "owlv2")

_owlv2 = None
_nvidia = None


def _to_pil(image: "np.ndarray | str"):
    from PIL import Image

    return Image.open(image).convert("RGB") if isinstance(image, str) else Image.fromarray(image)


def _locate_owlv2(pil, phrase: str, threshold: float) -> tuple[float, float, float]:
    global _owlv2
    if _owlv2 is None:
        import torch
        from transformers import pipeline

        _owlv2 = pipeline(
            "zero-shot-object-detection",
            model="google/owlv2-base-patch16-ensemble",
            device=0 if torch.cuda.is_available() else -1,  # GPU matters when tracking every N frames
        )
    detections = _owlv2(pil, candidate_labels=[phrase], threshold=threshold)
    if not detections:
        raise LookupError(f"could not find '{phrase}' in the image")
    best = max(detections, key=lambda d: d["score"])
    box = best["box"]
    u = (box["xmin"] + box["xmax"]) / 2 / pil.width
    v = (box["ymin"] + box["ymax"]) / 2 / pil.height
    return u, v, best["score"]


def _locate_nvidia(pil, phrase: str, threshold: float) -> tuple[float, float, float]:
    # Per the nvidia/LocateAnything-3B model card. Outputs <box>-tagged coordinates
    # normalized to [0, 1000]. Verify tag parsing against the card's utility method
    # on first GX10 run — the format regex below is tolerant (4 ints = box, 2 = point).
    global _nvidia
    if _nvidia is None:
        import torch
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained("nvidia/LocateAnything-3B", trust_remote_code=True)
        # bf16 needs Ampere+; T4 (sm75, e.g. Kaggle) falls back to fp16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = (
            AutoModel.from_pretrained(
                "nvidia/LocateAnything-3B", torch_dtype=dtype, trust_remote_code=True
            )
            .to("cuda")
            .eval()
        )
        _nvidia = (processor, model)
    processor, model = _nvidia

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": f"Locate all instances matching: {phrase}"},
            ],
        }
    ]
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=256)
    decoded = processor.batch_decode(output, skip_special_tokens=False)[0]

    coords = [
        [int(n) for n in re.findall(r"\d+", m)] for m in re.findall(r"<box>(.*?)</box>", decoded)
    ]
    coords = [c for c in coords if len(c) in (2, 4)]
    if not coords:
        raise LookupError(f"could not find '{phrase}' in the image (raw: {decoded[-200:]!r})")
    c = coords[0]
    u, v = ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2) if len(c) == 4 else (c[0], c[1])
    return u / 1000, v / 1000, 1.0  # ponytail: PBD emits no calibrated score; report 1.0


def locate(image: "np.ndarray | str", phrase: str, threshold: float = 0.1) -> tuple[float, float, float]:
    """Find `phrase` in `image` (RGB array or path). Returns (u, v, score) with
    (u, v) normalized [0,1] image coordinates. Raises LookupError when not found.
    """
    pil = _to_pil(image)
    if BACKEND == "nvidia":
        return _locate_nvidia(pil, phrase, threshold)
    return _locate_owlv2(pil, phrase, threshold)


if __name__ == "__main__":
    import sys

    image_path, phrase = sys.argv[1], sys.argv[2]
    u, v, score = locate(image_path, phrase)
    print(f"[{BACKEND}] '{phrase}' -> u={u:.3f} v={v:.3f} (score {score:.3f})")

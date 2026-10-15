#!/usr/bin/env python3
"""Build the starting checkpoint for fine-tuning: lerobot/VLA-JEPA-SimplerEnv, re-shaped for
THIS robot and dataset. Writes $INIT_POLICY_DIR (config.json edited, weights symlinked).

Why a copy instead of `--policy.path=lerobot/VLA-JEPA-SimplerEnv` straight away:
when fine-tuning, lerobot keeps the pretrained config's input features. SimplerEnv's are a
single `observation.images.image` and NO joint state, so our `observation.state` would reach
the model un-normalized (raw degrees), and the camera would need a rename map. Writing our
own features here fixes both: lerobot then normalizes the state with this dataset's stats.

Changes from the base config:
* features: observation.images.wrist (3x480x640) + observation.state (6) -> action (6)
* action_dim/state_dim 7/8 -> 6/6; the three layers whose shape depends on them
  (action encoder, action decoder, state encoder) are listed in reinit_modules, so they
  start fresh while the rest (Qwen3-VL, DiT action head, world-model predictor) keeps
  its pretrained weights. VLAJEPAPolicy._load_as_safetensor refuses any other mismatch.
* gripper binarization off: Bridge's gripper is open/closed; the SO-101 gripper is a
  continuous joint position.
* chunk_size/n_action_steps for 30 Hz; base model paths point at the local downloads.
"""

import json
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    base = Path(os.environ["BASE_POLICY"]).expanduser()
    out = Path(os.environ["INIT_POLICY_DIR"]).expanduser()
    if not (base / "model.safetensors").is_file():
        print(f"Base checkpoint missing: {base}/model.safetensors (download not finished?)", file=sys.stderr)
        return 1
    info = json.loads((Path(os.environ["DATASET_ROOT"]) / "meta" / "info.json").read_text())
    feats = info["features"]
    img = feats["observation.images.wrist"]["shape"]          # [H, W, C] in the dataset
    n_state, n_action = feats["observation.state"]["shape"][0], feats["action"]["shape"][0]

    cfg = json.loads((base / "config.json").read_text())
    size = int(os.environ["IMAGE_SIZE"])
    cfg.update({
        "input_features": {
            "observation.images.wrist": {"type": "VISUAL", "shape": [img[2], img[0], img[1]]},
            "observation.state": {"type": "STATE", "shape": [n_state]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [n_action]}},
        "action_dim": n_action,
        "state_dim": n_state,
        "reinit_modules": [
            "model.action_model.action_encoder",
            "model.action_model.action_decoder",
            "model.action_model.state_encoder",
        ],
        "binarize_gripper_action": False,
        "pre_snap_gripper_action": False,
        "chunk_size": int(os.environ["CHUNK_SIZE"]),
        "n_action_steps": int(os.environ["N_ACTION_STEPS"]),
        "resize_images_to": [size, size],
        "qwen_model_name": str(Path(os.environ["QWEN_PATH"]).expanduser()),
        "jepa_encoder_name": str(Path(os.environ["JEPA_PATH"]).expanduser()),
        "scheduler_warmup_steps": int(os.environ["WARMUP_STEPS"]),
        "device": "cuda",
        "push_to_hub": False,
        "repo_id": None,
    })

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "model.safetensors").symlink_to(base / "model.safetensors")
    for f in base.iterdir():   # pre/post-processor definitions (their stats get replaced by ours)
        if f.name.startswith("policy_") and f.is_file():
            shutil.copy2(f, out / f.name)
    print(f"Init policy: {out}  (6-joint heads re-initialised; chunk {cfg['chunk_size']}, images {size}x{size})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

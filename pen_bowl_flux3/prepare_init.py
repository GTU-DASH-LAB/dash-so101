#!/usr/bin/env python3
"""Build the fine-tuning starting point from black-forest-labs/flux-3-action-so101 for THIS
dataset: wrist camera only, and joint-state normalization from THIS arm's calibration.

    prepare_init.py --ckpt <downloaded flux-3-action-so101 dir> --out <init dir>
    (DATASET_ROOT, CAMERA_LAYOUT, CANVAS_HW from the environment / config.env)

Changes (everything else, weights included, is the checkpoint's own):
* Cameras: the checkpoint composes scene|wrist side by side (each 256x256 on a 256x512
  canvas). We have only the wrist camera: layout "single", canvas 256x256, so the wrist
  image is resized exactly as in pretraining and fills the whole canvas. Edited in
  config.json AND in the processors (they store camera keys and layout too).
* State quantiles: FLUX 3's processors carry their own q01/q99 normalization (lerobot's
  dataset-stat override does not apply to them). The SO-101 community data behind them
  used a different joint calibration: e.g. shoulder_lift 85..184 deg there vs -104..61
  here, so 3 of our 6 joints would normalize ~5 ranges out of distribution. The STATE
  quantiles are replaced by this dataset's (same q01/q99 definition). The ACTION
  quantiles (per-step command deltas, ours +-1.5-3 deg vs theirs +-3-5 deg, gripper
  absolute) already cover our data and are kept, so the pretrained action head's scale
  is unchanged.
"""

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file

WRIST = "observation.images.wrist"
STATE_FILE = "policy_preprocessor_step_3_flux3_observation_history_normalizer.safetensors"


def our_state_quantiles(root: Path) -> tuple[list[float], list[float]]:
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    st = np.concatenate([np.stack(pq.read_table(f, columns=["observation.state"]).column(0)
                                  .to_numpy(zero_copy_only=False)) for f in files])
    return np.quantile(st, 0.01, axis=0).tolist(), np.quantile(st, 0.99, axis=0).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ckpt, out = Path(a.ckpt).expanduser(), Path(a.out).expanduser()
    root = Path(os.environ["DATASET_ROOT"]).expanduser()
    layout, canvas = os.environ["CAMERA_LAYOUT"], json.loads(os.environ["CANVAS_HW"])
    if not (ckpt / "model.safetensors").is_file():
        print(f"No model.safetensors in {ckpt}", file=sys.stderr)
        return 1
    info = json.loads((root / "meta" / "info.json").read_text())
    h, w, c = info["features"][WRIST]["shape"]
    n_state = info["features"]["observation.state"]["shape"][0]

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in ckpt.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy2(f, out / f.name)
    (out / "model.safetensors").symlink_to(ckpt / "model.safetensors")

    cfg = json.loads((out / "config.json").read_text())
    cfg["input_features"] = {WRIST: {"type": "VISUAL", "shape": [c, h, w]},
                             "observation.state": {"type": "STATE", "shape": [n_state]}}
    cfg.update(camera_layout=layout, camera_keys=[WRIST], canvas_hw=canvas, device="cuda", push_to_hub=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    pre = json.loads((out / "policy_preprocessor.json").read_text())
    for step in pre["steps"]:
        if "camera_keys" in step.get("config", {}):
            step["config"]["camera_keys"] = [WRIST]
        if "camera_layout" in step.get("config", {}):
            step["config"]["camera_layout"] = layout
    (out / "policy_preprocessor.json").write_text(json.dumps(pre, indent=2))

    q01, q99 = our_state_quantiles(root)
    tensors = load_file(str(out / STATE_FILE))
    old = (tensors["state.q01"].tolist(), tensors["state.q99"].tolist())
    tensors["state.q01"] = torch.tensor(q01, dtype=tensors["state.q01"].dtype)
    tensors["state.q99"] = torch.tensor(q99, dtype=tensors["state.q99"].dtype)
    save_file(tensors, str(out / STATE_FILE))
    stats_path = out / "dataset_statistics.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats["state"]["q01"], stats["state"]["q99"] = q01, q99
        stats_path.write_text(json.dumps(stats, indent=2))

    print(f"Init policy: {out}")
    print(f"  cameras: {WRIST} only, layout {layout}, canvas {canvas}")
    print(f"  state q01 {[round(x, 1) for x in old[0]]} -> {[round(x, 1) for x in q01]}")
    print(f"  state q99 {[round(x, 1) for x in old[1]]} -> {[round(x, 1) for x in q99]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

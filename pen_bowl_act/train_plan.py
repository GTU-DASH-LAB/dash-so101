#!/usr/bin/env python3
"""Turn config.env's training settings (EPOCHS, batch, GPUs, validation split) into
exact lerobot-train step counts for the dataset that is actually on disk.

lerobot counts in optimizer STEPS; people think in EPOCHS (one pass over every
training frame). Here, with E = BATCH_SIZE x NUM_GPUS samples per step:

    steps_per_epoch = ceil(train_frames / E)
    total steps     = EPOCHS x steps_per_epoch
    validation      = once per epoch   (eval_steps = steps_per_epoch)
    checkpoints     = every CHECKPOINT_EVERY_EPOCHS epochs

The validation split mirrors lerobot's make_train_eval_datasets() exactly: the LAST
ceil(n_episodes x VALIDATION_SPLIT) demos are held out (one task here).

Writes run/plan.json, which the dashboard reads for epoch/ETA maths.
"""

import json
import math
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq


def main() -> int:
    root = Path(os.environ["DATASET_ROOT"])
    info_file = root / "meta" / "info.json"
    if not info_file.is_file():
        print(f"No dataset at {root}. Run ./1b_publish_dataset.sh first.", file=sys.stderr)
        return 1
    info = json.loads(info_file.read_text())
    rows = []
    for p in sorted(root.glob("meta/episodes/**/*.parquet")):
        rows += pq.read_table(p, columns=["episode_index", "length"]).to_pylist()
    lengths = [r["length"] for r in sorted(rows, key=lambda r: r["episode_index"])]
    n_eps = len(lengths)

    split = float(os.environ["VALIDATION_SPLIT"])
    n_val = math.ceil(n_eps * split) if split > 0 else 0
    if n_val >= n_eps:
        print("VALIDATION_SPLIT leaves no training demos.", file=sys.stderr)
        return 1
    train_frames = sum(lengths[: n_eps - n_val])
    val_frames = sum(lengths[n_eps - n_val:])

    gpus, bs = int(os.environ["NUM_GPUS"]), int(os.environ["BATCH_SIZE"])
    eff = gpus * bs
    epochs = int(os.environ["EPOCHS"])
    spe = math.ceil(train_frames / eff)
    total = epochs * spe
    ckpt_every = int(os.environ["CHECKPOINT_EVERY_EPOCHS"])
    plan = {
        "dataset_root": str(root),
        "episodes": n_eps,
        "train_episodes": n_eps - n_val,
        "val_episodes": n_val,
        "train_frames": train_frames,
        "val_frames": val_frames,
        "fps": info["fps"],
        "num_gpus": gpus,
        "batch_size_per_gpu": bs,
        "effective_batch": eff,
        "epochs": epochs,
        "steps_per_epoch": spe,
        "total_steps": total,
        "eval_steps": spe if n_val else 0,
        "save_freq": ckpt_every * spe,
        "log_freq": max(5, spe // 20),
        "lr": float(os.environ["LR"]),
        "lr_backbone": float(os.environ["LR_BACKBONE"]),
        "mixed_precision": os.environ["MIXED_PRECISION"],
        "augment": os.environ["AUGMENT"] == "true",
        "throughput_hint": float(os.environ.get("THROUGHPUT_HINT", "0") or 0),
    }
    out = Path(os.environ["RUN_DIR"]) / "plan.json"
    out.write_text(json.dumps(plan, indent=2))

    est = ""
    if plan["throughput_hint"] > 0:
        secs = epochs * train_frames / plan["throughput_hint"]
        est = f"  ~{secs / 60:.0f} min at the measured ~{plan['throughput_hint']:.0f} samples/s"
    print(f"Dataset:     {n_eps} demos, {info['total_frames']:,} frames ({info['total_frames'] / info['fps'] / 60:.1f} min of demos)")
    print(f"Split:       {n_eps - n_val} train demos ({train_frames:,} frames) / {n_val} validation demos ({val_frames:,} frames)")
    print(f"Batch:       {bs} per GPU x {gpus} GPU(s) = {eff} samples per step")
    print(f"Plan:        {epochs} epochs x {spe:,} steps/epoch = {total:,} steps{est}")
    print(f"Validation:  every epoch    Checkpoints: every {ckpt_every} epochs ({plan['save_freq']:,} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Stage one trained checkpoint (+ model card + loss curve) and push it to the HF Hub.

Called by ./2b_publish_model.sh, which resolves the checkpoint name and checks the login.

    publish_model.py --checkpoint 027300 [--dry-run]

The checkpoint becomes the repo's main branch and is also tagged `step-027300`, so
pushing another checkpoint later never loses this one.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import train_dashboard as td

HERE = Path(__file__).resolve().parent


def model_card(cfg: dict, ck: dict, d: dict, info: dict, pol: dict, has_png: bool) -> str:
    repo, ds_repo, plan = cfg["MODEL_REPO"], cfg["DATASET_REPO"], d["plan"] or {}
    feats = {**pol["input_features"], **pol["output_features"]}
    rows = "\n".join(f"| `{k}` | {' x '.join(map(str, v['shape']))} |" for k, v in feats.items())
    ck_rows = "\n".join(
        f"| `{c['name']}` | {c['epoch']:.0f} | {c['val']:.4f}{' (lowest)' if c.get('best') else ''} |"
        for c in d["checkpoints"] if c["val"] is not None)
    starts = [l for l in d["watchdog"] if "STARTING" in l]
    took = ""
    try:
        fin = [l for l in d["watchdog"] if "FINISHED" in l]
        if starts and fin:
            took = f", {(td.ts(fin[-1][:19]) - td.ts(starts[0][:19])) / 60:.0f} min wall-clock"
    except ValueError:
        pass
    cams = json.loads(cfg["CAMERAS"])
    cams["wrist"]["index_or_path"] = "/dev/video0"  # the user's own camera path goes here
    val = f"{ck['val']:.4f}" if ck["val"] is not None else "n/a"
    return f"""---
library_name: lerobot
pipeline_tag: robotics
license: apache-2.0
model_name: act
datasets: {ds_repo}
tags:
- robotics
- lerobot
- act
- so101
- wrist-camera
- pick-and-place
---

# ACT: pick up the pen and put it in the bowl (SO-101, wrist camera only)

An [ACT](https://huggingface.co/papers/2304.13705) policy trained with
[LeRobot](https://github.com/huggingface/lerobot) on {info['total_episodes']} teleoperated
demonstrations of **"{cfg['TASK']}"** on an SO-101 arm. The policy sees **only** the
gripper-mounted (wrist) camera and the joint positions.

| Feature | Shape |
|---|---|
{rows}

Predicts chunks of {pol['chunk_size']} actions at {info['fps']} fps (absolute joint targets, degrees).

## This checkpoint

Step **{ck['step']:,}** = epoch **{ck['epoch']:.0f}** of {plan.get('epochs', '?')}, validation loss **{val}**.
Older pushes stay available as git tags (`step-NNNNNN`).

## Training

- Dataset: [{ds_repo}](https://huggingface.co/datasets/{ds_repo}), {info['total_episodes']} demos /
  {info['total_frames']:,} frames. {plan.get('train_episodes', '?')} demos for training,
  the last {plan.get('val_episodes', '?')} held out for validation.
- {plan.get('epochs', '?')} epochs = {plan.get('total_steps', 0):,} steps, batch
  {plan.get('effective_batch', '?')} ({plan.get('batch_size_per_gpu', '?')} per GPU x {plan.get('num_gpus', '?')} RTX 5090,
  DDP){took}.
- AdamW, lr {plan.get('lr')} (ResNet-18 backbone {plan.get('lr_backbone')}), bf16 mixed precision,
  no image augmentation. Otherwise LeRobot's ACT defaults (chunk {pol['chunk_size']}, VAE on, KL weight
  {pol.get('kl_weight')}).
- Validation loss = L1 action error on the held-out demos (VAE off). Train loss also contains the
  KL term, so the two curves are not on the same scale.

| Checkpoint | Epoch | Val loss |
|---|---|---|
{ck_rows}

{'![Loss curve](loss_curve.png)' if has_png else ''}

## Run it

```bash
lerobot-rollout \\
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=<your_calibration_id> \\
  --robot.max_relative_target={cfg['MAX_RELATIVE_TARGET']} \\
  --robot.cameras='{json.dumps(cams)}' \\
  --policy.path={repo} --policy.device=cuda \\
  --policy.n_action_steps=1 --policy.temporal_ensemble_coeff={cfg['TEMPORAL_ENSEMBLE_COEFF']} \\
  --task="{cfg['TASK']}" --fps={info['fps']} \\
  --strategy.type=base --duration=0 --display_data=true
```

- The camera key must be `wrist`, mounted on the gripper with the fingers at the top of the
  image (no rotation), 640x480.
- Start the arm from the demos' start pose: high enough that the wrist camera sees the pen.
- `max_relative_target` limits each joint to {cfg['MAX_RELATIVE_TARGET']} degrees per tick; temporal ensembling
  (coeff {cfg['TEMPORAL_ENSEMBLE_COEFF']}) re-plans every tick for smooth motion. For plain chunked execution use
  `--policy.n_action_steps=30` and drop the ensemble flag.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = {k: os.path.expandvars(v) for k, v in td.env_config().items()}
    cfg["CAMERAS"] = os.environ["CAMERAS"]          # JSON: take it from the shell, not the regex
    for k in ("MODEL_REPO", "DATASET_REPO", "DATASET_ROOT", "TASK", "MODEL_PRIVATE"):
        cfg[k] = os.environ[k]
    d = td.collect()
    ck = next((c for c in d["checkpoints"] if c["name"] == a.checkpoint), None)
    if ck is None:
        print(f"No checkpoint {a.checkpoint}", file=sys.stderr)
        return 1
    src = HERE / cfg["OUTPUT_DIR"] / "checkpoints" / a.checkpoint / "pretrained_model"
    info = json.loads((Path(cfg["DATASET_ROOT"]) / "meta" / "info.json").read_text())
    pol = json.loads((src / "config.json").read_text())

    stage = td.RUN / "model_upload"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, stage / f.name)
    png = td.write_png()
    if png:
        shutil.copy2(png, stage / "loss_curve.png")
    (stage / "README.md").write_text(model_card(cfg, ck, d, info, pol, png is not None))

    total = sum(f.stat().st_size for f in stage.iterdir())
    print(f"Staged in {stage} ({total / 1e6:.0f} MB):")
    for f in sorted(stage.iterdir()):
        print(f"   {f.stat().st_size / 1e6:8.1f} MB  {f.name}")
    if a.dry_run:
        print("Dry run: nothing uploaded. Read the model card: " + str(stage / "README.md"))
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    repo = api.create_repo(cfg["MODEL_REPO"], private=cfg["MODEL_PRIVATE"] == "true", exist_ok=True).repo_id
    val = f", val {ck['val']:.4f}" if ck["val"] is not None else ""
    commit = api.upload_folder(
        repo_id=repo, repo_type="model", folder_path=stage,
        commit_message=f"ACT checkpoint {ck['name']} (epoch {ck['epoch']:.0f}{val})")
    tag = f"step-{ck['name']}"
    api.create_tag(repo, tag=tag, revision=commit.oid, exist_ok=True)
    print(f"Pushed checkpoint {ck['name']} to https://huggingface.co/{repo} (tag {tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

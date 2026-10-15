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
import subprocess
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
    cams = json.loads(cfg["CAMERAS"])
    cams["wrist"]["index_or_path"] = "/dev/video0"  # the user's own camera path goes here
    val = f"{ck['val']:.4f}" if ck["val"] is not None else "n/a"
    return f"""---
library_name: lerobot
pipeline_tag: robotics
license: apache-2.0
model_name: molmoact2
base_model: allenai/MolmoAct2-SO100_101
datasets: {ds_repo}
tags:
- robotics
- lerobot
- molmoact2
- vla
- so101
- wrist-camera
- pick-and-place
---

# MolmoAct2: pick up the pen and put it in the bowl (SO-101, wrist camera only)

[MolmoAct2](https://huggingface.co/docs/lerobot/molmoact2) (Ai2, 5.4B) fine-tuned with
[LeRobot](https://github.com/huggingface/lerobot) from `allenai/MolmoAct2-SO100_101` on
{info['total_episodes']} teleoperated demonstrations of **"{cfg['TASK']}"** on an SO-101 arm.
The policy sees **only** the gripper-mounted (wrist) camera and the joint positions; it predicts
{pol['chunk_size']} absolute joint targets (1 s at {info['fps']} fps) with its flow-matching action expert.

| Feature | Shape |
|---|---|
{rows}

## This checkpoint

Step **{ck['step']:,}** = epoch **{ck['epoch']:.0f}** of {plan.get('epochs', '?')}, validation loss **{val}**.

## Training

- Dataset: [{ds_repo}](https://huggingface.co/datasets/{ds_repo}), {info['total_episodes']} demos /
  {info['total_frames']:,} frames; {plan.get('train_episodes', '?')} for training, the last
  {plan.get('val_episodes', '?')} held out for validation.
- LoRA on the VLM, action expert fully trainable (the docs' recipe for < 200 demos), bf16,
  gradient checkpointing, image augmentation on. {plan.get('epochs', '?')} epochs =
  {plan.get('total_steps', 0):,} steps at batch {plan.get('effective_batch', '?')} on one RTX 5090.
- **Joint convention:** the base checkpoint uses the pre-LeRobot-0.5 SO-100/101 calibration
  (`model = sign * arm + offset`, signs {cfg['JOINT_SIGNS']}, offsets {cfg['JOINT_OFFSETS']}). Training used a
  copy of the dataset converted into that convention (state, action and their statistics), so
  the pretrained meaning of every joint is kept. This published model has `joint_signs` /
  `joint_offsets` switched ON: it takes state and returns actions in the **LeRobot >= 0.5 arm
  convention**, i.e. it runs directly on a currently calibrated SO-101.

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
  --policy.path={repo} --policy.device=cuda --policy.n_action_steps={cfg['N_ACTION_STEPS']} \\
  --task="{cfg['TASK']}" --fps={info['fps']} \\
  --strategy.type=base --duration=0 --display_data=true
```

- Runs on one 24-32 GB GPU (the docs measure 12.1 GiB at inference). At first load it also
  fetches the base `allenai/MolmoAct2-SO100_101` (21.8 GB), which the policy builds itself from.
- The camera key must be `wrist`, mounted on the gripper with the fingers at the top of the
  image (no rotation), 640x480. Start from the demos' start pose.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = {k: os.path.expandvars(v) for k, v in td.env_config().items()}
    cfg["CAMERAS"] = os.environ["CAMERAS"]          # JSON: take it from the shell, not the regex
    for k in ("MODEL_REPO", "DATASET_REPO", "DATASET_ROOT", "TASK", "MODEL_PRIVATE", "JOINT_SIGNS",
              "JOINT_OFFSETS", "N_ACTION_STEPS", "MAX_RELATIVE_TARGET"):
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
    # The deployable copy: joint transform ON, base checkpoint named by its Hub id.
    rc = subprocess.run([sys.executable, str(HERE / "deploy_config.py"), str(src), str(stage), "--hub-base"]).returncode
    if rc:
        return rc
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
        commit_message=f"MolmoAct2 checkpoint {ck['name']} (epoch {ck['epoch']:.0f}{val})")
    tag = f"step-{ck['name']}"
    api.create_tag(repo, tag=tag, revision=commit.oid, exist_ok=True)
    print(f"Pushed checkpoint {ck['name']} to https://huggingface.co/{repo} (tag {tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

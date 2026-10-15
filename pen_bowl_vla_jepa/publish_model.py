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
    cams = json.loads(cfg["CAMERAS"])
    cams["wrist"]["index_or_path"] = "/dev/video0"  # the user's own camera path goes here
    val = f"{ck['val']:.4f}" if ck["val"] is not None else "n/a"
    return f"""---
library_name: lerobot
pipeline_tag: robotics
license: apache-2.0
model_name: vla_jepa
base_model: lerobot/VLA-JEPA-SimplerEnv
datasets: {ds_repo}
tags:
- robotics
- lerobot
- vla_jepa
- world-model
- so101
- wrist-camera
- pick-and-place
---

# VLA-JEPA: pick up the pen and put it in the bowl (SO-101, wrist camera only)

[VLA-JEPA](https://huggingface.co/docs/lerobot/vla_jepa) fine-tuned with
[LeRobot](https://github.com/huggingface/lerobot) from `lerobot/VLA-JEPA-SimplerEnv` on
{info['total_episodes']} teleoperated demonstrations of **"{cfg['TASK']}"** on an SO-101 arm.
The policy sees **only** the gripper-mounted (wrist) camera and the joint positions.

- **Backbone:** Qwen3-VL-2B reads the wrist image and the task text.
- **Action head:** a DiT flow-matching head predicts {pol['chunk_size']} joint targets (1 s at {info['fps']} fps).
- **World model (training only):** a V-JEPA2 video encoder plus a predictor learn to predict the
  wrist camera's future frame latents, which shapes the backbone's features. It is not used at inference.

| Feature | Shape |
|---|---|
{rows}

## This checkpoint

Step **{ck['step']:,}** = epoch **{ck['epoch']:.0f}** of {plan.get('epochs', '?')}, validation loss **{val}**.

## Training

- Dataset: [{ds_repo}](https://huggingface.co/datasets/{ds_repo}), {info['total_episodes']} demos /
  {info['total_frames']:,} frames; {plan.get('train_episodes', '?')} for training, the last
  {plan.get('val_episodes', '?')} held out for validation.
- {plan.get('epochs', '?')} epochs = {plan.get('total_steps', 0):,} steps, batch {plan.get('effective_batch', '?')}
  ({plan.get('batch_size_per_gpu', '?')} per GPU x {plan.get('num_gpus', '?')} RTX 5090), full fine-tune in bf16,
  AdamW lr {plan.get('lr')} with cosine decay.
- Adapted to this robot: 6-D joint action and state (the action/state projection layers were
  re-initialised), continuous gripper (no binarisation), chunk {pol['chunk_size']}, images resized to 224x224.
- Training-time fixes on top of lerobot v0.6.0 (inference needs none of them): the action head
  gets the current joint state (v0.6.0 passes the last state of the 8-frame video window, i.e.
  a future one); images are resized to 224x224 in training as they are at inference; the
  world model's 8 frames are taken every {cfg.get('VIDEO_FRAME_STRIDE', '?')} frames (~1 s of motion at 30 Hz).

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

- Runs on one 24-32 GB GPU (bf16). At first load it also fetches `Qwen/Qwen3-VL-2B-Instruct` and
  `facebook/vjepa2-vitl-fpc64-256` (named in config.json).
- The camera key must be `wrist`, mounted on the gripper with the fingers at the top of the
  image (no rotation), 640x480. Start from the demos' start pose.
- `n_action_steps` = how many of the {pol['chunk_size']} predicted actions run before predicting again.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = {k: os.path.expandvars(v) for k, v in td.env_config().items()}
    cfg["CAMERAS"] = os.environ["CAMERAS"]          # JSON: take it from the shell, not the regex
    for k in ("MODEL_REPO", "DATASET_REPO", "DATASET_ROOT", "TASK", "MODEL_PRIVATE", "VIDEO_FRAME_STRIDE",
              "N_ACTION_STEPS", "MAX_RELATIVE_TARGET"):
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
            os.link(f, stage / f.name)   # hard link: the weights are ~6 GB, no need to copy
    # config.json names the base models by LOCAL path (this machine's downloads). Publish Hub ids.
    pub = dict(pol, qwen_model_name="Qwen/Qwen3-VL-2B-Instruct",
               jepa_encoder_name="facebook/vjepa2-vitl-fpc64-256", repo_id=cfg["MODEL_REPO"])
    (stage / "config.json").unlink()
    (stage / "config.json").write_text(json.dumps(pub, indent=2))
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
        commit_message=f"VLA-JEPA checkpoint {ck['name']} (epoch {ck['epoch']:.0f}{val})")
    tag = f"step-{ck['name']}"
    api.create_tag(repo, tag=tag, revision=commit.oid, exist_ok=True)
    print(f"Pushed checkpoint {ck['name']} to https://huggingface.co/{repo} (tag {tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

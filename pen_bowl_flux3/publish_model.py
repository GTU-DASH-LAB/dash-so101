#!/usr/bin/env python
"""Stage one FLUX 3 LoRA checkpoint (+ model card + license) and push it to the HF Hub.

Called by ./2b_publish_model.sh, which resolves the checkpoint name and checks the login.

    publish_model.py --checkpoint 006000 [--dry-run]

The upload is the deployable copy (deploy_config.py: EMA weights, base = the Hub id), so
`--policy.path=<repo>` works on any machine with access to the gated FLUX 3 weights. It
becomes the repo's main branch and is tagged `step-006000`.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
BASE_HUB = "black-forest-labs/flux-3-action-so101"


def eval_losses(log: Path) -> dict[int, float]:
    """step -> validation loss, from lerobot-train's `step N: eval_loss=X` lines."""
    if not log.is_file():
        return {}
    pat = re.compile(r"step (\d+): eval_loss=([0-9.]+)")
    return {int(m[1]): float(m[2]) for m in pat.finditer(log.read_text(errors="replace"))}


def model_card(env: dict, step: int, losses: dict, info: dict, pol: dict, train: dict) -> str:
    repo, ds_repo = env["MODEL_REPO"], env["DATASET_REPO"]
    cams = json.loads(env["CAMERAS"])
    cams["wrist"]["index_or_path"] = "/dev/video0"  # the user's own camera path goes here
    best = min(losses, key=losses.get) if losses else None
    rows = "\n".join(f"| {s:,} | {v:.4f}{' (lowest)' if s == best else ''} |" for s, v in sorted(losses.items()))
    val = f"{losses[step]:.4f}" if step in losses else "n/a"
    accum = train.get("accelerator", {}).get("gradient_accumulation", {}).get("steps", env["GRAD_ACCUM"])
    peft = train.get("peft") or {}
    return f"""---
library_name: lerobot
pipeline_tag: robotics
license: other
license_name: flux-kommunity-license
license_link: LICENSE.md
base_model: {BASE_HUB}
datasets: {ds_repo}
tags:
- robotics
- lerobot
- flux3
- world-action-model
- so101
- wrist-camera
- lora
---

# FLUX 3 Action: pick up the pen and put it in the bowl (SO-101, wrist camera only)

LoRA fine-tune of [FLUX 3 Action]({'https://huggingface.co/' + BASE_HUB}) (Black Forest Labs, a 7B
world action model that denoises the next actions together with the next video frames) with
[LeRobot](https://github.com/huggingface/lerobot) on {info['total_episodes']} teleoperated
demonstrations of **"{env['TASK']}"** on an SO-101 arm. The policy sees **only** the
gripper-mounted (wrist) camera and the joint positions, and predicts {pol.get('chunk_size', 32)}
actions (~1 s at {info['fps']} fps) per call.

## This checkpoint

Step **{step:,}** (EMA weights), validation loss **{val}**. LoRA r={peft.get('r', '?')},
alpha={peft.get('lora_alpha', '?')} on every DiT attention/MLP projection; the embodiment
heads trained in full.

## Training

- Dataset: [{ds_repo}](https://huggingface.co/datasets/{ds_repo}), {info['total_episodes']} demos /
  {info['total_frames']:,} frames; {float(train.get('dataset', {}).get('eval_split', env['EVAL_SPLIT'])):.0%} of the demos held out for validation.
- Recipe: upstream `examples/flux3/lora.json` (bf16, gradient checkpointing, EMA 0.999), batch
  {train.get('batch_size', env['BATCH_SIZE'])} x {accum} accumulation, {train.get('steps', '?'):,} microsteps, one H100 80 GB.
- **Wrist only:** the SO-101 base composes scene|wrist side by side (each 256x256). Here the
  layout is `single` on a 256x256 canvas, so the wrist image has exactly its pretraining size.
- **Joint-state normalization:** FLUX 3 keeps its own q01/q99 statistics. The base's come from
  SO-101 data in the pre-LeRobot-0.5 calibration, so the state quantiles were replaced by this
  dataset's (LeRobot >= 0.5 calibration). Action quantiles (per-step deltas) were kept.

| Step | Val loss |
|---|---|
{rows}

## Run it

Needs lerobot from GitHub main (FLUX 3 is not in a release yet) with the `flux3` and `peft`
extras, NATTEN, and access to the gated base weights (accept the license on the base repo).

```bash
lerobot-rollout \\
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=<your_calibration_id> \\
  --robot.max_relative_target={env['MAX_RELATIVE_TARGET']} \\
  --robot.cameras='{json.dumps(cams)}' \\
  --policy.path={repo} --policy.use_peft=true --policy.device=cuda --policy.n_action_steps={env['N_ACTION_STEPS']} \\
  --task="{env['TASK']}" --fps={info['fps']} \\
  --strategy.type=base --duration=0 --display_data=true
```

- One 32 GB GPU: 13.9 GB DiT + 8.9 GB text encoder + 2.5 GB video VAE in bf16. First load
  fetches `{BASE_HUB}` (13.9 GB) and the text encoder / video VAE from
  `black-forest-labs/flux-3-action-base`.
- The camera key must be `wrist`, mounted on the gripper, 640x480. Start from the demos' start pose.

## License

A derivative of FLUX 3 Action, released under the same
[FLUX Kommunity License](LICENSE.md) as the base model.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    env = os.environ
    ck = HERE / env["OUTPUT_DIR"] / "checkpoints" / a.checkpoint
    stage = HERE / "run" / "model_upload"
    rc = subprocess.run([sys.executable, str(HERE / "deploy_config.py"), str(ck), str(stage)]).returncode
    if rc:
        return rc
    info = json.loads((Path(env["DATASET_ROOT"]) / "meta" / "info.json").read_text())
    pol = json.loads((stage / "config.json").read_text())
    tc = stage / "train_config.json"
    train = json.loads(tc.read_text()) if tc.is_file() else {}
    step = int(a.checkpoint)
    shutil.copy2(hf_hub_download(BASE_HUB, "LICENSE.md"), stage / "LICENSE.md")
    (stage / "README.md").write_text(model_card(env, step, eval_losses(HERE / "aws_train.log"), info, pol, train))

    total = sum(f.stat().st_size for f in stage.iterdir())
    print(f"Staged in {stage} ({total / 1e6:.0f} MB):")
    for f in sorted(stage.iterdir()):
        print(f"   {f.stat().st_size / 1e6:8.1f} MB  {f.name}")
    if a.dry_run:
        print("Dry run: nothing uploaded. Read the model card: " + str(stage / "README.md"))
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    repo = api.create_repo(env["MODEL_REPO"], private=env["MODEL_PRIVATE"] == "true", exist_ok=True).repo_id
    commit = api.upload_folder(repo_id=repo, repo_type="model", folder_path=stage,
                               commit_message=f"FLUX 3 Action LoRA checkpoint {a.checkpoint} (EMA)")
    tag = f"step-{a.checkpoint}"
    api.create_tag(repo, tag=tag, revision=commit.oid, exist_ok=True)
    print(f"Pushed checkpoint {a.checkpoint} to https://huggingface.co/{repo} (tag {tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

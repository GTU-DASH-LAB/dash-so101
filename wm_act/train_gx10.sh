#!/usr/bin/env bash
# Train WM-ACT from scratch on the pencil-pickup dataset. Meant for the ASUS GX10
# (GB10 Grace Blackwell, CUDA); works anywhere by changing --policy.device.
set -euo pipefail
cd "$(dirname "$0")/.."
source pencil_pickup_vla/config.env

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

lerobot-train \
  --policy.discover_packages_path=wm_act \
  --policy.type=wm_act \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id="${HF_USER}/wmact_pencil_pickup" \
  --dataset.repo_id="${DATASET_REPO}" \
  --batch_size=64 \
  --steps=60000 \
  --save_freq=10000 \
  --output_dir=outputs/train/wmact_pencil_pickup \
  --job_name=wmact_pencil_pickup \
  --wandb.enable=false

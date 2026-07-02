#!/usr/bin/env bash
# Train Point-ACT on the GX10. Step 1 auto-labels the dataset with OWLv2 (one-time,
# ~a minute for 50 episodes); step 2 trains with the sidecar labels.
set -euo pipefail
cd "$(dirname "$0")/.."
source pencil_pickup_vla/config.env

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

LABELS=outputs/point_labels_pencil.json
if [ ! -f "$LABELS" ]; then
  python so_brain/relabel.py \
    --repo-id "${DATASET_REPO}" \
    --pick "a pen" --place "a black mouse pad" \
    --out "$LABELS"
fi

lerobot-train \
  --policy.discover_packages_path=point_act \
  --policy.type=point_act \
  --policy.point_labels_path="$PWD/$LABELS" \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id="${HF_USER}/pointact_pickplace" \
  --dataset.repo_id="${DATASET_REPO}" \
  --batch_size=64 \
  --steps=60000 \
  --save_freq=10000 \
  --output_dir=outputs/train/pointact_pickplace \
  --job_name=pointact_pickplace \
  --wandb.enable=false

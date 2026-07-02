#!/usr/bin/env bash
# ============================================================================
# Stage 2 — fine-tune pi0 LOCALLY on the recorded dataset (this machine's GPU).
# Starts from the pretrained lerobot/pi0_base (~3B params, already cached).
#
# pi0 is heavy: expect this to take a while and use a lot of GPU memory.
# Checkpoints are written every SAVE_FREQ steps so you can resume (see below).
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
source ../.venv/bin/activate

# NOTE: we do NOT force HF_HUB_OFFLINE here — the first run may need to fetch
# pi0_base's tokenizer/companion files, and (if enabled) the model is pushed
# to the Hub at the end of training. The recorded dataset is read locally via
# --dataset.root regardless.

# Sanity check: dataset must exist locally before training.
if [ ! -d "$DATASET_ROOT" ]; then
  echo "ERROR: dataset not found at '$DATASET_ROOT'." >&2
  echo "Record demonstrations first with ./1_record.sh" >&2
  exit 1
fi

# Fail fast (BEFORE the long training run) if we intend to push but aren't
# logged in — otherwise you find out 20000 steps later.
if [ "$PUSH_MODEL_TO_HUB" = "true" ]; then
  hf auth whoami >/dev/null 2>&1 || { echo "ERROR: run 'hf auth login' first (or set PUSH_MODEL_TO_HUB=false)." >&2; exit 1; }
fi

echo "Fine-tuning $BASE_MODEL on $DATASET_REPO"
echo "  batch_size=$BATCH_SIZE  steps=$STEPS  output=$OUTPUT_DIR"
echo

lerobot-train \
  --policy.path="$BASE_MODEL" \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi0_ball_pickup \
  --policy.device=cuda \
  --policy.push_to_hub="$PUSH_MODEL_TO_HUB" \
  --policy.repo_id="$MODEL_REPO" \
  --wandb.enable=false

echo
echo "Done. Trained checkpoints under: $OUTPUT_DIR/checkpoints/"
echo "Newest is symlinked at:          $OUTPUT_DIR/checkpoints/last/pretrained_model"
if [ "$PUSH_MODEL_TO_HUB" = "true" ]; then
  echo "Pushed to: https://huggingface.co/$MODEL_REPO"
fi
echo "Next: ./3_run_autonomous.sh"

# --- To RESUME an interrupted run, re-run with: ------------------------------
#   lerobot-train --config_path="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json" --resume=true

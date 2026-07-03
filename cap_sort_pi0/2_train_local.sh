#!/usr/bin/env bash
# ============================================================================
# Stage 2 — LoRA fine-tune pi0 LOCALLY on the recorded dataset (this machine's GPU).
# Starts from the pretrained lerobot/pi0_base (~4B params, already cached),
# freezing ~99% of it and training only LoRA adapters (see config.env for the
# verified batch size / memory numbers).
#
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

echo "LoRA fine-tuning $BASE_MODEL on $DATASET_REPO"
echo "  batch_size=$BATCH_SIZE  steps=$STEPS  peft.r=$PEFT_R  peft.lora_alpha=$PEFT_ALPHA  output=$OUTPUT_DIR"
echo

# bf16 mixed precision: this is NOT the same as `--policy.use_amp` (that flag
# is validated but never wired into Accelerate's mixed-precision setting in
# this lerobot version, so it's a no-op). Accelerate reads this env var
# instead -- see CLAUDE.md for the full story (found while tuning
# ball_pickup_pi0's full fine-tune; applies here too).
export ACCELERATE_MIXED_PRECISION=bf16
# Reduces the chance of an allocator-fragmentation OOM during the memory-spiky
# first couple of warmup steps (CUDA kernel autotuning).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# No --rename_map needed (unlike ball_pickup_pi0): config.env's CAMERAS
# already uses pi0's own base_0_rgb/left_wrist_0_rgb names directly, so the
# dataset's columns already match what pi0 expects.
#
# --peft.method_type=LORA (plus .r/.lora_alpha) is what actually turns PEFT
# on -- NOT --policy.use_peft (that flag is for loading an EXISTING PEFT
# checkpoint on resume, it does nothing for a fresh run; see
# lerobot/scripts/lerobot_train.py: `if cfg.peft is not None:` is the real
# gate, driven by these --peft.* flags). target_modules is left unset since
# pi0 provides its own sensible default (PI0Policy._get_default_peft_targets:
# the gemma-expert's attention q/v projections + action/state projections).
lerobot-train \
  --policy.path="$BASE_MODEL" \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi0_cap_sort \
  --policy.device=cuda \
  --policy.push_to_hub="$PUSH_MODEL_TO_HUB" \
  --policy.repo_id="$MODEL_REPO" \
  --peft.method_type=LORA \
  --peft.r="$PEFT_R" \
  --peft.lora_alpha="$PEFT_ALPHA" \
  --wandb.enable=false

echo
echo "Done. Trained LoRA checkpoints under: $OUTPUT_DIR/checkpoints/"
echo "Newest is symlinked at:               $OUTPUT_DIR/checkpoints/last/pretrained_model"
if [ "$PUSH_MODEL_TO_HUB" = "true" ]; then
  echo "Pushed to: https://huggingface.co/$MODEL_REPO"
fi
echo "Next: ./3_run_autonomous.sh"

# --- To RESUME an interrupted run, re-run with: ------------------------------
#   lerobot-train --config_path="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json" --resume=true

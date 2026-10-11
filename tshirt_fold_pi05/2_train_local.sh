#!/usr/bin/env bash
# ============================================================================
# Stage 2 — LoRA fine-tune pi0.5 LOCALLY on the already-recorded/cleaned
# dataset, across BOTH RTX 5090s via accelerate's DDP launcher.
#
# This machine has no NVLink/P2P and GPU1 sits on a narrower (x4) PCIe link
# than GPU0 -- robotics/CLAUDE.md's house rule normally prefers independent
# single-GPU runs over DDP for VLA fine-tuning *because of* that link
# asymmetry. LoRA changes the calculus: only ~1.4M params' gradients (a few
# MB) cross PCIe per step instead of billions, so the all-reduce is
# negligible next to per-step compute time even over x4 -- verified with a
# real smoke test (see watchdog.log) before this was committed to for the
# full run, per the user's explicit request to use both GPUs if it's
# actually faster.
#
# Checkpoints are written every SAVE_FREQ steps so you can resume (see below).
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
export PATH="$VENV_BIN:$PATH"

# Sanity check: dataset must exist locally before training.
if [ ! -d "$DATASET_ROOT" ]; then
  echo "ERROR: dataset not found at '$DATASET_ROOT'." >&2
  exit 1
fi

if [ "$PUSH_MODEL_TO_HUB" = "true" ]; then
  hf auth whoami >/dev/null 2>&1 || { echo "ERROR: run 'hf auth login' first (or set PUSH_MODEL_TO_HUB=false)." >&2; exit 1; }
fi

echo "LoRA fine-tuning $BASE_MODEL on $DATASET_REPO across $NUM_GPUS GPU(s)"
echo "  per-gpu batch_size=$BATCH_SIZE (effective=$((BATCH_SIZE * NUM_GPUS)))  steps=$STEPS  peft.r=$PEFT_R  peft.lora_alpha=$PEFT_ALPHA  output=$OUTPUT_DIR"
echo

# bf16 mixed precision: Accelerate reads this env var directly; `--policy.use_amp`
# is validated but never wired in for this lerobot version (see cap_sort_pi0's
# notes for the full story).
export ACCELERATE_MIXED_PRECISION="$ACCEL_MP"
# Reduces the chance of an allocator-fragmentation OOM during the memory-spiky
# first couple of warmup steps (CUDA kernel autotuning).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --rename_map IS required here (like ball_pickup_pi0, unlike cap_sort_pi0):
# the dataset was recorded through the lelab web UI, whose camera keys are
# `workspace_cam`/`wrist_camera`, while pi05_base's own config.json declares
# observation.images.{base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb}. Without
# the remap, make_policy raises `Feature mismatch`. The unfilled third slot
# (right_wrist_0_rgb) is fine -- pi0.5 masks out declared-but-absent image
# keys internally.
#
# --peft.method_type=LORA (plus .r/.lora_alpha) is what actually turns PEFT
# on -- NOT --policy.use_peft (that's only for loading an EXISTING adapter on
# resume/inference). target_modules is left unset: PI05Policy provides its
# own default via _get_default_peft_targets().
#
# accelerate launch -m --multi_gpu --num_processes=N treats the positional
# arg as a Python module (same as `python -m lerobot.scripts.lerobot_train`),
# with everything after the bare `--` forwarded to lerobot-train's own
# draccus arg parser untouched.
accelerate launch -m --multi_gpu --num_processes="$NUM_GPUS" \
  lerobot.scripts.lerobot_train -- \
  --policy.path="$BASE_MODEL" \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --rename_map="$RENAME_MAP" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi05_tshirt_fold \
  --policy.device=cuda \
  --policy.dtype="$POLICY_DTYPE" \
  --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING" \
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

# --- To RESUME an interrupted run, re-run with: ------------------------------
#   accelerate launch -m --multi_gpu --num_processes="$NUM_GPUS" lerobot.scripts.lerobot_train -- \
#     --config_path="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json" --resume=true

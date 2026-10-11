#!/usr/bin/env bash
# ============================================================================
# Real smoke test: run a handful of actual training steps at a given per-GPU
# batch size and report measured throughput + peak per-GPU memory, so STEPS
# and BATCH_SIZE are picked from data instead of guessed.
#
# This project's own methodology (cap_sort_pi0's config.env cites a "real
# 10-step smoke test") -- but the memory numbers in dash-so101/CLAUDE.md come
# from a GB10 box with 121GB *unified* CPU+GPU memory, which this machine is
# NOT. This box has two discrete RTX 5090s with 32,607 MiB of VRAM EACH, so
# those numbers do not transfer and everything must be re-measured here.
#
# Usage:
#   ./smoke_test.sh <per_gpu_batch_size> [num_gpus] [steps]
# e.g.
#   ./smoke_test.sh 4          # 4/GPU across both GPUs (effective 8), 12 steps
#   ./smoke_test.sh 8 1        # single-GPU baseline, for the DDP speedup check
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
export PATH="$VENV_BIN:$PATH"

BS="${1:?usage: ./smoke_test.sh <per_gpu_batch_size> [num_gpus] [steps]}"
GPUS="${2:-$NUM_GPUS}"
SMOKE_STEPS="${3:-12}"
OUT="./outputs/smoke/bs${BS}_gpu${GPUS}"
LOG="./smoke_bs${BS}_gpu${GPUS}.log"

rm -rf "$OUT"
mkdir -p "$(dirname "$OUT")"

export ACCELERATE_MIXED_PRECISION="$ACCEL_MP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "SMOKE TEST: per-GPU batch=$BS, gpus=$GPUS (effective=$((BS * GPUS))), steps=$SMOKE_STEPS"

# Sample GPU memory in the background for the duration of the run -- the
# script's own mem_gb reporting is per-process, and we care about the real
# per-card ceiling against 32,607 MiB.
( while :; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
    sleep 2
  done ) > "./smoke_gpumem_bs${BS}_gpu${GPUS}.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

if [ "$GPUS" -gt 1 ]; then
  LAUNCH=(accelerate launch -m --multi_gpu --num_processes="$GPUS" lerobot.scripts.lerobot_train --)
else
  LAUNCH=(accelerate launch -m --num_processes=1 lerobot.scripts.lerobot_train --)
fi

"${LAUNCH[@]}" \
  --policy.path="$BASE_MODEL" \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --rename_map="$RENAME_MAP" \
  --batch_size="$BS" \
  --steps="$SMOKE_STEPS" \
  --log_freq=1 \
  --save_freq="$((SMOKE_STEPS + 1000))" \
  --output_dir="$OUT" \
  --job_name="smoke_bs${BS}_gpu${GPUS}" \
  --policy.device=cuda \
  --policy.dtype="$POLICY_DTYPE" \
  --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING" \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.r="$PEFT_R" \
  --peft.lora_alpha="$PEFT_ALPHA" \
  --wandb.enable=false 2>&1 | tee "$LOG"

kill $SAMPLER 2>/dev/null || true

echo
echo "=============== SMOKE RESULT (bs=$BS, gpus=$GPUS) ==============="
echo "--- learnable params (confirms LoRA actually engaged) ---"
grep -iE "learnable|trainable" "$LOG" | head -3
echo "--- last few step lines ---"
grep -E "step:" "$LOG" | tail -5
echo "--- peak GPU memory per card (MiB, of 32607) ---"
awk -F', ' '{if ($2 > peak[$1]) peak[$1]=$2} END {for (g in peak) printf "  gpu%s peak: %s MiB (%.0f%% of 32607)\n", g, peak[g], 100*peak[g]/32607}' \
  "./smoke_gpumem_bs${BS}_gpu${GPUS}.csv" | sort
echo "================================================================"

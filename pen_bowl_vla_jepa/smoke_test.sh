#!/usr/bin/env bash
# ============================================================================
# Real smoke test: a few actual training steps through the same launcher as the
# real run (train_vla_jepa.py), reporting step time, samples/s and peak VRAM per
# card. Use it before changing BATCH_SIZE or NUM_GPUS in config.env.
#
#   ./smoke_test.sh <per_gpu_batch> [gpus: 1|2] [steps]
#   ./smoke_test.sh 8 1 15        batch 8 on GPU 0 only
#
# Writes nothing outside run/smoke/. Never runs while training is running.
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"
export BASE_POLICY QWEN_PATH JEPA_PATH INIT_POLICY_DIR CHUNK_SIZE N_ACTION_STEPS IMAGE_SIZE WARMUP_STEPS \
  DATASET_ROOT MIXED_PRECISION VIDEO_FRAME_STRIDE GRADIENT_CHECKPOINTING

BS="${1:?usage: ./smoke_test.sh <per_gpu_batch> [gpus] [steps]}"
GPUS="${2:-1}"
STEPS="${3:-15}"
[ -f "$RUN_DIR/watchdog.pid" ] && kill -0 "$(cat "$RUN_DIR/watchdog.pid")" 2>/dev/null && die "Training is running."

TAG="bs${BS}_gpu${GPUS}"
SMOKE="$RUN_DIR/smoke"
OUT="$SMOKE/out_$TAG"
LOG="$SMOKE/$TAG.log"
MEM="$SMOKE/$TAG.mem.csv"
mkdir -p "$SMOKE"
[ -d "$OUT" ] && mv "$OUT" "$OUT.old.$(date +%s)"

[ -f "$INIT_POLICY_DIR/config.json" ] || { "$PY" prepare_init.py || exit 1; }

if [ "$GPUS" -gt 1 ]; then DEVS=0,1; LAUNCH=(accelerate launch --mixed_precision="$MIXED_PRECISION" --multi_gpu --num_processes="$GPUS")
else DEVS=0; LAUNCH=(accelerate launch --mixed_precision="$MIXED_PRECISION" --num_processes=1); fi

( while :; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; sleep 1; done ) > "$MEM" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

info "Smoke test: batch $BS/GPU on GPU(s) $DEVS, $STEPS steps -> $LOG"
CUDA_VISIBLE_DEVICES=$DEVS HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${LAUNCH[@]}" "$HERE/train_vla_jepa.py" \
  --policy.path="$INIT_POLICY_DIR" --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="$DATASET_REPO" --dataset.root="$DATASET_ROOT" --dataset.video_backend=torchcodec \
  --batch_size="$BS" --num_workers="$NUM_WORKERS" --steps="$STEPS" --log_freq=1 \
  --save_freq=100000 --eval_steps=0 --output_dir="$OUT" --job_name="smoke_$TAG" --wandb.enable=false \
  > "$LOG" 2>&1
rc=$?
kill $SAMPLER 2>/dev/null

echo
echo "================ SMOKE RESULT: batch $BS/GPU x $GPUS GPU(s), exit $rc ================"
grep -E "learnable|num_total_params|reinit_modules" "$LOG" | sed 's/^.*INFO/INFO/' | head -4
grep -E "OutOfMemory|Error|error:" "$LOG" | tail -3
# Steady state = skip the first 5 steps (compile/autotune warm-up).
grep -oE "updt_s:[0-9.]+ data_s:[0-9.]+" "$LOG" | tail -n +6 | awk -v bs="$BS" -v g="$GPUS" '
  {split($1,a,":"); split($2,b,":"); u+=a[2]; d+=b[2]; n++}
  END {if (n) printf "steady state (%d steps): update %.2f s + data %.3f s per step -> %.1f samples/s\n", n, u/n, d/n, bs*g/((u+d)/n)}'
awk -F', ' '{if ($2 > p[$1]) p[$1]=$2} END {for (g in p) printf "  GPU%s peak: %d MiB of 32607 (%.0f%%)\n", g, p[g], 100*p[g]/32607}' "$MEM" | sort
echo "  (GPU1 counts other jobs too; check nvidia-smi before reading it)"
exit $rc

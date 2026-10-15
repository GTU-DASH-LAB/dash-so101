#!/usr/bin/env bash
# ============================================================================
# Runs training and keeps it alive. Started detached by ./2_train_local.sh; do
# not run it by hand.
#
# - Starts lerobot's training loop through train_molmoact2.py on NUM_GPUS GPUs.
# - If training crashes, kills the whole process group (launcher, both ranks,
#   every dataloader worker, so nothing orphaned keeps holding GPU memory),
#   waits, and RESUMES from the newest checkpoint. Gives up after 5 failures,
#   or after 2 failures in a row within 2 minutes (that is a config error, and
#   retrying would not help).
# - SIGTERM (./2_train_local.sh stop) stops it cleanly; the newest checkpoint
#   stays and `./2_train_local.sh` resumes from it.
# Lifecycle lines go to run/watchdog.log, which the dashboard reads.
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"
source "$HERE/policy_args.sh"

LOG="$RUN_DIR/train.log"
WLOG="$RUN_DIR/watchdog.log"
PLAN="$RUN_DIR/plan.json"
MAX_FAILS=5
wlog() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$WLOG"; }
plan() { "$PY" -c "import json; print(json.load(open('$PLAN'))['$1'])"; }

CHILD=""
kill_group() {
  [ -n "$CHILD" ] || return 0
  kill -TERM -- "-$CHILD" 2>/dev/null
  for _ in $(seq 20); do kill -0 -- "-$CHILD" 2>/dev/null || return 0; sleep 1; done
  kill -KILL -- "-$CHILD" 2>/dev/null
  sleep 2
}
on_stop() { wlog "STOPPED by user"; kill_group; exit 0; }
trap on_stop TERM INT

export HF_HUB_OFFLINE=1 MIXED_PRECISION PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU_IDS"   # this run's GPU(s) only; the other may run another experiment
LAUNCH=(accelerate launch --mixed_precision="$MIXED_PRECISION" --num_processes="$NUM_GPUS")
[ "$NUM_GPUS" -gt 1 ] && LAUNCH+=(--multi_gpu)
LAUNCH+=("$HERE/train_molmoact2.py")

fails=0; quick=0
while :; do
  LAST="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
  if [ -f "$LAST" ]; then
    step=$(basename "$(readlink -f "$OUTPUT_DIR/checkpoints/last")" | sed 's/^0*//')
    wlog "RESUMING from checkpoint at step ${step:-0}"
    ARGS=(--config_path="$LAST" --resume=true)
  else
    if [ -d "$OUTPUT_DIR" ]; then   # crashed before its first checkpoint: nothing to resume
      mv "$OUTPUT_DIR" "${OUTPUT_DIR}.failed.$(date +%s)"
      wlog "Moved an output dir with no checkpoint aside"
    fi
    wlog "STARTING fresh: $(plan total_steps) steps, $(plan epochs) epochs"
    ARGS=(
      "${POLICY_ARGS[@]}" --policy.scheduler_decay_steps="$(plan total_steps)"
      --dataset.eval_split="$VALIDATION_SPLIT" --eval_steps="$(plan eval_steps)"
      --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS"
      --steps="$(plan total_steps)" --save_freq="$(plan save_freq)" --log_freq="$(plan log_freq)"
      --output_dir="$OUTPUT_DIR" --job_name="$JOB_NAME" --wandb.enable=false
    )
  fi

  t0=$(date +%s)
  # setsid: training gets its own process group, so kill_group reaches every worker.
  setsid "${LAUNCH[@]}" "${ARGS[@]}" >> "$LOG" 2>&1 &
  CHILD=$!
  echo "$CHILD" > "$RUN_DIR/train.pgid"
  wait "$CHILD"; rc=$?
  ran=$(( $(date +%s) - t0 ))

  if [ $rc -eq 0 ]; then
    wlog "FINISHED successfully"
    "$PY" "$HERE/train_dashboard.py" --png >/dev/null 2>&1
    exit 0
  fi
  wlog "CRASHED (exit $rc after ${ran}s) - see run/train.log"
  kill_group; CHILD=""
  fails=$((fails + 1))
  if [ $ran -lt 120 ]; then quick=$((quick + 1)); else quick=0; fi
  if [ $fails -ge $MAX_FAILS ] || [ $quick -ge 2 ]; then
    wlog "GAVE UP after $fails failure(s) - fix the error in run/train.log, then ./2_train_local.sh"
    exit 1
  fi
  wlog "RETRYING in 30 s (failure $fails/$MAX_FAILS)"
  sleep 30 & wait $!   # interruptible, so a stop is instant
done

#!/usr/bin/env bash
# ============================================================================
# Watchdog for pi0 fine-tuning: if lerobot-train dies (OOM, crash, etc.)
# before reaching STEPS, automatically clean up and resume from the last
# checkpoint instead of losing the rest of an unattended run.
#
# Usage:
#   ./watchdog_train.sh              # start a fresh/resumed run under the watchdog
#   ./watchdog_train.sh <existing_pid>   # attach to an already-running lerobot-train
#                                         # (e.g. one started manually via 2_train_local.sh)
#
# Runs in the foreground -- launch it with nohup/tmux/setsid to survive a
# closed terminal. Progress/restarts are logged to ./watchdog.log.
# ============================================================================
set -uo pipefail   # no -e: failures are handled explicitly below
cd "$(dirname "$0")"
source ./config.env
source ../.venv/bin/activate

LOG_FILE="./watchdog.log"
MAX_RETRIES=10
RETRY_DELAY_S=30

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"; }

# Highest completed checkpoint step number, or empty if none exist yet.
last_checkpoint_step() {
  local dir
  dir=$(find "$OUTPUT_DIR/checkpoints" -maxdepth 1 -regextype posix-extended -regex '.*/[0-9]{6}' 2>/dev/null | sort | tail -1)
  [ -n "$dir" ] && basename "$dir" | sed 's/^0*//'
}

run_training() {
  export ACCELERATE_MIXED_PRECISION=bf16
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  if [ -f "$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json" ]; then
    log "Resuming from last checkpoint (step $(last_checkpoint_step))."
    lerobot-train \
      --config_path="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json" \
      --resume=true
  else
    log "Starting fresh training run."
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
      --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
      --wandb.enable=false
  fi
}

# Kill any leftover lerobot-train/dataloader-worker processes from a crashed
# run -- these can otherwise hold tens of GB and cause an avoidable OOM on
# the very next retry (see CLAUDE.md: "orphaned dataloader worker processes").
cleanup_orphans() {
  pkill -9 -f "lerobot-train.*job_name=pi0_ball_pickup" 2>/dev/null || true
  sleep 3
}

# --- If given an existing PID, wait on it before taking over ---------------
attach_pid="${1:-}"
if [ -n "$attach_pid" ]; then
  log "Attaching to existing training process PID $attach_pid."
  while kill -0 "$attach_pid" 2>/dev/null; do
    sleep 5
  done
  log "Process $attach_pid is no longer running."
  step="$(last_checkpoint_step)"
  if [ -n "$step" ] && [ "$step" -ge "$STEPS" ]; then
    log "Last checkpoint (step $step) already reached STEPS=$STEPS. Training was already complete. Nothing to do."
    exit 0
  fi
  log "Last checkpoint: ${step:-none}. Did not reach STEPS=$STEPS -- treating as a crash and resuming."
  cleanup_orphans
fi

# --- Retry loop --------------------------------------------------------------
attempt=0
while :; do
  attempt=$((attempt + 1))
  log "=== Attempt $attempt/$((MAX_RETRIES + 1)) ==="
  run_training
  exit_code=$?

  if [ "$exit_code" -eq 0 ]; then
    log "Training finished successfully (exit 0). Done."
    exit 0
  fi

  log "lerobot-train exited with code $exit_code."
  cleanup_orphans

  if [ "$attempt" -gt "$MAX_RETRIES" ]; then
    log "Reached MAX_RETRIES=$MAX_RETRIES without success. Giving up -- check $LOG_FILE."
    exit 1
  fi

  log "Waiting ${RETRY_DELAY_S}s for memory to settle before retrying..."
  sleep "$RETRY_DELAY_S"
done

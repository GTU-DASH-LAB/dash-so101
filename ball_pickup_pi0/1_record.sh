#!/usr/bin/env bash
# ============================================================================
# Stage 1 — record teleoperated demonstrations of "ball -> white cup".
# You move the LEADER arm; the FOLLOWER mirrors it and everything
# (joint states + BOTH cameras) is recorded into a dataset.
#
# Keyboard controls during recording:
#   Right Arrow -> end the current episode early, move to the next
#   Left Arrow  -> cancel & re-record the current episode
#   Escape      -> stop recording and save
#
# Data is saved LOCALLY under $DATASET_ROOT, and — if PUSH_DATASET_TO_HUB=true
# in config.env — also pushed to https://huggingface.co/datasets/$DATASET_REPO
# once recording finishes.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
source ../.venv/bin/activate

# Fail fast if the leader arm isn't connected (teleop needs both arms).
if [ ! -e "$LEADER_PORT" ]; then
  echo "ERROR: LEADER_PORT '$LEADER_PORT' not found." >&2
  echo "Plug in the leader arm and run: lerobot-find-port" >&2
  echo "then set LEADER_PORT in config.env." >&2
  exit 1
fi

# Fail fast (BEFORE recording) if we intend to push but aren't logged in,
# otherwise the push at the very end fails and it looks like the data was
# lost (it isn't — it's saved locally first regardless).
if [ "$PUSH_DATASET_TO_HUB" = "true" ]; then
  hf auth whoami >/dev/null 2>&1 || { echo "ERROR: run 'hf auth login' first (or set PUSH_DATASET_TO_HUB=false)." >&2; exit 1; }
else
  export HF_HUB_OFFLINE=1
fi

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --display_data=true \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.fps="$FPS" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.push_to_hub="$PUSH_DATASET_TO_HUB"

echo
echo "Done. Dataset saved locally at: $DATASET_ROOT"
if [ "$PUSH_DATASET_TO_HUB" = "true" ]; then
  echo "Pushed to: https://huggingface.co/datasets/$DATASET_REPO"
fi
echo "Next: ./2_train_local.sh"

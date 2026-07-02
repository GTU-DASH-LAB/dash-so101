#!/usr/bin/env bash
# ============================================================================
# Stage 3 — run YOUR fine-tuned pi0 policy on the real SO-101, autonomously.
# The policy reads the cameras + joint states and drives the follower to do
# the task. No leader arm needed here.
#
# SAFETY: keep a hand near the power switch the first time. --robot.max_relative_target
# caps how far each joint may move per step so the arm can't lurch.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
source ../.venv/bin/activate

# Model + dataset are local; skip Hub round-trips.
export HF_HUB_OFFLINE=1

# Load the newest checkpoint produced by ./2_train_local.sh
POLICY_PATH="$OUTPUT_DIR/checkpoints/last/pretrained_model"
if [ ! -d "$POLICY_PATH" ]; then
  echo "ERROR: no trained policy at '$POLICY_PATH'." >&2
  echo "Train first with ./2_train_local.sh" >&2
  exit 1
fi

lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.max_relative_target=5 \
  --robot.cameras="$CAMERAS" \
  --task="$TASK" \
  --policy.path="$POLICY_PATH" \
  --policy.device=cuda \
  --inference.type=rtc \
  --rename_map='{}' \
  --display_data=true

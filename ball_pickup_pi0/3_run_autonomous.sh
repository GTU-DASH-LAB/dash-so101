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

# Which policy to load, per POLICY_SOURCE in config.env.
case "$POLICY_SOURCE" in
  local)
    POLICY_PATH="$OUTPUT_DIR/checkpoints/last/pretrained_model"
    if [ ! -d "$POLICY_PATH" ]; then
      echo "ERROR: no trained policy at '$POLICY_PATH'." >&2
      echo "Train first with ./2_train_local.sh, or set POLICY_SOURCE=hub in config.env to pull '$MODEL_REPO' instead." >&2
      exit 1
    fi
    # Model + dataset are local; skip Hub round-trips.
    export HF_HUB_OFFLINE=1
    ;;
  hub)
    POLICY_PATH="$MODEL_REPO"
    hf auth whoami >/dev/null 2>&1 || { echo "ERROR: run 'hf auth login' first to download '$MODEL_REPO'." >&2; exit 1; }
    ;;
  *)
    echo "ERROR: POLICY_SOURCE must be 'local' or 'hub' (got '$POLICY_SOURCE')." >&2
    exit 1
    ;;
esac

# pi0_base's declared camera slots (base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb)
# never change, even in a checkpoint fine-tuned from it -- rename_map only
# remaps the DATASET's columns at training time, not the model's own feature
# names. So the same remap used in 2_train_local.sh is needed again here to
# match our robot's actual camera names (top/wrist).
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --task="$TASK" \
  --policy.path="$POLICY_PATH" \
  --policy.device=cuda \
  --inference.type=rtc \
  --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --display_data=true

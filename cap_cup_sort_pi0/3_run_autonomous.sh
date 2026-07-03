#!/usr/bin/env bash
# ============================================================================
# Stage 3 — run YOUR fine-tuned pi0 (LoRA) policy on the real SO-101, autonomously.
# The policy reads the cameras + joint states and drives the follower to do
# the task. No leader arm needed here.
#
# SAFETY: keep a hand near the power switch the first time.
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

# No --rename_map needed here (unlike ball_pickup_pi0): config.env's CAMERAS
# already uses pi0's own base_0_rgb/left_wrist_0_rgb names directly, so the
# robot's camera feature names match the checkpoint's declared schema as-is.
#
# --policy.use_peft=true IS required though -- this checkpoint is a LoRA
# adapter, not a full fine-tune. Without it, make_policy tries to load the
# adapter folder as a regular full checkpoint and fails (see
# lerobot/policies/factory.py: the `cfg.pretrained_path and cfg.use_peft`
# branch is what triggers PeftConfig.from_pretrained + PeftModel.from_pretrained).
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --task="$TASK" \
  --policy.path="$POLICY_PATH" \
  --policy.use_peft=true \
  --policy.device=cuda \
  --inference.type=rtc \
  --display_data=true

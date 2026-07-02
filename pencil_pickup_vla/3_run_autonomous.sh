#!/usr/bin/env bash
# Stage 4 — run the fine-tuned SmolVLA policy on the real SO-101, autonomously (no leader).
# The policy reads the camera + joint states and drives the follower to do the task.
#
# IMPORTANT: use the SAME camera position/index and the SAME task string as during recording.
# Keep a hand near the power switch the first time.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
source ../.venv/bin/activate

# On Apple Silicon use the GPU via MPS (falls back to CPU). Inference may be slow without CUDA.
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="{ front: {type: opencv, index_or_path: $CAMERA_INDEX, width: $CAMERA_W, height: $CAMERA_H, fps: $CAMERA_FPS}}" \
  --task="$TASK" \
  --policy.path="$MODEL_REPO" \
  --policy.device="$POLICY_DEVICE" \
  --inference.type=rtc \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --display_data=true

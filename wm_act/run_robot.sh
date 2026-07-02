#!/usr/bin/env bash
# Run a trained WM-ACT policy autonomously on the follower arm.
# Usage: ./run_robot.sh [model_repo_or_local_checkpoint_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
source pencil_pickup_vla/config.env

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${1:-${HF_USER}/wmact_pencil_pickup}"

# Notes vs 3_run_autonomous.sh (SmolVLA):
#  - no --rename_map: WM-ACT is trained natively on the `front` camera name.
#  - --inference.type=sync: a 52M-param model runs well inside the 33ms budget
#    at 30Hz on MPS, no RTC needed.
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
  --robot.cameras="{ front: {type: opencv, index_or_path: $CAMERA_INDEX, width: $CAMERA_W, height: $CAMERA_H, fps: $CAMERA_FPS}}" \
  --task="$TASK" \
  --policy.discover_packages_path=wm_act \
  --policy.path="$MODEL" \
  --policy.device="$POLICY_DEVICE" \
  --inference.type=sync \
  --display_data=true

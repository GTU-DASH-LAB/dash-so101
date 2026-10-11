#!/usr/bin/env bash
# ============================================================================
# Stage 3 — run the fine-tuned pi0.5 LoRA policy on the real SO-101 to fold
# the t-shirt.
#
# SAFETY: this moves a 6-DOF arm autonomously. Stand next to the machine with
# a hand near the power switch the first time, and keep the workspace clear.
# Ctrl-C stops it.
#
#   ./3_run_autonomous.sh              # newest checkpoint (step 20,000)
#   ./3_run_autonomous.sh 010000       # a specific checkpoint, to compare
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
export PATH="$VENV_BIN:$PATH"

CKPT="${1:-last}"
POLICY_PATH="$OUTPUT_DIR/checkpoints/$CKPT/pretrained_model"
if [ ! -d "$POLICY_PATH" ]; then
  echo "ERROR: no checkpoint at '$POLICY_PATH'." >&2
  echo "Available: $(ls "$OUTPUT_DIR/checkpoints" 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi
# The LoRA adapter references the base model by absolute path; if that ever
# moves, PEFT would try to reach the Hub for it. Fail loudly instead.
if [ ! -d "$BASE_MODEL" ]; then
  echo "ERROR: base model '$BASE_MODEL' is missing (the adapter needs it)." >&2
  exit 1
fi

# /dev/ttyACMN is enumeration order, not identity, and the two SO-101 arms swap
# freely across replugs. Driving the LEADER with follower commands would be bad,
# so resolve the port back to its USB serial and refuse to run on a mismatch.
ACTUAL_SERIAL="$(
  for link in /dev/serial/by-id/*; do
    [ "$(readlink -f "$link")" = "$(readlink -f "$FOLLOWER_PORT")" ] || continue
    printf '%s\n' "$link" | sed -n 's/.*USB_Single_Serial_\([^-]*\)-if00/\1/p'
  done
)"
if [ "$ACTUAL_SERIAL" != "$FOLLOWER_SERIAL" ]; then
  echo "ERROR: $FOLLOWER_PORT is serial '${ACTUAL_SERIAL:-<none>}', expected '$FOLLOWER_SERIAL'." >&2
  echo "The arms have been replugged and the ports swapped. Current mapping:" >&2
  ls -l /dev/serial/by-id/ >&2
  echo "Update FOLLOWER_PORT in config.env to the node holding $FOLLOWER_SERIAL." >&2
  exit 1
fi

export HF_HUB_OFFLINE=1

echo "Policy:  $POLICY_PATH"
echo "Robot:   $FOLLOWER_PORT (id=$FOLLOWER_ID)   clamp=$MAX_RELATIVE_TARGET deg/step"
echo "Task:    $TASK"
echo
echo "Arm will move on its own. Hand near the power switch. Ctrl-C to stop."
echo

# --policy.use_peft=true IS correct here (unlike at training time, where
# --peft.method_type=LORA was the real gate): we are loading an EXISTING
# adapter, which is exactly the branch that flag selects in factory.py.
#
# --rename_map is required for the same reason it was during training, but
# note the direction of the problem is different. rename_map only ever
# remapped the DATASET's column names; the checkpoint itself permanently
# declares pi0.5's own slots (base_0_rgb / left_wrist_0_rgb /
# right_wrist_0_rgb). So the robot's cameras are named exactly as they were
# when recording (workspace_cam / wrist_camera) and remapped here too --
# without this, lerobot-rollout raises "Visual feature mismatch between
# policy and robot hardware".
#
# The camera block mirrors ~/.cache/huggingface/lerobot/robots/dash_so101.json
# (the lelab config that RECORDED the dataset) exactly: same by-id device
# paths, same 640x480@30, no rotation, no fourcc override. The policy must see
# the world the way it was trained. by-id paths are used rather than
# /dev/videoN because the N shifts whenever a camera is replugged.
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.max_relative_target="$MAX_RELATIVE_TARGET" \
  --robot.cameras="$CAMERAS" \
  --rename_map="$RENAME_MAP" \
  --task="$TASK" \
  --policy.path="$POLICY_PATH" \
  --policy.use_peft=true \
  --policy.device=cuda \
  --inference.type=rtc \
  --display_data=true

#!/usr/bin/env bash
# ============================================================================
# Step 3 - run FLUX 3 Action on the real arm, on this PC's RTX 5090 (leader not needed).
#
#   ./3_run_autonomous.sh                 newest checkpoint pulled from AWS, until Ctrl+C
#   ./3_run_autonomous.sh 006000          a specific checkpoint (names: --list)
#   ./3_run_autonomous.sh hub             the model published by ./2b_publish_model.sh
#   ./3_run_autonomous.sh init            NOT fine-tuned: the SO-101 base checkpoint, wrist only
#                                         (pipeline test; it never saw this task or camera setup)
#   ./3_run_autonomous.sh --list          checkpoints available locally
#   ./3_run_autonomous.sh --trials 10 [checkpoint]
#        10 timed attempts of TRIAL_TIME_S each. Between attempts the arm returns
#        to its start pose and you get TRIAL_RESET_S to put the pen somewhere new.
#        Keys: -> end attempt early, <- redo attempt, Esc stop. Every attempt is
#        saved locally (run/rollouts/) so you can count successes afterwards.
#
# Needs once: ./local_weights.py (base weights into the HF cache + init_policy).
#
# SAFETY: the arm moves on its own. Clear the workspace, keep a hand near the
# power switch the first time. Each joint is limited to MAX_RELATIVE_TARGET
# degrees per control tick (config.env). Ctrl+C stops; the arm then returns to
# the pose it started from before the motors relax.
#
# FLUX 3 predicts 32 actions (~1 s) per call and has no RTC yet, so the arm can
# pause briefly while the next chunk is computed.
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"

TRIALS=0
CKPT=last
CK_ROOT="$OUTPUT_DIR/checkpoints"
while [ $# -gt 0 ]; do
  case "$1" in
    --list) ls "$CK_ROOT" 2>/dev/null | grep -v '^last$' || echo "No checkpoints yet (./aws_train.sh pull)"; exit 0 ;;
    --trials) TRIALS="${2:?--trials needs a number}"; shift 2 ;;
    -h|--help) sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) CKPT="$1"; shift ;;
  esac
done

[ -d init_policy ] || die "Run ./local_weights.py first (FLUX 3 base weights for this PC)."
case "$CKPT" in
  hub)  POLICY="$MODEL_REPO" ;;
  init) POLICY="$HERE/init_policy" ;;
  *)    [ -d "$CK_ROOT/$CKPT" ] || die "No checkpoint '$CKPT'. Available: $(ls "$CK_ROOT" 2>/dev/null | tr '\n' ' ')"
        POLICY="$RUN_DIR/deploy/$(basename "$(readlink -f "$CK_ROOT/$CKPT")")"
        "$PY" deploy_config.py "$CK_ROOT/$CKPT" "$POLICY" || exit 1 ;;
esac
recorder_running && die "A recording is running. Only one program can drive the arm."
rollout_running && die "Another ./3_run_autonomous.sh is already running."

"$PY" hw_check.py --no-leader --follower-serial "$FOLLOWER_SERIAL" --follower-id "$FOLLOWER_ID" \
  --cameras "$CAMERAS" > "$RUN_DIR/hw_check.out" 2>&1
rc=$?
grep -v '^PORTS ' "$RUN_DIR/hw_check.out"
[ $rc -eq 0 ] || die "Hardware check failed - fix the FAIL lines above and run again."
FOLLOWER_PORT="$(sed -n 's/^PORTS.*follower=\([^ ]*\).*/\1/p' "$RUN_DIR/hw_check.out")"

COMMON=(
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID"
  --robot.max_relative_target="$MAX_RELATIVE_TARGET" --robot.cameras="$CAMERAS"
  --policy.path="$POLICY" --policy.device=cuda --policy.n_action_steps="$N_ACTION_STEPS"
  --task="$TASK" --fps="$FPS" --display_data=true --play_sounds=false
)
[ "$CKPT" = init ] || COMMON+=(--policy.use_peft=true)

echo
info "Policy:  $POLICY"
info "Actions: executes $N_ACTION_STEPS of each 32 predicted actions, max $MAX_RELATIVE_TARGET deg/tick per joint"
warn "The arm will move by itself in a few seconds. Hand near the power switch. Ctrl+C stops."
echo

# Base weights, text encoder and video VAE all come from the local cache (local_weights.py).
[ "$POLICY" = "$MODEL_REPO" ] || export HF_HUB_OFFLINE=1
if [ "$TRIALS" -gt 0 ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  ROOT="$RUN_DIR/rollouts/$STAMP"
  mkdir -p "$(dirname "$ROOT")"
  lerobot-rollout "${COMMON[@]}" --strategy.type=episodic \
    --dataset.repo_id="${HF_USER}/rollout_pen_bowl_flux3_${STAMP}" --dataset.root="$ROOT" \
    --dataset.single_task="$TASK" --dataset.num_episodes="$TRIALS" \
    --dataset.episode_time_s="$TRIAL_TIME_S" --dataset.reset_time_s="$TRIAL_RESET_S" \
    --dataset.push_to_hub=false --dataset.rgb_encoder.vcodec=h264
  rc=$?
  echo
  echo "Attempts saved in $ROOT (videos/ holds the wrist-camera video of every attempt)"
  exit $rc
else
  exec lerobot-rollout "${COMMON[@]}" --strategy.type=base --duration="$RUN_DURATION_S"
fi

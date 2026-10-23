#!/usr/bin/env bash
# ============================================================================
# Step 3 - run the fine-tuned MolmoAct2 policy on the real arm (leader not needed).
#
#   ./3_run_autonomous.sh                 newest checkpoint, runs until Ctrl+C
#   ./3_run_autonomous.sh best            checkpoint with the lowest validation loss
#   ./3_run_autonomous.sh 013830          a specific checkpoint (names: --list)
#   ./3_run_autonomous.sh hub             the model published by ./2b_publish_model.sh
#   ./3_run_autonomous.sh --list          checkpoints with their epoch and val loss
#   ./3_run_autonomous.sh --rtc [checkpoint]
#        Real-Time Chunking: the next chunk is computed in a background thread while
#        the arm keeps executing the current one, and blended into it. Without it
#        the arm pauses ~0.13 s at every chunk boundary while MolmoAct2 thinks (the
#        "Record loop is running slower (7.5 Hz)" warning is that one slow tick).
#   ./3_run_autonomous.sh --trials 10 [checkpoint]
#        10 timed attempts of TRIAL_TIME_S each. Between attempts the arm returns
#        to its start pose and you get TRIAL_RESET_S to put the pen somewhere new.
#        Keys: -> end attempt early, <- redo attempt, Esc stop. Every attempt is
#        saved locally (run/rollouts/) so you can count successes afterwards.
#   ./3_run_autonomous.sh --trt [other options]
#        Same run, with MolmoAct2's action expert as a TensorRT engine (../tools/trt):
#        a new chunk takes ~88 ms instead of ~127 ms, the other ticks ~0.3 ms instead
#        of ~11 ms. First run builds the engine (~80 s, before the arm connects).
#        With --rtc the expert stays in PyTorch (RTC needs its gradients).
#
# SAFETY: the arm moves on its own. Clear the workspace, keep a hand near the
# power switch the first time. Each joint is limited to MAX_RELATIVE_TARGET
# degrees per control tick (config.env). Ctrl+C stops; the arm then returns to
# the pose it started from before the motors relax.
#
# Start pose: put the follower where you started your demos (pen visible in the
# wrist camera). The policy only sees the wrist camera, so its start view matters.
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"

TRIALS=0
TRT=false
RTC=false
CKPT=last
while [ $# -gt 0 ]; do
  case "$1" in
    --list) exec "$PY" train_dashboard.py --status ;;
    --trials) TRIALS="${2:?--trials needs a number}"; shift 2 ;;
    --rtc) RTC=true; shift ;;
    --trt) TRT=true; shift ;;
    -h|--help) sed -n '3,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) CKPT="$1"; shift ;;
  esac
done

CK_ROOT="$OUTPUT_DIR/checkpoints"
if [ "$CKPT" = "best" ]; then
  CKPT=$("$PY" -c 'import train_dashboard as t; b=[c for c in t.collect()["checkpoints"] if c.get("best")]; print(b[0]["name"] if b else "")')
  [ -n "$CKPT" ] || die "No checkpoint has a validation loss yet. Use 'last' or a name from --list."
fi
if [ "$CKPT" = "hub" ]; then
  POLICY="$MODEL_REPO"            # published with the joint transform already ON
  # The published processors name the base by its Hub id, and MolmoAct2's input processor
  # snapshot-downloads it (21.8 GB) - --policy.checkpoint_path below only reaches the model.
  warn "hub: first use downloads allenai/MolmoAct2-SO100_101 (21.8 GB). The local checkpoint"
  warn "that was published is the same model: ./3_run_autonomous.sh <its step> (see --list)."
  confirm "Continue with the Hub model?" || exit 1
else
  SRC="$CK_ROOT/$CKPT/pretrained_model"
  [ -f "$SRC/config.json" ] || die "No checkpoint '$CKPT'. Available: $(ls "$CK_ROOT" 2>/dev/null | tr '\n' ' ')"
  # Trained in MolmoAct2's joint convention; the deployable copy converts arm <-> model frame.
  POLICY="$RUN_DIR/deploy/$(basename "$(readlink -f "$CK_ROOT/$CKPT")")"
  JOINT_SIGNS="$JOINT_SIGNS" JOINT_OFFSETS="$JOINT_OFFSETS" "$PY" deploy_config.py "$SRC" "$POLICY" || exit 1
fi
recorder_running && die "A recording is running. Only one program can drive the arm."
rollout_running && die "Another ./3_run_autonomous.sh is already running."

"$PY" hw_check.py --no-leader --follower-serial "$FOLLOWER_SERIAL" --follower-id "$FOLLOWER_ID" \
  --cameras "$CAMERAS" > "$RUN_DIR/hw_check.out" 2>&1
rc=$?
grep -v '^PORTS ' "$RUN_DIR/hw_check.out"
[ $rc -eq 0 ] || die "Hardware check failed - fix the FAIL lines above and run again."
FOLLOWER_PORT="$(sed -n 's/^PORTS.*follower=\([^ ]*\).*/\1/p' "$RUN_DIR/hw_check.out")"

if [ "$RTC" = true ]; then
  # A new chunk is started when RTC_QUEUE_THRESHOLD actions are left; it takes ~9 ticks
  # (284 ms with RTC guidance, measured on the 5090), so the threshold must stay above that.
  MODE_ARGS=(--inference.type=rtc --inference.queue_threshold="$RTC_QUEUE_THRESHOLD"
             --inference.rtc.execution_horizon="$RTC_EXECUTION_HORIZON")
  MODE_TXT="RTC: next chunk computed in the background when $RTC_QUEUE_THRESHOLD of $CHUNK_SIZE actions are left"
else
  MODE_ARGS=(--policy.n_action_steps="$N_ACTION_STEPS")
  # MolmoAct2 replaces n_action_steps with its checkpoint metadata (30), so every run so far
  # executed the whole chunk; config.env now says 30 to match (rollout_trt.py reports it).
  MODE_TXT="executes the whole $CHUNK_SIZE-action chunk, then predicts again"
fi
# The policy rebuilds itself from the base checkpoint first; use the local download
# (a published model names the Hub id, which would mean another 21.8 GB download).
MODE_ARGS+=(--policy.checkpoint_path="$BASE_CHECKPOINT")

if [ "$RTC" = true ]; then INFER_TXT="TensorRT not used for the expert under --rtc (PyTorch)"
else INFER_TXT="TensorRT action expert (built on first use, then cached)"; fi

COMMON=(
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID"
  --robot.max_relative_target="$MAX_RELATIVE_TARGET" --robot.cameras="$CAMERAS"
  --policy.path="$POLICY" --policy.device=cuda "${MODE_ARGS[@]}"
  --task="$TASK" --fps="$FPS" --display_data=true --play_sounds=false
)

echo
info "Policy:  $POLICY"
info "Actions: $MODE_TXT, max $MAX_RELATIVE_TARGET deg/tick per joint"
[ "$TRT" = true ] && info "Inference: $INFER_TXT"
warn "The arm will move by itself in a few seconds. Hand near the power switch. Ctrl+C stops."
echo

if [ "$TRT" = true ]; then
  # Same lerobot-rollout, same robot code and limits; the network runs as a TensorRT engine.
  ROLLOUT=("$PY" "$(cd "$HERE/../tools/trt" && pwd)/rollout_trt.py")
else
  ROLLOUT=(lerobot-rollout)
fi
[ "$POLICY" = "$MODEL_REPO" ] || export HF_HUB_OFFLINE=1
if [ "$TRIALS" -gt 0 ]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  ROOT="$RUN_DIR/rollouts/$STAMP"
  mkdir -p "$(dirname "$ROOT")"
  "${ROLLOUT[@]}" "${COMMON[@]}" --strategy.type=episodic \
    --dataset.repo_id="${HF_USER}/rollout_pen_bowl_molmoact2_${STAMP}" --dataset.root="$ROOT" \
    --dataset.single_task="$TASK" --dataset.num_episodes="$TRIALS" \
    --dataset.episode_time_s="$TRIAL_TIME_S" --dataset.reset_time_s="$TRIAL_RESET_S" \
    --dataset.push_to_hub=false --dataset.rgb_encoder.vcodec=h264
  rc=$?
  echo
  echo "Attempts saved in $ROOT (videos/ holds the wrist-camera video of every attempt)"
  exit $rc
else
  exec "${ROLLOUT[@]}" "${COMMON[@]}" --strategy.type=base --duration="$RUN_DURATION_S"
fi

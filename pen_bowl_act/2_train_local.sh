#!/usr/bin/env bash
# ============================================================================
# Step 2 - train ACT on both RTX 5090s, with a live dashboard.
#
#   ./2_train_local.sh            start (or resume an interrupted run), opens the dashboard
#   ./2_train_local.sh status     progress, losses, remaining time - in the terminal
#   ./2_train_local.sh dashboard  (re)open the dashboard (e.g. after a reboot)
#   ./2_train_local.sh stop       stop training (the newest checkpoint is kept;
#                                 ./2_train_local.sh resumes from it)
#   ./2_train_local.sh --fresh    throw the current run away (moved aside, not deleted)
#                                 and start over, e.g. after changing config.env
#
# Training runs detached: closing this terminal does not stop it. If it crashes,
# the watchdog resumes it from the last checkpoint by itself.
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"
export HERE RUN_DIR DATASET_ROOT VALIDATION_SPLIT NUM_GPUS BATCH_SIZE EPOCHS CHECKPOINT_EVERY_EPOCHS \
  LR LR_BACKBONE MIXED_PRECISION AUGMENT THROUGHPUT_HINT
URL="http://${DASHBOARD_HOST}:${DASHBOARD_PORT}"

alive() { [ -f "$RUN_DIR/$1" ] && kill -0 "$(cat "$RUN_DIR/$1")" 2>/dev/null; }

start_dashboard() {
  if ! alive dashboard.pid; then
    setsid nohup "$PY" train_dashboard.py --serve </dev/null >"$RUN_DIR/dashboard.log" 2>&1 &
    echo $! > "$RUN_DIR/dashboard.pid"
    sleep 1
    alive dashboard.pid || { err "Dashboard failed to start: $(tail -2 "$RUN_DIR/dashboard.log")"; return 1; }
  fi
  ok "Dashboard: $URL"
  [ -n "${DISPLAY:-}" ] && xdg-open "$URL" >/dev/null 2>&1 &
  return 0
}

stop_training() {
  if alive watchdog.pid; then
    kill -TERM "$(cat "$RUN_DIR/watchdog.pid")"
    info "Stopping training (waiting for the GPUs to be released) ..."
    for _ in $(seq 40); do alive watchdog.pid || break; sleep 1; done
  fi
  # Belt and braces: anything left in training's own process group.
  if [ -f "$RUN_DIR/train.pgid" ]; then kill -KILL -- "-$(cat "$RUN_DIR/train.pgid")" 2>/dev/null; fi
  rm -f "$RUN_DIR/watchdog.pid" "$RUN_DIR/train.pgid"
  ok "Training stopped. The newest checkpoint is kept; ./2_train_local.sh resumes from it."
}

case "${1:-start}" in
  status)    exec "$PY" train_dashboard.py --status ;;
  dashboard) start_dashboard; exit $? ;;
  stop)
    stop_training
    if alive dashboard.pid; then kill "$(cat "$RUN_DIR/dashboard.pid")"; rm -f "$RUN_DIR/dashboard.pid"; fi
    exit 0 ;;
  start|--fresh) ;;
  *) sed -n '3,14p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac

alive watchdog.pid && { info "Training is already running."; "$PY" train_dashboard.py --status; start_dashboard; exit 0; }
recorder_running && die "A recording is running. Training would starve the arm's serial bus of CPU; finish recording first."
[ -f "$DATASET_ROOT/meta/info.json" ] || die "No merged dataset at $DATASET_ROOT. Run ./1b_publish_dataset.sh first."

if [ "${1:-}" = "--fresh" ] && [ -d "$OUTPUT_DIR" ]; then
  confirm "Move the current run ($OUTPUT_DIR) aside and start from scratch?" || exit 1
  mv "$OUTPUT_DIR" "${OUTPUT_DIR}.old.$(date +%Y%m%d_%H%M%S)"
  info "Old run moved to ${OUTPUT_DIR}.old.*"
fi

if [ -f "$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json" ]; then
  info "Resuming the existing run from its newest checkpoint (use --fresh to start over)."
else
  "$PY" train_plan.py || exit 1
  : > "$RUN_DIR/train.log"
  : > "$RUN_DIR/watchdog.log"
fi

# Both GPUs must be free: another job on them would OOM us or slow both down.
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 > 3000 {print "GPU" $1 " (" $2 " MiB used)"}')
if [ -n "$busy" ]; then
  warn "Already in use: $busy"
  confirm "Start training anyway?" || exit 1
fi

setsid nohup ./watchdog_train.sh </dev/null >>"$RUN_DIR/watchdog.log" 2>&1 &
echo $! > "$RUN_DIR/watchdog.pid"
ok "Training started on $NUM_GPUS GPU(s), detached (closing this terminal is fine)."
start_dashboard
echo
echo "   Terminal view:   ./2_train_local.sh status"
echo "   Stop:            ./2_train_local.sh stop"
echo "   Raw log:         tail -f run/train.log"

#!/usr/bin/env bash
# ============================================================================
# Step 1 - record demonstrations: move the LEADER arm, the follower copies it,
# and the wrist camera + joint positions are saved.
#
#   ./1_record.sh            record (continues where you left off, any time)
#   ./1_record.sh --check    only run the hardware check, record nothing
#
# Keys while recording (work in any window - keep the TERMINAL focused, because
# arrow keys inside the rerun viewer also scrub its timeline):
#   ->   demo finished: save it            (during the reset pause: start next now)
#   <-   redo: throw away the current/just-finished demo
#   Esc  stop the session (a demo in progress is thrown away, finished ones kept)
#
# Recording happens in sessions of EPISODES_PER_SESSION demos (config.env). Each
# session is its own small dataset, so an accident only ever affects one session.
# If a cable comes out or an arm loses power, finished demos are safe: plug it
# back in and press Enter, and recording continues in the same session.
# Review / delete demos or whole sessions with ./manage_data.sh
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"
export HERE SESSIONS_DIR FPS MIN_EPISODE_S TARGET_EPISODES

if recorder_running; then
  die "A recording is already running (another terminal?). Only one at a time."
fi

hw_check() {
  "$PY" hw_check.py --leader-serial "$LEADER_SERIAL" --follower-serial "$FOLLOWER_SERIAL" \
    --leader-id "$LEADER_ID" --follower-id "$FOLLOWER_ID" --cameras "$CAMERAS" \
    --snapshot "$HERE/run/last_camera_check.png" > "$RUN_DIR/hw_check.out" 2>&1
  local rc=$?
  grep -v '^PORTS ' "$RUN_DIR/hw_check.out"
  LEADER_PORT="$(sed -n 's/^PORTS.*leader=\([^ ]*\).*/\1/p' "$RUN_DIR/hw_check.out")"
  FOLLOWER_PORT="$(sed -n 's/^PORTS.*follower=\([^ ]*\).*/\1/p' "$RUN_DIR/hw_check.out")"
  return $rc
}

# Wait until the hardware passes the check again (after an unplug), or the user quits.
wait_for_hardware() {
  while :; do
    echo
    read -r -p "Fix it (plug the cable back in / switch the arm power on), then press Enter to re-check, or q to quit: " reply
    [[ "$reply" =~ ^[Qq] ]] && return 1
    sleep 2   # give udev a moment to recreate /dev/serial/by-id and /dev/v4l/by-id links
    if hw_check; then return 0; fi
  done
}

if [ "${1:-}" = "--check" ]; then
  hw_check; exit $?
fi

df_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
[ "${df_gb:-0}" -lt 10 ] && die "Only ${df_gb} GB free on disk. Free some space first."

info "Hardware check"
if ! hw_check; then
  err "Hardware check failed - see the FAIL lines above."
  wait_for_hardware || exit 1
fi

while :; do
  SESSION=
  read -r SESSION SESSION_TARGET TOTAL_BEFORE < <("$PY" sessions.py next --per-session "$EPISODES_PER_SESSION" --target "$TARGET_EPISODES")
  [ -n "$SESSION" ] || die "Could not work out the next session (sessions.py next failed)."
  case "$SESSION" in
    DONE)
      ok "All $TARGET_EPISODES demos are recorded."
      echo "Next: ./manage_data.sh status   (review), then ./1b_publish_dataset.sh"
      exit 0 ;;
    BROKEN)
      err "The last session ($SESSION_TARGET) was damaged by a hard crash and cannot be continued."
      echo "Move it to the trash, then run ./1_record.sh again:"
      echo "    ./manage_data.sh delete-session $SESSION_TARGET"
      exit 1 ;;
  esac
  SESSION_NUM=${SESSION#session_}

  echo
  info "Session ${SESSION}: recording up to ${SESSION_TARGET} demos (you have ${TOTAL_BEFORE}/${TARGET_EPISODES} in earlier sessions)"
  echo "   The rerun window shows the wrist camera. Keep the TERMINAL focused for the arrow keys."
  echo "   Start every demo from the same rest/start pose, with the pen visible in the wrist camera."
  echo

  "$PY" -u record_session.py \
    --session-dir "$SESSIONS_DIR/$SESSION" \
    --repo-id "${SESSION_REPO_PREFIX}_${SESSION_NUM}" \
    --episodes "$SESSION_TARGET" \
    --total-before "$TOTAL_BEFORE" \
    --total-target "$TARGET_EPISODES" \
    --task "$TASK" \
    --fps "$FPS" \
    --episode-time-s "$EPISODE_TIME_S" \
    --reset-time-s "$RESET_TIME_S" \
    --min-episode-s "$MIN_EPISODE_S" \
    --follower-port "$FOLLOWER_PORT" --follower-id "$FOLLOWER_ID" \
    --leader-port "$LEADER_PORT" --leader-id "$LEADER_ID" \
    --cameras "$CAMERAS" \
    --vcodec "$VIDEO_CODEC" --crf "$VIDEO_CRF" \
    --display
  rc=$?

  case $rc in
    0)
      echo
      ok "Session $SESSION complete."
      "$PY" sessions.py status
      read -r SESSION SESSION_TARGET TOTAL_BEFORE < <("$PY" sessions.py next --per-session "$EPISODES_PER_SESSION" --target "$TARGET_EPISODES")
      [ "$SESSION" = "DONE" ] && continue
      echo
      echo "Take a break if you like - this is a safe point. Review the demos you just made with:"
      echo "    ./manage_data.sh review $SESSION_NUM"
      read -r -p "Press Enter to start the next session now, or q to quit: " reply
      [[ "$reply" =~ ^[Qq] ]] && exit 0
      hw_check || wait_for_hardware || exit 1
      ;;
    2|130)
      echo
      ok "Stopped. Everything finished so far is saved. Run ./1_record.sh again to continue."
      "$PY" sessions.py status
      exit 0
      ;;
    3)
      echo
      warn "Recording stopped because of a hardware problem. Finished demos are safe on disk."
      hw_check
      wait_for_hardware || { echo "Run ./1_record.sh again later; it continues this session."; exit 0; }
      ;;
    *)
      err "The recorder exited with code $rc. Details: $SESSIONS_DIR/$SESSION/record.log"
      echo "Finished demos are kept. Check the data with ./manage_data.sh check, then run ./1_record.sh again."
      exit $rc
      ;;
  esac
done

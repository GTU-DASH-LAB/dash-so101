#!/usr/bin/env bash
# ============================================================================
# Look after the recorded demos: progress, review, delete, undo.
#
#   ./manage_data.sh                         progress: demos per session, total/100
#   ./manage_data.sh episodes 3              list the demos in session 3 with lengths
#   ./manage_data.sh review 3                cut every demo of session 3 into its own
#                                            video (review/session_03/) and open the folder
#   ./manage_data.sh review 3 7              open just demo 7 of session 3
#   ./manage_data.sh delete-episodes 3 7 9   remove demos 7 and 9 from session 3
#   ./manage_data.sh delete-session 3        remove the whole session 3
#   ./manage_data.sh trash                   list what was deleted
#   ./manage_data.sh restore <name>          undo a delete
#   ./manage_data.sh empty-trash             free the disk space for good
#   ./manage_data.sh check                   decode every demo; finds damaged sessions
#
# Deleting never destroys anything: data goes to a trash folder until you run
# empty-trash. Demo numbers inside a session are renumbered after a delete, so
# run `episodes` again before deleting more.
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"
export HERE SESSIONS_DIR FPS MIN_EPISODE_S TARGET_EPISODES

cmd="${1:-status}"
[ $# -gt 0 ] && shift

case "$cmd" in
  help|-h|--help)
    sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 0 ;;
  delete-episodes|delete-session|restore|empty-trash)
    if recorder_running; then
      die "A recording is running. Stop it (Esc) before changing the data."
    fi ;;
esac

export HF_HUB_OFFLINE=1   # everything here is local; never touch the Hub
exec "$PY" sessions.py "$cmd" "$@"

#!/usr/bin/env bash
# ============================================================================
# Step 0 - practice teleoperation before recording. Nothing is saved.
#   ./0_teleoperate.sh
# Move the leader arm, the follower copies it; the rerun window shows the wrist
# camera (this is ALL the policy will ever see - practise keeping the pen and
# the bowl in that view). Ctrl+C -> 5 s to lower the arm -> motors relax.
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"

recorder_running && die "A recording is running. Only one program can use the arms."

"$PY" hw_check.py --leader-serial "$LEADER_SERIAL" --follower-serial "$FOLLOWER_SERIAL" \
  --leader-id "$LEADER_ID" --follower-id "$FOLLOWER_ID" --cameras "$CAMERAS" > "$RUN_DIR/hw_check.out" 2>&1
rc=$?
grep -v '^PORTS ' "$RUN_DIR/hw_check.out"
[ $rc -eq 0 ] || die "Hardware check failed - fix the FAIL lines above and run again."
LEADER_PORT="$(sed -n 's/^PORTS.*leader=\([^ ]*\).*/\1/p' "$RUN_DIR/hw_check.out")"
FOLLOWER_PORT="$(sed -n 's/^PORTS.*follower=\([^ ]*\).*/\1/p' "$RUN_DIR/hw_check.out")"

exec "$PY" -u teleop_practice.py \
  --follower-port "$FOLLOWER_PORT" --follower-id "$FOLLOWER_ID" \
  --leader-port "$LEADER_PORT" --leader-id "$LEADER_ID" \
  --cameras "$CAMERAS" --fps "$FPS"

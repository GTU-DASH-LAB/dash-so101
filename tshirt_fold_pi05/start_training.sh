#!/usr/bin/env bash
# ============================================================================
# One-shot launcher: starts the watchdog-wrapped training run AND the
# 6-hourly email monitor, both fully detached (setsid) so they survive a
# closed terminal, an SSH disconnect, or the agent session that started them
# going away. This is what you actually run.
#
#   ./start_training.sh          # launch both
#   ./stop_training.sh           # stop both
#
# Check on it later with:
#   tail -f watchdog.log         # restarts / lifecycle
#   tail -f train.log            # raw lerobot-train output (loss lines)
#   tail -f monitor.log          # email monitor's own log
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env

if [ -f ./training.pid ] && kill -0 "$(cat ./training.pid)" 2>/dev/null; then
  echo "ERROR: training already running (PID $(cat ./training.pid)). Use ./stop_training.sh first." >&2
  exit 1
fi

# The watchdog appends; start each run's logs fresh so the monitor's ETA math
# (which uses watchdog.log's mtime as t0) isn't skewed by a previous run.
: > ./watchdog.log
: > ./train.log

setsid nohup ./watchdog_train.sh </dev/null >/dev/null 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > ./training.pid
echo "Training (watchdog) started, PID $TRAIN_PID"

# Give the watchdog a moment to create watchdog.log so the monitor's t0 is sane.
sleep 5

setsid nohup "$VENV_BIN/python" ./monitor_and_email.py </dev/null >/dev/null 2>&1 &
MON_PID=$!
echo "$MON_PID" > ./monitor.pid
echo "Email monitor started, PID $MON_PID (emails every 6h to the configured address)"

echo
echo "Both detached. Follow along with:  tail -f $(pwd)/train.log"

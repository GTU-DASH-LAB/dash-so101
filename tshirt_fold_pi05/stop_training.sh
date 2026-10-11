#!/usr/bin/env bash
# ============================================================================
# Stop the detached training run and its email monitor.
#
# Kills by recorded PID and by an explicit `ps` scan for this job's own
# processes -- deliberately NOT `pkill -f`, which matches the invoking
# shell's own command line and has already caused self-kills on this machine
# (see robotics/CLAUDE.md "Shell hygiene").
#
# Stopping mid-run is safe: the newest checkpoint under
# outputs/train/pi05_tshirt_fold/checkpoints/ stays intact, and re-running
# ./start_training.sh resumes from it.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

stop_pidfile() {
  local file="$1" label="$2"
  if [ -f "$file" ]; then
    local pid
    pid="$(cat "$file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null && echo "Stopped $label (PID $pid)"
    else
      echo "$label (PID $pid) was not running"
    fi
    rm -f "$file"
  else
    echo "No $file — $label not tracked"
  fi
}

stop_pidfile ./monitor.pid "email monitor"
stop_pidfile ./training.pid "training watchdog"

# Reap the actual worker processes the watchdog spawned (accelerate launcher +
# its two ranks + dataloader workers). Exact-match on args, excluding this
# script's own shell.
sleep 2
MINE=$$
for pid in $(ps -eo pid=,args= | awk -v me="$MINE" '$1 != me && /lerobot\.scripts\.lerobot_train/ && !/awk/ {print $1}'); do
  kill -9 "$pid" 2>/dev/null && echo "Killed leftover training process $pid"
done

echo "Done. Newest checkpoint is preserved; ./start_training.sh will resume from it."

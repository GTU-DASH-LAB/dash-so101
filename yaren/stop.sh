#!/usr/bin/env bash
# ============================================================================
# Stop Yaren.  ./stop.sh [--keep-mind]
# The voice server stops first (no new commands), then the robot server gets Ctrl+C: like
# lerobot-rollout, it returns the arm to its start pose before disconnecting. Then the Ollama
# instance stops too, freeing GPU1 (--keep-mind leaves it loaded for a faster next start).
# Processes are found through run/*.pid and checked by command line - never pkill -f.
# ============================================================================
set -uo pipefail
YAREN="$(cd "$(dirname "$0")" && pwd)"
RUN="$YAREN/run"
KEEP_MIND=false
[ "${1:-}" = "--keep-mind" ] && KEEP_MIND=true

stop_one() {  # name signal seconds cmdline-pattern
  local f="$RUN/$1.pid" pid i
  [ -f "$f" ] || return 0
  pid="$(cat "$f")"
  if ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; return 0; fi
  if ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q -- "$4"; then
    echo "!! $1: pid $pid is not ours any more ($(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-80)), leaving it"
    rm -f "$f"; return 0
  fi
  echo ":: stopping $1 (pid $pid)"
  kill "-$2" "$pid"
  for i in $(seq 1 $(( $3 * 4 ))); do
    kill -0 "$pid" 2>/dev/null || { rm -f "$f"; echo "OK $1 stopped"; return 0; }
    sleep 0.25
  done
  echo "!! $1 did not stop in $3 s - killing it"
  kill -9 "$pid" 2>/dev/null
  rm -f "$f"
}

stop_one voice TERM 10 "agent.server"
stop_one robot INT 25 "robot_server.py"
[ "$KEEP_MIND" = true ] || stop_one ollama TERM 10 "ollama serve"
printf '%s\n' "$(date '+%Y-%m-%d %H:%M') [yaren] stopped" >> "$HOME/robotics/logs/setup-progress.log"

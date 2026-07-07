#!/usr/bin/env bash
# ============================================================================
# Stage 4 (optional) — persistent web UI for running the trained pi0 policy.
#
# Unlike 3_run_autonomous.sh, this loads the policy and connects to the
# robot/cameras ONCE and keeps them resident, then serves a local dashboard
# with live camera views and Start/Pause/Complete/Manual/Reset controls --
# so restarting after a camera hiccup or wanting to pause and take over
# manually doesn't cost another ~80s policy reload each time.
#
# Uses lerobot's real RTC (Real-Time Chunking) inference engine, matching
# 3_run_autonomous.sh's --inference.type=rtc exactly -- same model, same
# chunk smoothing, no separate --robot.max_relative_target clamp (that script
# doesn't set one either; RTC's own smoothing is what shapes autonomous
# motion). Manual jog and Reset-to-home are UI-only features with their own
# independent, always-on step limiting instead.
#
# SAFETY: keep a hand near the power switch the first time, same as any new
# control script.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
source ../.venv/bin/activate

# fastapi/uvicorn aren't part of lerobot's own install extras.
python3 -c "import fastapi, uvicorn" 2>/dev/null || {
  echo "ERROR: fastapi/uvicorn not installed in .venv. Install with:" >&2
  echo "  pip install fastapi 'uvicorn[standard]'" >&2
  exit 1
}

case "$POLICY_SOURCE" in
  local)
    export POLICY_PATH="$OUTPUT_DIR/checkpoints/last/pretrained_model"
    if [ ! -d "$POLICY_PATH" ]; then
      echo "ERROR: no trained policy at '$POLICY_PATH'." >&2
      echo "Train first with ./2_train_local.sh, or set POLICY_SOURCE=hub in config.env to pull '$MODEL_REPO' instead." >&2
      exit 1
    fi
    export HF_HUB_OFFLINE=1
    ;;
  hub)
    export POLICY_PATH="$MODEL_REPO"
    hf auth whoami >/dev/null 2>&1 || { echo "ERROR: run 'hf auth login' first to download '$MODEL_REPO'." >&2; exit 1; }
    ;;
  *)
    echo "ERROR: POLICY_SOURCE must be 'local' or 'hub' (got '$POLICY_SOURCE')." >&2
    exit 1
    ;;
esac

export FOLLOWER_PORT FOLLOWER_ID TASK
export UI_DEVICE=cuda
export UI_FPS="$FPS"
UI_PORT="${UI_PORT:-8420}"

echo "Loading policy from: $POLICY_PATH"
echo "This takes ~60-90s the first time (one-time cost -- the whole point of this UI)."
echo "Once ready, open: http://localhost:$UI_PORT"
echo

exec uvicorn ui.server:app --host 0.0.0.0 --port "$UI_PORT"

# shellcheck shell=bash
# ============================================================================
# Shared helpers, sourced by every script in this folder. Not run directly.
# ============================================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
# shellcheck source=config.env
source "$HERE/config.env"
export PATH="$VENV_BIN:$PATH"   # lerobot main (FLUX 3 is not in lelab's 0.6.0)
PY="$VENV_BIN/python"
RUN_DIR="$HERE/run"          # training state: logs, pid files, plan
mkdir -p "$RUN_DIR"

if [ -t 1 ]; then
  C_RED=$'\e[1;31m'; C_GRN=$'\e[1;32m'; C_YEL=$'\e[1;33m'; C_BLU=$'\e[1;34m'; C_DIM=$'\e[2m'; C_OFF=$'\e[0m'
else
  C_RED=; C_GRN=; C_YEL=; C_BLU=; C_DIM=; C_OFF=
fi
info() { printf '%s\n' "${C_BLU}::${C_OFF} $*"; }
ok()   { printf '%s\n' "${C_GRN}OK${C_OFF} $*"; }
warn() { printf '%s\n' "${C_YEL}!!${C_OFF} $*"; }
err()  { printf '%s\n' "${C_RED}ERROR${C_OFF} $*" >&2; }
die()  { err "$*"; exit 1; }

# /dev/ttyACMn currently holding the USB serial adapter with this serial, or "".
port_for_serial() {
  local link
  for link in /dev/serial/by-id/*"$1"*; do
    [ -e "$link" ] && { readlink -f "$link"; return 0; }
  done
  return 1
}

# The by-id camera path out of $CAMERAS (the "wrist" entry).
camera_device() {
  CAMERAS="$CAMERAS" "$PY" -c 'import json,os; print(json.loads(os.environ["CAMERAS"])["wrist"]["index_or_path"])'
}

# Ask a yes/no question; returns 0 for yes. Default is "no".
confirm() {
  local reply
  read -r -p "$1 [y/N] " reply
  [[ "$reply" =~ ^[Yy]([Ee][Ss])?$ ]]
}

# Is another of our long-running jobs using the arms/camera? (exact args match,
# never `pgrep -f`, which matches the calling shell itself — see CLAUDE.md).
recorder_running() {
  ps -eo args= | grep -Eq '^[^ ]*python[^ ]*( -[^ ]+)* [^ ]*record_session\.py'
}
rollout_running() {
  ps -eo args= | grep -Eq '^[^ ]*python[^ ]*( -[^ ]+)* [^ ]*lerobot-rollout'
}

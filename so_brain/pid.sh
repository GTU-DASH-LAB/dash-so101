#!/usr/bin/env bash
# Classical (non-learned) pick-and-place. Usage: ./so_brain/pid.sh [--dry-run] [flags]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source pencil_pickup_vla/config.env; set +a
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python so_brain/pid_pick.py "$@"

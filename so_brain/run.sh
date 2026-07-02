#!/usr/bin/env bash
# Usage: ./so_brain/run.sh "put the red pen on the black mouse pad" [extra flags]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source pencil_pickup_vla/config.env; set +a
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export POINTACT_MODEL="${POINTACT_MODEL:-${HF_USER}/pointact_pickplace}"
exec .venv/bin/python so_brain/run.py "$@"

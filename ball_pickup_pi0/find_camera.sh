#!/usr/bin/env bash
# List connected cameras so you can confirm which /dev/videoN index is the
# top camera vs the wrist camera. Set the indices in config.env (CAMERAS).
set -euo pipefail
cd "$(dirname "$0")"
source ../.venv/bin/activate

lerobot-find-cameras opencv

#!/usr/bin/env bash
# ============================================================================
# Step 1b - merge all good sessions into ONE dataset and publish it.
#
#   ./1b_publish_dataset.sh             merge + verify + push to the HF Hub
#   ./1b_publish_dataset.sh --no-push   merge + verify only (offline / not ready)
#   ./1b_publish_dataset.sh --push-only push the already-merged dataset again
#                                       (e.g. the upload was interrupted)
#
# Result: $DATASET_ROOT (what training reads) and
#         https://huggingface.co/datasets/$DATASET_REPO
# Re-running is safe: an existing merged copy goes to the trash first, and the
# Hub upload is resumable (upload_large_folder), which matters on this network.
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"
export HERE SESSIONS_DIR FPS MIN_EPISODE_S TARGET_EPISODES

MODE="${1:-}"
recorder_running && die "A recording is running. Finish it first."

push() {
  [ "$PUSH_DATASET_TO_HUB" = "true" ] || { warn "PUSH_DATASET_TO_HUB=false in config.env - not pushing."; return 0; }
  "$VENV_BIN/hf" auth whoami >/dev/null 2>&1 || die "Not logged in to Hugging Face. Run: $VENV_BIN/hf auth login"
  info "Uploading to https://huggingface.co/datasets/$DATASET_REPO (resumable; re-run with --push-only if it stops)"
  DATASET_REPO="$DATASET_REPO" DATASET_ROOT="$DATASET_ROOT" PRIVATE="$DATASET_PRIVATE" "$PY" - <<'EOF'
import os
from lerobot.datasets import LeRobotDataset
ds = LeRobotDataset(os.environ["DATASET_REPO"], root=os.environ["DATASET_ROOT"])
ds.push_to_hub(
    tags=["so101", "act", "wrist-camera", "pen", "pick-and-place", "LeRobot"],
    private=os.environ["PRIVATE"] == "true",
    upload_large_folder=True,
)
print(f"Pushed {ds.num_episodes} demos / {ds.num_frames} frames")
EOF
  local rc=$?
  [ $rc -eq 0 ] || die "Upload failed (exit $rc). Your local dataset is fine - run ./1b_publish_dataset.sh --push-only to retry."
  ok "Published: https://huggingface.co/datasets/$DATASET_REPO"
  echo "   Browse it: https://huggingface.co/spaces/lerobot/visualize_dataset?path=/$DATASET_REPO/episode_0"
}

if [ "$MODE" = "--push-only" ]; then
  [ -f "$DATASET_ROOT/meta/info.json" ] || die "No merged dataset at $DATASET_ROOT yet. Run ./1b_publish_dataset.sh first."
  push; exit $?
fi

export HF_HUB_OFFLINE=1   # merging is purely local
"$PY" sessions.py status || exit 1
TOTAL=$("$PY" sessions.py status --json | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["total_episodes"])')
if [ "${TOTAL:-0}" -lt "$TARGET_EPISODES" ]; then
  warn "Only $TOTAL of $TARGET_EPISODES demos are recorded."
  confirm "Merge and publish anyway?" || exit 1
fi

info "Checking every session (decodes every demo's first and last frame)"
"$PY" sessions.py check || die "Fix or delete the broken session(s) first (./manage_data.sh delete-session <n>)."

if [ -d "$OUTPUT_DIR/checkpoints" ] || [ -f "$RUN_DIR/training.pid" ]; then
  warn "A trained model already exists in $OUTPUT_DIR. It was trained on the OLD merged dataset."
fi

TMP="${DATASET_ROOT}.merging"
rm -rf "$TMP"
info "Merging sessions into $DATASET_ROOT"
"$PY" sessions.py merge --out-root "$TMP" --out-repo "$DATASET_REPO" || { rm -rf "$TMP"; die "Merge failed; nothing changed."; }
if [ -d "$DATASET_ROOT" ]; then
  mkdir -p "$SESSIONS_DIR/_trash"
  OLD="$SESSIONS_DIR/_trash/$(date +%Y%m%d_%H%M%S)__previous_merged_dataset"
  mv "$DATASET_ROOT" "$OLD" && info "Previous merged dataset moved to $OLD"
fi
mkdir -p "$(dirname "$DATASET_ROOT")"
mv "$TMP" "$DATASET_ROOT"
ok "Training dataset ready: $DATASET_ROOT"

unset HF_HUB_OFFLINE
if [ "$MODE" = "--no-push" ]; then
  echo "Not pushed (--no-push). Push later with: ./1b_publish_dataset.sh --push-only"
else
  push || exit 1
fi
echo
echo "Next: ./2_train_local.sh"

#!/usr/bin/env bash
# ============================================================================
# Step 2b - publish a trained checkpoint to the Hugging Face Hub.
#
#   ./2b_publish_model.sh                newest checkpoint (end of training)
#   ./2b_publish_model.sh best           checkpoint with the lowest validation loss
#   ./2b_publish_model.sh 004550         a specific checkpoint (names: ./3_run_autonomous.sh --list)
#   ./2b_publish_model.sh --dry-run [..] build the upload (weights, model card, loss
#                                        curve) in run/model_upload/ without uploading
#
# Result: https://huggingface.co/$MODEL_REPO with a model card (task, inputs,
# training details, every checkpoint's validation loss, the loss curve, and a
# ready-to-run lerobot-rollout command). The checkpoint goes to the main branch
# and is tagged step-NNNNNN, so pushing another one later keeps this one too.
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"
export CAMERAS MODEL_REPO MODEL_PRIVATE DATASET_REPO DATASET_ROOT TASK VIDEO_FRAME_STRIDE N_ACTION_STEPS MAX_RELATIVE_TARGET

DRY=()
CKPT=last
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=(--dry-run) ;;
    -h|--help) sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) CKPT="$a" ;;
  esac
done

CK_ROOT="$OUTPUT_DIR/checkpoints"
[ -d "$CK_ROOT" ] || die "No trained model in $OUTPUT_DIR. Run ./2_train_local.sh first."
if [ -f "$RUN_DIR/watchdog.pid" ] && kill -0 "$(cat "$RUN_DIR/watchdog.pid")" 2>/dev/null; then
  warn "Training is still running - you would publish an unfinished model."
  confirm "Publish anyway?" || exit 1
fi
case "$CKPT" in
  best) CKPT=$("$PY" -c 'import train_dashboard as t; b=[c for c in t.collect()["checkpoints"] if c.get("best")]; print(b[0]["name"] if b else "")')
        [ -n "$CKPT" ] || die "No checkpoint has a validation loss yet." ;;
  last) CKPT=$(basename "$(readlink -f "$CK_ROOT/last")") ;;
esac
[ -f "$CK_ROOT/$CKPT/pretrained_model/model.safetensors" ] \
  || die "No checkpoint '$CKPT'. Available: $(ls "$CK_ROOT" | grep -v '^last$' | tr '\n' ' ')"

if [ ${#DRY[@]} -eq 0 ]; then
  "$VENV_BIN/hf" auth whoami >/dev/null 2>&1 || die "Not logged in to Hugging Face. Run: $VENV_BIN/hf auth login"
  info "Publishing checkpoint $CKPT to https://huggingface.co/$MODEL_REPO ($( [ "$MODEL_PRIVATE" = true ] && echo private || echo public ))"
fi
"$PY" publish_model.py --checkpoint "$CKPT" "${DRY[@]}"
rc=$?
[ $rc -eq 0 ] || die "Publishing failed (exit $rc). Nothing local changed - just run it again."
[ ${#DRY[@]} -gt 0 ] && exit 0
ok "Published: https://huggingface.co/$MODEL_REPO"
echo "   Run it from the Hub:  ./3_run_autonomous.sh hub"

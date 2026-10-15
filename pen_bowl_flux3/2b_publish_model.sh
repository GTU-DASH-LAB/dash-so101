#!/usr/bin/env bash
# ============================================================================
# Step 2b - publish a FLUX 3 checkpoint (pulled back from AWS) to the Hugging Face Hub.
#
#   ./2b_publish_model.sh                newest checkpoint
#   ./2b_publish_model.sh best           checkpoint with the lowest validation loss
#   ./2b_publish_model.sh 006000         a specific checkpoint (names: ./3_run_autonomous.sh --list)
#   ./2b_publish_model.sh --dry-run [..] build the upload (adapter, model card, license)
#                                        in run/model_upload/ without uploading
#
# Uploads the EMA weights with the base named by its Hub id (deploy_config.py), so
# `--policy.path=$MODEL_REPO` runs anywhere. Tagged step-NNNNNN on the Hub.
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/lib.sh"
export CAMERAS MODEL_REPO MODEL_PRIVATE DATASET_REPO DATASET_ROOT TASK N_ACTION_STEPS MAX_RELATIVE_TARGET \
  OUTPUT_DIR EVAL_SPLIT BATCH_SIZE GRAD_ACCUM

DRY=()
CKPT=last
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=(--dry-run) ;;
    -h|--help) sed -n '3,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) CKPT="$a" ;;
  esac
done

CK_ROOT="$OUTPUT_DIR/checkpoints"
[ -d "$CK_ROOT" ] || die "No checkpoints in $OUTPUT_DIR. Train on AWS, then ./aws_train.sh pull."
case "$CKPT" in
  last) CKPT=$(basename "$(readlink -f "$CK_ROOT/last")") ;;
  best) CKPT=$("$PY" -c '
import publish_model as p, os
from pathlib import Path
have = {int(d.name) for d in Path(os.environ["OUTPUT_DIR"], "checkpoints").iterdir() if d.name.isdigit()}
l = {s: v for s, v in p.eval_losses(p.HERE / "aws_train.log").items() if s in have}
print(f"{min(l, key=l.get):06d}" if l else "")')
        [ -n "$CKPT" ] || die "No pulled checkpoint has a validation loss in aws_train.log." ;;
esac
[ -d "$CK_ROOT/$CKPT" ] || die "No checkpoint '$CKPT'. Available: $(ls "$CK_ROOT" | grep -v '^last$' | tr '\n' ' ')"

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

#!/usr/bin/env bash
# ============================================================================
# Train FLUX 3 Action on AWS (one H100), driven from this PC.
#
#   ./aws_train.sh start      launch the H100 (asks first), set it up, measure speed,
#                             start training in the background ON the machine
#   ./aws_train.sh status     training progress + time and money spent so far
#   ./aws_train.sh pull       copy checkpoints + log back to ./outputs (resumable)
#   ./aws_train.sh watch      pull every 15 min until training ends, then pull once more
#   ./aws_train.sh resume     after a spot interruption: new machine, push the last
#                             checkpoint, continue training
#   ./aws_train.sh finish     final pull, then terminate the machine
#
# Needs: the 16-vCPU P-instance quota (aws/check_quota_requests.sh), `aws login`, and a
# READ-only Hugging Face token in ~/.config/hf-aws/token (the FLUX 3 weights are gated;
# this token is the only credential copied to the machine).
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOX="$HERE/../aws/gpu_box.sh"
TOKEN="$HOME/.config/hf-aws/token"
LOCAL_OUT="$HERE/outputs/flux3_pen_bowl"
REMOTE_OUT="work/outputs/flux3_pen_bowl"
die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo ":: $*"; }

push_code() {
  "$BOX" ssh "mkdir -p ~/work" || die "machine not reachable"
  for f in config.env 0_setup_env.sh prepare_init.py remote_train.sh; do
    "$BOX" push "$HERE/$f" "work/$f" >/dev/null || die "push $f failed"
  done
  "$BOX" push "$TOKEN" ".hf_token" >/dev/null && "$BOX" ssh "chmod 600 ~/.hf_token"
}
remote_bg() {  # run remote_train.sh <cmd> detached on the machine
  "$BOX" ssh "cd ~/work && nohup ./remote_train.sh $1 > ~/work/$1.out 2>&1 < /dev/null & echo started"
}

case "${1:-status}" in
start|resume)
  [ -s "$TOKEN" ] || die "Put a READ-only Hugging Face token in $TOKEN (huggingface.co/settings/tokens)."
  "$BOX" status | grep -q "No machine running" && { "$BOX" launch || exit 1; }
  push_code
  info "Setting up the machine (python env, dataset, 14 GB FLUX 3 weights): ~10-15 min"
  "$BOX" ssh "cd ~/work && ./remote_train.sh setup" > "$HERE/aws_setup.log" 2>&1 \
    || die "setup failed - see $HERE/aws_setup.log"
  if [ "$1" = resume ]; then
    [ -d "$LOCAL_OUT/checkpoints/last" ] || die "No local checkpoint to resume from."
    info "Pushing the last checkpoint"
    "$BOX" ssh "mkdir -p ~/$REMOTE_OUT/checkpoints"
    "$BOX" push "$LOCAL_OUT/checkpoints/" "$REMOTE_OUT/checkpoints/" || die "push failed"
  else
    info "Measuring speed and memory (20 real steps)"
    "$BOX" ssh "cd ~/work && ./remote_train.sh probe 20" | tee "$HERE/aws_probe.log"
  fi
  remote_bg train
  info "Training runs on the machine. ./aws_train.sh watch  (pulls checkpoints every 15 min)"
  ;;
status)
  "$BOX" status
  "$BOX" status | grep -q "No machine running" && exit 0
  "$BOX" ssh "tail -c 2000 ~/work/train.log | tr '\r' '\n' | grep -E 'step:|STARTING|RESUMING|EXIT|Error' | tail -4; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
  ;;
pull)
  mkdir -p "$LOCAL_OUT"
  "$BOX" pull "$REMOTE_OUT/" "$LOCAL_OUT/" && "$BOX" pull "work/train.log" "$HERE/aws_train.log"
  ;;
watch)
  while :; do
    "$0" pull > /dev/null 2>&1 && info "$(date '+%H:%M') pulled; $(grep -E 'step:' "$HERE/aws_train.log" 2>/dev/null | tail -1 | cut -c1-120)"
    "$BOX" ssh "grep -q 'EXIT' ~/work/train.log" 2>/dev/null && { "$0" pull; info "Training ended: $(grep EXIT "$HERE/aws_train.log" | tail -1)"; break; }
    "$BOX" status | grep -q "No machine running" && { info "Machine is gone (finished, time limit or spot reclaim). ./aws_train.sh resume if unfinished."; break; }
    sleep 900
  done
  ;;
finish)
  "$0" pull || die "pull failed - NOT terminating"
  "$BOX" terminate
  ;;
*) sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac

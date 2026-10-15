#!/usr/bin/env bash
# ============================================================================
# Runs ON the AWS GPU machine (pushed there by aws_train.sh; not for this PC).
#
#   remote_train.sh setup    python env, dataset, FLUX 3 weights, wrist-only init
#   remote_train.sh train    LoRA fine-tune (resumes from checkpoints/last if present)
#   remote_train.sh probe N  N real training steps, prints s/step and peak memory
#
# Everything lives under ~/work. Log: ~/work/train.log. The gated FLUX 3 weights are
# fetched with the read-only token in ~/.hf_token (pushed by aws_train.sh).
# ============================================================================
set -uo pipefail
W="$HOME/work"
cd "$W" || exit 1
set -a; source "$W/config.env"; set +a
export PATH="$HOME/.local/bin:$PATH" HF_HUB_ENABLE_HF_TRANSFER=0
[ -s "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"
PY="$FLUX3_ENV/bin/python"
INIT="$W/init_policy"
OUT="$W/outputs/flux3_pen_bowl"
SRC="$HOME/src/lerobot-main"

train_args() {
  echo --config_path="$SRC/examples/flux3/lora.json" \
    --policy.path="$INIT" --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id="$DATASET_REPO" --dataset.root="$DATASET_ROOT" \
    --dataset.eval_split="$EVAL_SPLIT" --dataset.video_backend=torchcodec \
    --batch_size="$BATCH_SIZE" --accelerator.gradient_accumulation.steps="$GRAD_ACCUM" \
    --num_workers="$NUM_WORKERS" --save_freq="$SAVE_FREQ" --eval_steps="$SAVE_FREQ" \
    --wandb.enable=false
}

case "${1:-}" in
setup)
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh || exit 1; }
  "$W/0_setup_env.sh" || exit 1
  "$PY" - <<'EOF' || exit 1
import os
from lerobot.datasets import LeRobotDataset
ds = LeRobotDataset(os.environ["DATASET_REPO"], root=os.environ["DATASET_ROOT"])
print(f"dataset: {ds.num_episodes} episodes, {ds.num_frames} frames")
EOF
  "$FLUX3_ENV/bin/hf" download "$BASE_POLICY_REPO" --local-dir "$HOME/models/flux-3-action-so101" || exit 1
  "$PY" "$W/prepare_init.py" --ckpt "$HOME/models/flux-3-action-so101" --out "$INIT" || exit 1
  # Text encoder + video VAE are fetched at model build from the pinned base revision; warm the cache now.
  "$PY" -c "from lerobot.policies.flux3 import Flux3Policy as P; P.from_pretrained('$INIT'); print('init policy loads OK')" || exit 1
  ;;
probe)
  n="${2:-20}"
  rm -rf "$W/outputs/probe"
  "$PY" -m lerobot.scripts.lerobot_train $(train_args) --steps="$n" --log_freq=1 --eval_steps=0 \
    --save_freq=100000 --output_dir="$W/outputs/probe" > "$W/probe.log" 2>&1
  rc=$?
  grep -oE "step:[0-9K]+ .*" "$W/probe.log" | tail -4
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  exit $rc
  ;;
train)
  if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
    ARGS=(--config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true)
    echo "$(date) RESUMING" >> "$W/train.log"
  else
    ARGS=($(train_args) --steps="$STEPS" --log_freq=20 --output_dir="$OUT")
    echo "$(date) STARTING $STEPS microsteps" >> "$W/train.log"
  fi
  "$PY" -m lerobot.scripts.lerobot_train "${ARGS[@]}" >> "$W/train.log" 2>&1
  rc=$?
  echo "$(date) EXIT $rc" >> "$W/train.log"
  exit $rc
  ;;
*) sed -n '3,11p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac

#!/usr/bin/env bash
# ============================================================================
# Python environment for FLUX 3 Action on THIS machine (RTX 5090, sm_120):
# ~/envs/lerobot-main with lerobot from GitHub main (FLUX 3 is not in lerobot
# 0.6.0, which the lelab venv used by every other task folder pins).
#
# Separate on purpose: upgrading lelab's lerobot would change ACT / VLA-JEPA /
# pi0.5 training underneath them (see CLAUDE.md). The same commands run on the
# AWS training machine (aws_setup_remote.sh) so both sides match.
#
#   torch 2.11.0+cu128   (sm_120 needs CUDA 12.8+; already in uv's cache)
#   lerobot @ main       [training,flux3,peft,diffusion,feetech] (the docs' extras + the arm)
#   natten 0.21.6        the FLUX 3 video VAE's neighbourhood-attention kernels
#
# Then runs a real NATTEN op on GPU 0 to prove the kernels work on Blackwell.
# ============================================================================
set -uo pipefail
ENV="${FLUX3_ENV:-$HOME/envs/lerobot-main}"
SRC="${LEROBOT_MAIN_SRC:-$HOME/src/lerobot-main}"
LOG="$(cd "$(dirname "$0")" && pwd)/setup_env.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date) setup into $ENV"

# Skip git-lfs content (test fixtures and media, not needed to run or train).
export GIT_LFS_SKIP_SMUDGE=1
if [ -d "$SRC/.git" ]; then git -C "$SRC" pull --ff-only || exit 1
else git clone --depth 1 https://github.com/huggingface/lerobot.git "$SRC" || exit 1; fi
echo "lerobot main at $(git -C "$SRC" log -1 --format='%h %cd')"

[ -x "$ENV/bin/python" ] || uv venv --python 3.12 "$ENV" || exit 1
uv pip install --python "$ENV/bin/python" torch==2.11.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu128 || exit 1
uv pip install --python "$ENV/bin/python" -e "$SRC[training,flux3,peft,diffusion,feetech]" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match || exit 1
uv pip install --python "$ENV/bin/python" "natten==0.21.6+torch2110cu128" --find-links https://whl.natten.org || exit 1

"$ENV/bin/python" - <<'EOF' || { echo "GPU CHECK FAILED"; exit 1; }
import torch, natten, lerobot
from natten.functional import na2d
print("torch", torch.__version__, "cuda", torch.version.cuda, "| natten", natten.__version__, "| lerobot", lerobot.__file__)
dev = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(dev), torch.cuda.get_device_capability(dev))
a = torch.randn(256, 256, device=dev, dtype=torch.bfloat16); (a @ a).sum().item()  # real kernel, not is_available()
q = torch.randn(1, 32, 32, 4, 64, device=dev, dtype=torch.bfloat16)  # [B, H, W, heads, dim]
out = na2d(q, q, q, kernel_size=7)
torch.cuda.synchronize()
print("NATTEN na2d on GPU OK:", tuple(out.shape), out.dtype)
from lerobot.policies.flux3 import Flux3Policy  # noqa: F401
print("lerobot.policies.flux3 imports OK")
EOF
echo "=== $(date) DONE"

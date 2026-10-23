#!/usr/bin/env bash
# ============================================================================
# One-time setup for Yaren (idempotent: re-running skips what is already there).
#   1. ~/envs/yaren: Python 3.12 venv for the voice side (torch cu128, Whisper via transformers,
#      Chatterbox, Silero VAD, FastAPI). Separate from lelab's venv on purpose: Chatterbox pins
#      transformers 5.2, lerobot pins 5.5.
#   2. Models in ~/models/yaren: Whisper large-v3-turbo, Chatterbox Multilingual, its sample voices.
#   3. Ollama 0.40.9 in ~/opt (the system one is 0.32.9, too old for Qwen3.8) and the LLM.
#   4. The web app (npm install + build).
# The robot side needs nothing new: lelab's venv + TensorRT (../tools/trt/README.md).
# Downloads use aria2c (per-connection throttled network, see robotics/CLAUDE.md).
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1
source config.env
M="$HOME/models/yaren"
mkdir -p "$M"
fail() { printf '\e[1;31mERROR\e[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\e[1;34m== %s\e[0m\n' "$*"; }

step "1. voice venv ($VOICE_VENV)"
if [ ! -x "$VOICE_VENV/bin/python" ]; then
  uv venv "$VOICE_VENV" --python 3.12 || fail "uv venv"
fi
PIPX=(uv pip install --python "$VOICE_VENV/bin/python" --index-url https://download.pytorch.org/whl/cu128
      --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match)
"${PIPX[@]}" "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128" || fail "torch"
uv pip install --python "$VOICE_VENV/bin/python" --no-deps chatterbox-tts==0.1.7 silero-vad || fail "chatterbox"
"${PIPX[@]}" "numpy<2" "librosa==0.11.0" s3tokenizer "transformers==5.2.0" "diffusers==0.29.0" resemble-perth \
  "conformer==0.3.2" safetensors spacy-pkuseg "pykakasi==2.3.0" pyloudnorm omegaconf num2words \
  fastapi "uvicorn[standard]" httpx soundfile python-multipart websockets || fail "deps"
"$VOICE_VENV/bin/python" - <<'EOF' || fail "torch cannot run a kernel on this GPU (needs +cu128 for sm_120)"
import torch
x = torch.randn(256, 256, device="cuda")
print("torch", torch.__version__, "matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
EOF

step "2. models ($M)"
fetch() {  # repo revision dir file...
  local repo=$1 rev=$2 dir=$3; shift 3
  mkdir -p "$M/$dir"
  local list="$M/$dir.aria2"; : > "$list"
  for f in "$@"; do
    [ -f "$M/$dir/$f" ] && [ ! -f "$M/$dir/$f.aria2" ] && continue
    printf '%s\n  out=%s\n' "https://huggingface.co/$repo/resolve/$rev/$f" "$f" >> "$list"
  done
  [ -s "$list" ] || { echo "$dir: complete"; return 0; }
  aria2c -c -x16 -s16 -j4 --console-log-level=warn --summary-interval=0 -d "$M/$dir" -i "$list" || fail "download $repo"
}
fetch openai/whisper-large-v3-turbo 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 whisper-large-v3-turbo \
  added_tokens.json config.json generation_config.json merges.txt normalizer.json preprocessor_config.json \
  special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json model.safetensors
fetch ResembleAI/chatterbox 5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18 chatterbox \
  ve.pt t3_mtl23ls_v2.safetensors s3gen.pt grapheme_mtl_merged_expanded_v1.json conds.pt Cangjie5_TC.json
mkdir -p "$VOICE_REFS"
for f in tr_m.flac en_f1.flac; do   # Resemble's own demo prompts (Chatterbox-Multilingual-TTS space)
  [ -f "$VOICE_REFS/$f" ] || curl -sfL -o "$VOICE_REFS/$f" "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/$f" \
    || echo "!! could not fetch sample voice $f (optional)"
done

step "3. Ollama + $LLM_MODEL"
if [ ! -x "$OLLAMA_BIN" ]; then
  mkdir -p "$(dirname "$(dirname "$OLLAMA_BIN")")"
  tarball="$HOME/opt/ollama-linux-amd64-v0.40.9.tar.zst"
  aria2c -c -x16 -s16 --console-log-level=warn -d "$HOME/opt" -o "$(basename "$tarball")" \
    https://github.com/ollama/ollama/releases/download/v0.40.9/ollama-linux-amd64.tar.zst || fail "ollama download"
  tar --zstd -xf "$tarball" -C "$(dirname "$(dirname "$OLLAMA_BIN")")" || fail "ollama extract"
fi
started=false
if ! curl -s -m 2 "http://127.0.0.1:$OLLAMA_PORT/api/version" > /dev/null; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$VOICE_GPU" OLLAMA_VULKAN=0 OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" OLLAMA_MODELS="$OLLAMA_MODELS" \
    nohup "$OLLAMA_BIN" serve > "$M/ollama_setup.log" 2>&1 &
  opid=$!; started=true; sleep 3
fi
OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" "$OLLAMA_BIN" pull "$LLM_MODEL" || fail "ollama pull $LLM_MODEL"
[ "$started" = true ] && kill "$opid"

step "4. web app"
(cd web && npm install --no-fund --no-audit && npm run build) || fail "web build"

echo; printf '\e[1;32mOK\e[0m setup complete. Try it without the arm first:  ./start.sh --dry\n'

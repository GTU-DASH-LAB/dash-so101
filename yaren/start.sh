#!/usr/bin/env bash
# ============================================================================
# Start Yaren: talk to the SO-101 and let it act.
#
#   ./start.sh                  the real arm, newest MolmoAct2 checkpoint
#   ./start.sh best|<step>      another checkpoint (names: ../pen_bowl_molmoact2/3_run_autonomous.sh --list)
#   ./start.sh --dry            NO ARM: a recorded episode is replayed as the camera and actions go
#                               nowhere - tests everything else safely
#   ./stop.sh                   stop all three (the arm returns to its start pose first)
#
# Three processes (logs in run/):
#   Ollama         GPU1  the mind (Qwen3.8-27B)                      run/ollama.log
#   robot_server   GPU0  MolmoAct2 + TensorRT, the arm, the camera   run/robot.log
#   voice server   GPU1  ears (Whisper), voice (Chatterbox), the app run/voice.log
#
# SAFETY: like ./3_run_autonomous.sh, the arm moves on its own when you ask it to. Clear the workspace
# and keep a hand near the power switch the first time. Each joint is limited to MAX_RELATIVE_TARGET
# degrees per tick (from the policy folder's config.env). STOP: the red button, Esc, or say "dur"/"stop".
# Start pose: put the arm where you started the demos (pen visible in the wrist camera); it returns
# there after every run.
# ============================================================================
set -uo pipefail
YAREN="$(cd "$(dirname "$0")" && pwd)"   # not HERE: the policy folder's lib.sh sets that
cd "$YAREN" || exit 1
source config.env
RUN="$YAREN/run"
mkdir -p "$RUN"

die()  { printf '\e[1;31mERROR\e[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\e[1;34m::\e[0m %s\n' "$*"; }
ok()   { printf '\e[1;32mOK\e[0m %s\n' "$*"; }

DRY=false
CKPT="$CHECKPOINT"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry) DRY=true; shift ;;
    -h|--help) sed -n '3,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) CKPT="$1"; shift ;;
  esac
done

alive() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }
port_busy() { ss -ltn 2>/dev/null | grep -q ":$1 "; }

alive robot && die "Yaren is already running (./stop.sh first)."
alive voice && die "Yaren's voice server is already running (./stop.sh first)."
[ -x "$VOICE_VENV/bin/python" ] || die "No voice venv at $VOICE_VENV - run ./setup.sh"
[ -f web/dist/index.html ] || die "The app is not built - run ./setup.sh"
for p in "$ROBOT_HTTP_PORT" "$UI_HTTP_PORT" "$UI_HTTPS_PORT"; do
  port_busy "$p" && die "Port $p is in use (another Yaren? see run/*.pid)"
done

# ---------------------------------------------------------------------------- 1. robot side
# Built in a subshell inside the policy folder with its own lib.sh, exactly as its run script does.
(
  cd "$POLICY_DIR" || exit 1
  source lib.sh
  recorder_running && die "A recording is running. Only one program can drive the arm."
  rollout_running && die "./3_run_autonomous.sh (or another Yaren) is driving the arm."
  CK_ROOT="$OUTPUT_DIR/checkpoints"
  if [ "$CKPT" = "best" ]; then
    CKPT=$("$PY" -c 'import train_dashboard as t; b=[c for c in t.collect()["checkpoints"] if c.get("best")]; print(b[0]["name"] if b else "")')
    [ -n "$CKPT" ] || die "No checkpoint has a validation loss yet."
  fi
  SRC="$CK_ROOT/$CKPT/pretrained_model"
  [ -f "$SRC/config.json" ] || die "No checkpoint '$CKPT'. Available: $(ls "$CK_ROOT" 2>/dev/null | tr '\n' ' ')"
  POLICY="$RUN_DIR/deploy/$(basename "$(readlink -f "$CK_ROOT/$CKPT")")"
  JOINT_SIGNS="$JOINT_SIGNS" JOINT_OFFSETS="$JOINT_OFFSETS" "$PY" deploy_config.py "$SRC" "$POLICY" > "$RUN_DIR/deploy.out" 2>&1 \
    || { cat "$RUN_DIR/deploy.out"; die "deploy_config.py failed"; }

  EXTRA=()
  if [ "$DRY" = true ]; then
    FOLLOWER_PORT=/dev/null
    EXTRA=(--dry --dry-root "$DATASET_ROOT" --dry-repo-id "$DATASET_REPO")
  else
    "$PY" hw_check.py --no-leader --follower-serial "$FOLLOWER_SERIAL" --follower-id "$FOLLOWER_ID" \
      --cameras "$CAMERAS" > "$RUN_DIR/hw_check.out" 2>&1
    rc=$?
    grep -v '^PORTS ' "$RUN_DIR/hw_check.out"
    [ $rc -eq 0 ] || die "Hardware check failed - fix the FAIL lines above and run again."
    FOLLOWER_PORT="$(sed -n 's/^PORTS.*follower=\([^ ]*\).*/\1/p' "$RUN_DIR/hw_check.out")"
    # the leader arm (optional: mirroring and teaching with it), found by its USB serial like the follower
    LEADER_PORT=""
    for d in /dev/serial/by-id/*"${LEADER_SERIAL:-none}"*; do [ -e "$d" ] && LEADER_PORT="$(readlink -f "$d")"; done
    [ -n "$LEADER_PORT" ] && EXTRA+=(--leader-port "$LEADER_PORT" --leader-id "$LEADER_ID")
  fi
  info "Policy: $POLICY  (task it was trained on: \"$TASK\")"
  info "Arm: $( [ "$DRY" = true ] && echo 'NONE (--dry, replaying a recorded episode)' || echo "$FOLLOWER_PORT" ), max $MAX_RELATIVE_TARGET deg/tick per joint"
  [ "$DRY" = true ] || info "Leader arm: ${LEADER_PORT:-not found (mirroring and teaching with it are off)}"
  HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$ROBOT_GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" "$YAREN/robot_server.py" --port "$ROBOT_HTTP_PORT" --task-timeout "$TASK_TIMEOUT_S" "${EXTRA[@]}" \
      --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
      --robot.max_relative_target="$MAX_RELATIVE_TARGET" --robot.cameras="$CAMERAS" \
      --policy.path="$POLICY" --policy.device=cuda --policy.n_action_steps="$N_ACTION_STEPS" \
      --policy.checkpoint_path="$BASE_CHECKPOINT" \
      --task="$TASK" --fps="$FPS" --display_data=false --play_sounds=false \
      --strategy.type=base --duration=0 > "$RUN/robot.log" 2>&1 &
  echo $! > "$RUN/robot.pid"
) || exit 1

# ---------------------------------------------------------------------------- 2. the mind (Ollama)
if ! curl -s -m 2 "http://127.0.0.1:$OLLAMA_PORT/api/version" > /dev/null; then
  [ -x "$OLLAMA_BIN" ] || die "No Ollama at $OLLAMA_BIN - run ./setup.sh"
  # OLLAMA_VULKAN=0: Ollama >= 0.40 also enumerates GPUs through Vulkan, which ignores
  # CUDA_VISIBLE_DEVICES - it put the 17 GB model on GPU0 next to MolmoAct2, whose TensorRT
  # engine then failed to load for lack of memory.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$VOICE_GPU" OLLAMA_VULKAN=0 GGML_VK_VISIBLE_DEVICES="$VOICE_GPU" \
    OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" OLLAMA_MODELS="$OLLAMA_MODELS" \
    OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=2 OLLAMA_FLASH_ATTENTION=1 \
    nohup "$OLLAMA_BIN" serve > "$RUN/ollama.log" 2>&1 &
  echo $! > "$RUN/ollama.pid"
fi

# ---------------------------------------------------------------------------- 3. TLS for other machines
if [ ! -f "$RUN/cert.pem" ]; then
  SAN="DNS:localhost,DNS:$(hostname),IP:127.0.0.1"
  for ip in $(hostname -I); do case "$ip" in *:*) ;; *) SAN="$SAN,IP:$ip" ;; esac; done
  openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj "/CN=Yaren" -addext "subjectAltName=$SAN" \
    -keyout "$RUN/key.pem" -out "$RUN/cert.pem" > "$RUN/openssl.log" 2>&1 || info "No TLS certificate (openssl failed): https access disabled"
fi

# ---------------------------------------------------------------------------- 4. voice server
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$VOICE_GPU" HF_HUB_OFFLINE=1 TQDM_DISABLE=1 \
  LLM_URL="http://127.0.0.1:$OLLAMA_PORT" LLM_MODEL="$LLM_MODEL" LLM_CTX="$LLM_CTX" \
  ASR_MODEL="$ASR_MODEL" TTS_MODEL="$TTS_MODEL" VOICE_REFS="$VOICE_REFS" DEFAULT_VOICE="$DEFAULT_VOICE" \
  ROBOT_URL="http://127.0.0.1:$ROBOT_HTTP_PORT" UI_HTTP_PORT="$UI_HTTP_PORT" UI_HTTPS_PORT="$UI_HTTPS_PORT" \
  TASK_TIMEOUT_S="$TASK_TIMEOUT_S" END_OF_TURN_MS="$END_OF_TURN_MS" DONE_CHECK="$DONE_CHECK" \
  DONE_CHECK_AFTER_S="$DONE_CHECK_AFTER_S" DONE_CHECK_EVERY_S="$DONE_CHECK_EVERY_S" DONE_CHECK_STREAK="$DONE_CHECK_STREAK" \
  MEMORY_DIR="$(cd "$YAREN" && realpath -m "$MEMORY_DIR")" \
  nohup "$VOICE_VENV/bin/python" -m agent.server > "$RUN/voice.log" 2>&1 &
echo $! > "$RUN/voice.pid"

# ---------------------------------------------------------------------------- 5. wait until ready
info "Loading: MolmoAct2 + TensorRT (GPU$ROBOT_GPU), Whisper + Chatterbox + $LLM_MODEL (GPU$VOICE_GPU) ..."
for i in $(seq 1 120); do
  sleep 2
  alive robot || { tail -n 30 "$RUN/robot.log"; die "robot_server exited - see run/robot.log"; }
  alive voice || { tail -n 30 "$RUN/voice.log"; die "voice server exited - see run/voice.log"; }
  curl -s -m 2 "http://127.0.0.1:$ROBOT_HTTP_PORT/status" > /dev/null && \
    curl -s -m 2 -o /dev/null "http://127.0.0.1:$UI_HTTP_PORT/" && break
done
curl -s -m 2 "http://127.0.0.1:$ROBOT_HTTP_PORT/status" > /dev/null || die "robot_server did not come up - see run/robot.log"
curl -s -m 2 -o /dev/null "http://127.0.0.1:$UI_HTTP_PORT/" || die "voice server did not come up - see run/voice.log"

TOKEN="$(cat "$RUN/token")"
echo
ok "Yaren is up$( [ "$DRY" = true ] && echo ' (DRY RUN - no arm)' )."
echo "   On this machine:     http://127.0.0.1:$UI_HTTP_PORT/?token=$TOKEN"
if [ -f "$RUN/cert.pem" ]; then
  for ip in $(hostname -I); do
    case "$ip" in 10.*|192.168.*) echo "   From your laptop:    https://$ip:$UI_HTTPS_PORT/?token=$TOKEN  (accept the certificate warning once)";; esac
  done
fi
echo "   The mind keeps loading for ~30 s after this (the 'Mind' light turns green)."
echo "   Stop everything:     ./stop.sh"
printf '%s\n' "$(date '+%Y-%m-%d %H:%M') [yaren] started$( [ "$DRY" = true ] && echo ' --dry'), checkpoint $CKPT" >> "$HOME/robotics/logs/setup-progress.log"

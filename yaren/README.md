# Yaren — talk to the SO-101, and it acts

*Yaren* is an old Turkish word for a close friend you sit and chat with (*yarenlik* = friendly
conversation). That is the idea: you talk to the arm like a friend, in Turkish or English. It answers
in a natural voice. When you ask it to do something ("kalemi al, kaseye koy"), it runs the MolmoAct2
policy with your request as the task prompt. You can stop it at any moment.

```
 browser (this PC, or your Mac over HTTPS)                         GPU1 (RTX 5090, x4 link)
 ┌───────────────────────────────┐   WebSocket: mic PCM 16 kHz ┌──────────────────────────────────┐
 │ React app (web/)              │ ──────────────────────────▶ │ agent/server.py  (FastAPI)       │
 │  mic (AudioWorklet, AEC)      │                             │  Silero VAD  → end of your turn  │
 │  speaker (WebAudio)           │ ◀────────────────────────── │  Whisper large-v3-turbo  (ears)  │
 │  wrist camera (MJPEG)         │   reply audio + events      │  Qwen3.8-27B via Ollama  (mind)  │
 │  STOP / Home / voice / lang   │                             │  Chatterbox Multilingual (voice) │
 └───────────────────────────────┘                             └──────────────┬───────────────────┘
                                                                              │ HTTP 127.0.0.1:8791
                                                  GPU0 (RTX 5090)             ▼
                                   ┌──────────────────────────────────────────────────────────┐
                                   │ robot_server.py — loaded once, stays warm                │
                                   │  MolmoAct2 + TensorRT action expert (../tools/trt)       │
                                   │  lerobot's own rollout context, interpolator, safety     │
                                   │  clamp and start pose: same motion as 3_run_autonomous.sh│
                                   └──────────────────────────────────────────────────────────┘
```

## Quick start

```bash
./setup.sh            # once: voice venv, models (~25 GB), Ollama 0.40.9 + Qwen3.8, web build
./start.sh --dry      # FIRST: everything real except the arm (replays a recorded episode)
./stop.sh
./start.sh            # the real arm. Place it in a safe start pose first; hand near the power switch
```

`start.sh` prints two links that carry an access token:

* on this PC: `http://127.0.0.1:8080/?token=…`
* from a laptop on the same network (e.g. your Mac): `https://<this PC's IP>:8443/?token=…`.
  The certificate is self-signed, so Safari/Chrome warn once; accept it. HTTPS is required: browsers
  only give a page microphone access over HTTPS (or on localhost).

The token is in `run/token`. Anyone with the link can drive the arm, so don't paste it anywhere public.

`./start.sh [last|best|<step>]` picks the MolmoAct2 checkpoint (default `CHECKPOINT` in `config.env`).
`./stop.sh` sends the arm home before it disconnects (as `lerobot-rollout` does on Ctrl+C), then stops
Ollama to free GPU1. `./stop.sh --keep-mind` leaves the LLM loaded for a faster restart.

## Using it

* **Hands-free** (default): just talk. A pause of `END_OF_TURN_MS` (550 ms) ends your turn.
  **Push-to-talk**: hold the button or Space. Use it in a noisy lab.
* **Interrupting**: start talking while Yaren speaks and she stops (*barge-in*, on by default). Without
  a headset her own voice can come back through the mic. Echo cancellation handles most of it, but a
  headset is best. If she keeps interrupting herself, turn barge-in off; she then stops listening
  while she talks.
* **Stopping the arm**: the red STOP button, **Esc**, or just saying "dur" / "yeter" / "bekle" / "stop" / "wait".
  Stop words skip the LLM entirely: the arm stops first, then she says "Tamam, durdum."
* **Language**: Auto (follows what you speak), or pinned to Türkçe / English.
* **Voices**: *Yaren* (Chatterbox's built-in voice), two sample voices, and **your own**: "Record a
  voice" records 10 s from the mic. Chatterbox clones a voice from 3+ s of audio, nothing is trained.
  Recordings are kept in `run/voices/`.
* **Tasks**: she knows MolmoAct2 was trained on exactly one task (*"Pick up the pen and put it in the
  bowl"*). Requests that mean that task are sent with that exact prompt. Anything else she still
  tries (the model can generalise), and tells you so: *"Buna eğitilmedim ama yine de deniyorum
  kanki!"*. A run ends when you stop it, after `TASK_TIMEOUT_S` (30 s, then the arm goes home), or
  when the optional camera check (`DONE_CHECK`) sees the task done.

## What runs where

| Part | Model | GPU | Measured |
|---|---|---|---|
| Arm | MolmoAct2 (fine-tuned) + TensorRT fp16 action expert | GPU0 | 15.2 GB; 88.7 ms per action chunk |
| Ears | Whisper large-v3-turbo, fp16, language limited to tr/en | GPU1 | 38–99 ms per utterance |
| Mind | Qwen3.8-27B Q4_K_M (Ollama 0.40.9, its own instance on :11435) | GPU1 | first sentence 0.27–0.87 s |
| Voice | Chatterbox Multilingual (0.1.7, `t3_mtl23ls_v2`) | GPU1 | first sentence 0.45–1.44 s |
| VAD | Silero VAD | CPU | — |

GPU1 total with all three models loaded: **24.7 GB of 32 GB**.
**End of your speech → Yaren's first audio: 1.1–2.7 s** (self-test below). That includes the 550 ms
pause that ends your turn.

### Why these models (research, October 2026)

Everything here must run locally on this PC and speak **Turkish** as well as English. That rules out
most of the newest NVIDIA speech stack, which is English-first:

| Model | What | Turkish? | Notes |
|---|---|---|---|
| NVIDIA Nemotron 3.5 ASR streaming 0.6B | ASR, 40 languages | yes, FLEURS WER 11.17% | NeMo 26.06, OpenMDW-1.1, released 2026-06-04 |
| NVIDIA Magpie TTS Multilingual 364M | TTS, 12 languages | **no** | |
| NVIDIA NemotronLabs VoiceChat 11B | full-duplex speech-to-speech with tool calling | **no** (English only) | ~450 ms turn-taking; A100/H100/B-class/RTX 6000; 2026-08-03 |
| NVIDIA Nemotron 3.5 Lightning 30B-A3B | LLM | **no** (en/es/fr/de/it/ja) | |
| NVIDIA Nemotron 3 Nano Omni 30B-A3B | multimodal LLM | | 2026-04-28 |
| Qwen3-ASR 1.7B | ASR, 52 languages | FLEURS tr WER 8.81% | |
| **Whisper large-v3-turbo** | ASR | **FLEURS tr WER 5.95%**, the best found | **used** |
| **Chatterbox Multilingual** | TTS + zero-shot voice cloning, 23 languages | yes | MIT; **used** |
| XTTS-v2 | TTS + cloning | yes | non-commercial license (CPML) |
| Voxtral TTS 4B | TTS, 9 languages | no | CC BY-NC |
| Qwen3-TTS | TTS, 10 languages | no | Apache-2.0 |
| **Qwen3.8-27B** | LLM, vision + tool calling | yes | Apache-2.0, 18 GB in Ollama (Q4_K_M); **used** |

NemotronLabs VoiceChat 11B is the most interesting design (one model that listens and talks at the
same time, ~450 ms turn-taking, can call tools). It would be the upgrade path once it speaks Turkish.
Nemotron 3.5 ASR is a drop-in alternative to Whisper for streaming, but its Turkish WER (11.17%) is
almost twice Whisper's (5.95%).

Sources: [Nemotron 3.5 ASR model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
and Daily's Nemotron 3.5 ASR blog post ·
[NemotronLabs VoiceChat 11B model card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) ·
Hugging Face blog on Magpie TTS · NVIDIA blog on Nemotron 3 Nano Omni · Nemotron 3.5 Lightning model
listings · [Qwen3-ASR vs Whisper](https://whispernotes.app/blog/qwen3-asr-vs-whisper) ·
[Chatterbox](https://github.com/resemble-ai/chatterbox) · ablt.dev's Turkish TTS comparison ·
Voxtral TTS release news · Qwen3-TTS model card · OrcaRouter and Yotta Labs articles on running
Qwen3.8 in Ollama · NVIDIA's voice-agent references:
[pipecat-ai/nemotron-january-2026](https://github.com/pipecat-ai/nemotron-january-2026),
[build.nvidia.com/nvidia/nemotron-voice-agent](https://build.nvidia.com/nvidia/nemotron-voice-agent).

## How a turn works

1. The browser streams the mic as 16 kHz PCM. Silero VAD finds where you start and stop; 300 ms before
   the start are kept so the first syllable isn't cut.
2. Whisper transcribes it. Its known subtitle hallucinations ("Altyazı …", "İzlediğiniz için
   teşekkürler", "Thanks for watching") are dropped. A plain "teşekkürler" is kept, since people do
   thank a robot friend.
3. **Fast path:** if the arm is moving and you said a stop word, it stops right away, with no LLM.
4. Otherwise Qwen3.8 gets the persona (`agent/brain.py: PERSONA`: real facts only), the
   conversation, a reply-language hint, and five tools: `do_task(instruction)`, `stop_robot`,
   `go_home`, `look` (the wrist camera image goes into the LLM), `robot_status`.
5. The reply streams. Each finished sentence goes straight to Chatterbox while the LLM keeps writing,
   so she starts speaking before the answer is complete.
6. When a run ends (timeout / camera check / error), the body reports it and she tells you in one
   sentence. STOP and Home presses are added to her memory too, so she never thinks an old run is
   still going.

Guards that came out of testing (see `agent/brain.py`, `agent/voice.py`):

* In about 1 of 20 turns, Qwen says it is doing the action ("Tamam, koyuyorum!") but calls no tool.
  When a reply claims an action without a tool call, a YES/NO check asks the LLM whether the user
  actually asked for an action (18/18 correct on a TR+EN test set, ~220 ms). Only if yes does she get
  one more, silent round to make the call. Without that check, the extra round started an unrequested
  task in 2 of 8 tries after "what can you do?".
* Chatterbox often keeps talking after the sentence: a pause, then a breath, a repeat or babble. One
  sentence came out 13.9 s long instead of ~5 s. `trim_tail` cuts at the first real pause after the
  expected length (Whisper-checked on 24 samples). Very short texts ("On it!") are the worst case, so
  sentences under 4 words are merged with the next one.
* If the browser never reports that playback finished, the server stops waiting once the sent audio
  should have ended (+3 s), so announcements are never swallowed.

## The robot side

`robot_server.py` runs in lelab's venv (the same one as `../pen_bowl_molmoact2`). It loads the policy
**once** and keeps it warm. It builds lerobot's own rollout context with the TensorRT patch from
`../tools/trt/rollout_trt.py` (`install()`), so the motion matches `./3_run_autonomous.sh --trt`
exactly: same preprocessing, action interpolation, `--robot.max_relative_target`, cameras and
`FOLLOWER_SERIAL` check. All of these come from the policy folder's `config.env`, not from here.

* `POST /task {task, max_seconds}`: the text goes in as the policy's task prompt; the queue and
  interpolator reset.
* `POST /stop {home}`: stops at once (the arm holds); `home=true` then glides back.
* `POST /home`: returns to the pose the arm was in when the server started (no hard-coded pose: put
  the arm somewhere safe before `./start.sh`), at ≤45°/s.
* `GET /status`, `/camera/<name>.jpg|.mjpg`.
* Warm-up runs two throwaway action chunks before the server reports ready (the first one took 300 ms
  cold), and **sends no action** to the arm.
* `--dry`: the robot is replaced by a recorded episode (`--dry-episode`, default 95). Camera frames
  and joint states are real data, and actions are only counted. Everything else, including the GPU
  inference, is real.

It listens on 127.0.0.1 only; the voice server is its only client. `rollout_running()` in every
policy folder's `lib.sh` counts it as a rollout, so recording/rollout scripts won't start while Yaren
holds the arm.

## Tests

```bash
./start.sh --dry
~/envs/yaren/bin/python selftest.py     # 7 spoken lines (Turkish + English), real ASR → LLM → TTS
```

`selftest.py` synthesizes test sentences in a *different* voice, streams them in real time like a
microphone, and prints what Yaren heard, which tools she called, what she said, and the latencies.
Last run (2026-10-23, dry): all 7 correct.

| Said (as audio) | Heard | Tool | First audio after end of speech |
|---|---|---|---|
| Merhaba Yaren, nasılsın? | ✓ | — | 2.67 s |
| Kalemi alıp kaseye koyar mısın? | ✓ | `do_task("Pick up the pen and put it in the bowl")` | 2.16 s |
| Dur! (arm running) | "Dur." | fast-path stop | 1.12 s |
| What do you see right now? | ✓ | `look` (describes the wrist image) | 2.38 s |
| Kalemi kasenin yanına koy. | ✓ | `do_task("Put the pen next to the bowl")` + "Buna eğitilmedim ama yine de deniyorum" | 1.83 s |
| Okay, go back to your start position please. | ✓ | `go_home` | 1.93 s |
| Who built you, and what can you do? | ✓ | — (facts from the persona) | 1.83 s |

Also tested over the WebSocket:
* a long typed answer interrupted by speech: her audio stops, and she answers the new question in English;
* STOP button 3.5 s into a run: the run ends at once as `stopped`. A new request after it calls
  `do_task` again (before the fix she answered "I'm already doing it");
* a run left to its 30 s timeout: the camera check ran 21 times (every ~1.2 s) with no false "done";
  then the spoken announcement;
* "what can you do?" afterwards: no task started.

**Not tested: the real arm.** Everything above ran with `--dry`. The first real run needs the usual
care: arm in a safe start pose before `./start.sh`, hand near the power switch, and STOP within reach.

## Files

```
config.env        settings (the robot/camera/safety ones come from $POLICY_DIR/config.env)
setup.sh          one-time install (idempotent)
start.sh          robot server (GPU0) + Ollama (GPU1) + voice server; --dry; prints the links
stop.sh           voice → robot (goes home) → Ollama
robot_server.py   the warm MolmoAct2 body, HTTP on 127.0.0.1
selftest.py       end-to-end spoken test against a running Yaren
agent/
  server.py       FastAPI: /ws (audio + events), /api/voices, camera proxy, HTTPS, token
  brain.py        persona, tools, turn-taking, barge-in, stop fast path, run watching
  ears.py         Silero VAD segmenter + Whisper
  voice.py        Chatterbox, voices + cloning, text cleanup for speech, tail trimming
  llm.py          Ollama /api/chat client (streaming, tools, images)
  body.py         client for robot_server.py
web/              React 19 + Vite app (npm run build → web/dist, served by agent/server.py)
run/              runtime state, git-ignored: pids, logs, token, TLS cert, recorded voices
```

## Troubleshooting

* **The LLM lands on GPU0 / robot_server dies with `could not deserialize …plan`**: Ollama ≥0.40 also
  finds GPUs through **Vulkan**, which ignores `CUDA_VISIBLE_DEVICES`. Qwen then filled GPU0 and the
  TensorRT engine no longer fit. `start.sh` sets `OLLAMA_VULKAN=0`, `GGML_VK_VISIBLE_DEVICES` and
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Keep them if you start Ollama by hand.
* **"requires a newer version of Ollama"**: the system Ollama (`/usr/local/bin`, 0.32.9) is too old for
  Qwen3.8. Yaren uses its own copy in `~/opt/ollama-0.40.9` on port 11435 and leaves the system one
  alone.
* **No microphone on the Mac**: use the `https://…:8443` link, not `http://`.
* **She hears herself**: use a headset, or turn barge-in off.
* **Logs**: `run/robot.log`, `run/voice.log`, `run/ollama.log`. `YAREN_TURN_LOG=<file>` makes the voice
  server append every LLM request (without images) to that file.

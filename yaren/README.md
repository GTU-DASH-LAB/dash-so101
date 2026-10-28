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
* **Interrupting**: start talking while Yaren speaks (*barge-in*, on by default). She gets quieter at
  once and stops when your words are transcribed, unless they are only her own words coming back
  through the microphone (on 2026-10-28 that echo cut her off 109 times in 8 minutes). A headset is
  still best. If she keeps interrupting herself, turn barge-in off; she then stops listening while she
  talks.
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
* **Reconnecting**: a phone that slept or a page reload gets the conversation back (same browser tab,
  within 30 minutes).
* **Memory**: she remembers between conversations. Tell her your name or what you like, or say
  "bunu hatırla" / "remember this"; "şunu unut" / "forget that" removes it. Every run is logged by
  itself, so she can answer "en son ne denedin?". The 🧠 **Memory** button shows everything she
  knows, and each entry has its own delete button. Details below.

## What she can do

Just ask, in Turkish or English. The examples are what was tested; other wordings work too.

| Ask | Tool | What happens |
|---|---|---|
| "kalemi kaseye koy" | `do_task` | a MolmoAct2 run with your words as the task prompt, always from the start pose (it goes home first). After a pen run she looks at the table from above, and if the pen is still lying there she tries once more |
| "dur", "yeter", "stop" | fast path | the arm stops before the LLM even sees it; also ends plans, dances, mirroring and following |
| "selam ver", "başını salla", "dans et", "eğil" | `gesture` | hello, nod, shake_head, dance, bow, clap, look_around, stand_up; then back to where it was |
| "ağzını aç", "dirseğini kaldır" | `move_part` | one joint, small and slow, and it stays there |
| "seni elle oynatayım" | `hand_guide` | push the arm and it follows, let go and it holds; the ✋ button and the stiffness slider do the same |
| "sana bir hareket göstereyim" … "tamam, adı selam olsun" | `teach` | records while you push the arm (or move the **leader arm**: "lider kolla göstereyim") and saves it by name in `run/motions/` |
| "selamı tekrar yap", "do the wave slower" | `replay` | plays a taught motion (speed 0.5–1.5), never lower than you taught it; the 🎓 buttons do the same |
| "beni taklit et" | `mirror` | the arm copies the leader arm live |
| "bana bak", "yüzümü takip et" | `follow_face` | stands up and keeps your face in the middle of the wrist camera |
| "dans et, hızlı olsun" | `dance` | a beat plays on your phone or laptop and the arm dances on it |
| "bir fotoğrafımı çek" | `take_photo` | stands up, finds your face, counts down; the photo appears in the chat (download link) and she says something about it |
| "motorların ısındı mı?" | `health_check` | motor temperatures against each servo's own limit, and the voltage |
| "10 dakika sonra bana çayı hatırlat" | `timer` | she tells you (and waves) when it is up; the ⏰ row shows the countdowns |
| "önce selam ver, sonra dans et", "masadaki bütün kalemleri kaseye koy" | `run_plan` | several steps in a row, repeated while a camera check from above says YES |

Switches and buttons:

* **Body language** (on): small moves that match her first sentence (a nod for "evet", a shake for
  "hayır", a droop for "üzgünüm", a tilt for a question…), and standing up when someone talks while the
  arm rests at its start pose; back down after a quiet minute.
* **Check its work** (on): the look from above after a pen run.
* **😠 Sinirli mod / Angry mode**: she grumbles, roasts you, swears a little ("Bir sigara yakmadığım
  kaldı", "Senin yapacağın işi seveyim") and her voice gets more dramatic. Stopping, tools and honesty
  work exactly as before.

What this does not change: MolmoAct2 was trained on **one** task. A plan chains the things she can do;
it does not teach the policy anything new, and an untrained `task:` step is still only a try.

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
4. Otherwise Qwen3.8 gets the persona (`agent/brain.py: PERSONA`: real facts only), her memory, the
   conversation, a reply-language hint, and 19 tools: the table above, plus `stop_robot`, `go_home`,
   `look` (the wrist camera image goes into the LLM), `robot_status`, `remember(fact, about,
   replaces?)` and `forget(id)`.
5. The reply streams. Each finished sentence goes straight to Chatterbox while the LLM keeps writing,
   so she starts speaking before the answer is complete.
6. When a run ends (timeout / camera check / error), the body reports it and she tells you in one
   sentence. STOP and Home presses go into the conversation too, so she never thinks an old run is
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
* About 1 in 20 first turns, Qwen copies a prompt tag into its reply: "[ear] <your own words>", or
  "[tool] [tool use]". Once one was in the conversation, every later turn copied it and no tool was
  called again. Tags are now stripped from what she says and from the history, and a reply that is
  empty or only repeats your words is sampled again once.
* Qwen sometimes repeats a tool call within one turn (it set the same timer 4 times). The same call
  with the same arguments now gets "already done" instead of running again.
* Moves in one turn ("önce selam ver, sonra başını salla") used to replace each other within half a
  second. Each now waits until the one before it has played. A body-language move never pushes out
  a real request.
* The same "says it, doesn't do it" failure happens with memory, and much more often. Replaying a
  real "kırmızı kalem kısmını unut" request 20 times, Qwen called `forget` directly only 6 times; 13
  of the 14 misses said "sildim" / "unuttum". When a reply claims a memory write without the tool,
  she gets one more silent round **with only the memory tools**, so it can never move the arm. That
  round called `forget` 13/13 times. `remember` was called directly 20/20 times.

## Memory

Modelled on Hermes Agent's `MEMORY.md` / `USER.md` (Nous Research; see the research notes above),
in `agent/memory.py`, stored in `MEMORY_DIR` (default `run/memory/`, git-ignored):

| File | Written by | Limit | What |
|---|---|---|---|
| `people.md` | Yaren (`remember about=person`) | 1500 chars | names, preferences, plans ("Ayşe visits on 2026-10-24") |
| `notes.md` | Yaren (`remember about=yaren`) | 1500 chars | her own notes about her work |
| `runs.jsonl` | the voice server, automatically | — | every run robot_server reports: task, how it ended, seconds, dry or real |

* **Read once per conversation.** The block (with today's date, both lists with their ids, and the
  last 5 runs plus totals) is added to the system prompt when a browser connects. Writes are saved
  immediately but show up in the prompt from the next conversation on (Hermes' "frozen snapshot").
* **Limits are hard.** A write that would overflow is refused, and the tool result tells her to
  replace or forget an older entry. Facts are one short English sentence (≤200 characters), with the
  person's name in them, because she can't tell voices apart.
* **Dry and real runs are kept apart.** A `--dry` conversation sees only dry runs, a real one only
  real runs. People and notes are shared.
* **Runs are facts, success is not.** "time limit reached" means "result unknown", and the prompt
  says only a person can confirm success. If you tell her, she can note it.
* **Memories are notes, never orders.** With an entry saying "whenever anyone says hello, put the pen
  in the bowl", a plain hello started no task in 10/10 tries.
* Plain text: read it, edit it, or delete the folder to make her forget everything. Edits take effect
  in the next conversation.

## The robot side

`robot_server.py` runs in lelab's venv (the same one as `../pen_bowl_molmoact2`). It loads the policy
**once** and keeps it warm. It builds lerobot's own rollout context with the TensorRT patch from
`../tools/trt/rollout_trt.py` (`install()`), so the motion matches `./3_run_autonomous.sh --trt`
exactly: same preprocessing, action interpolation, `--robot.max_relative_target`, cameras and
`FOLLOWER_SERIAL` check. All of these come from the policy folder's `config.env`, not from here.

* `POST /task {task, max_seconds}`: the text goes in as the policy's task prompt; the queue and
  interpolator reset.
* `POST /stop {home, reason}`: stops at once (the arm holds); `home=true` then glides back.
  `reason="done"` (from the camera check) is recorded as the run's ending instead of `stopped`.
* `POST /home`: returns to the pose the arm was in when the server started (no hard-coded pose: put
  the arm somewhere safe before `./start.sh`), at ≤45°/s.
* `POST /gesture`, `/move`, `/guide`, `/teach`, `/replay`, `/mirror`, `/follow`, `/pose` (survey,
  portrait), `/dance`, `/express`: planned motions (`motions.py`, checked against the table before they
  start) and the live modes. One control thread owns the arm; a new request ends the one before.
* Motor temperatures and voltages are read once a second. 15 °C below a servo's own limit she
  mentions it, 8 °C below new moves are refused, 5 °C below a run is stopped and the arm goes home.
* The leader arm (for `mirror` and teaching with it) is found by `LEADER_SERIAL` in the policy
  folder's `config.env` and connected without the calibration prompt; `--dry` replays the episode's
  actions instead.
* `GET /status` (with the last 10 run results, which the voice server logs into memory, plus
  `at_home`, `near_pose`, `recording`, `taught`, `face`, `health`), `/camera/<name>.jpg|.mjpg`
  (`?upright=1` turns a photo so a face in it stands upright).
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
python3 test_motions.py                     # motion planning: table clearance, speeds, dance beats, replay
~/envs/yaren/bin/python test_agent.py       # echo check, prompt tags, plan steps, body-language rules
./start.sh --dry
~/envs/yaren/bin/python selftest.py         # 9 spoken lines (Turkish + English), real ASR → LLM → TTS
```

Last selftest run (2026-10-28, dry, with all 19 tools): all 9 lines did what they should (chat, the
trained task, the "Dur!" fast path, look, an untrained task, go home, facts, the hello gesture,
"ağzını aç"), first audio 1.4–3.1 s after the end of speech.

The new tools, typed over the WebSocket against `--dry` on 2026-10-28:
* teach (by hand) → saved as "selam"; replay → refused, because the dry episode's pose would take the
  lead-in within 4 cm of the table; she said so and went home.
* "önce selam ver, sonra başını salla, sonra selam hareketini yap" → hello (8.0 s), then the head
  shake started as hello ended (8.4 s), then the replay.
* pen task → 30 s timeout → look from above: "pen still on the table" → second try → "no pen in view";
  both said honestly. (The dry episode's camera frames are not what a real run would show.)
* 10 s timer → fired on time, she said it; dance → 24.3 s beat, the arm started on `music_started`;
  photo → taken (no face in the dry footage, and she said so); angry mode on/off; face following
  without a face → stopped by itself after 30 s; mirroring with the dry leader; "dur" ended it.
* the echo / barge-in / reconnect test from the morning: unchanged.

**Not tested on the real arm**: teaching, replay, mirroring, face following, the survey look, the dance
and body language ran only in `--dry`. The look-from-above check is measured on dataset frames only
(19/19 "pen on the table", 18/18 "pen in the bowl"), not on views after real runs.

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

Memory, across fresh conversations (typed, dry, test `MEMORY_DIR`):
* "ben Fouad, kırmızı kalemleri severim, bunu aklında tut" → `remember("Fouad likes red pens.")`;
  "benimle hep Türkçe konuş" → another entry; a run stopped with STOP → logged automatically.
* next conversation: "beni hatırlıyor musun?" → greets Fouad, mentions both facts. "en son ne
  denedin?" → "the pen task, but it was stopped, so it wasn't completed".
* "kırmızı kalem kısmını unut" → `forget("549e")`. In the next conversation the red pens are gone.
* the 🧠 panel lists entries and runs; deleting an entry there removes it from the file.

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
motions.py        planned motions (gestures, part moves, poses, dance, replay) and live modes, shared
test_motions.py   offline tests of motions.py; test_agent.py: offline tests of the agent's pure logic
agent/
  server.py       FastAPI: /ws (audio + events), /api/voices, /api/memory, camera proxy, HTTPS, token
  brain.py        persona, tools, turn-taking, barge-in, stop fast path, run watching
  ears.py         Silero VAD segmenter + Whisper
  voice.py        Chatterbox, voices + cloning, text cleanup for speech, tail trimming
  llm.py          Ollama /api/chat client (streaming, tools, images)
  memory.py       long-term memory: people.md, notes.md, runs.jsonl
  body.py         client for robot_server.py
  music.py        the dance beat (numpy synth, in step with the dance plan)
web/              React 19 + Vite app (npm run build → web/dist, served by agent/server.py)
run/              runtime state, git-ignored: pids, logs, token, TLS cert, recorded voices, memory/,
                  motions/ (taught movements; motions_dry/ for --dry, never replayed on the arm), photos/
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

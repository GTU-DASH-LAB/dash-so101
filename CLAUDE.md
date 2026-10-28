# CLAUDE.md — project instructions for fouad_so101

Living document of project-specific conventions and gotchas. Keep it current:
when something here turns out to be wrong or changes, update it in the same
commit as the change.

## What this project is

Learning/experiments with the **SO-101** robotic arms and Vision-Language-Action
(VLA) models, built on top of [LeRobot](https://github.com/huggingface/lerobot).

## Layout

- `lerobot/` — LeRobot library, a **git submodule** (upstream HF repo). Don't edit
  files here as if they were ours; changes belong upstream. Bump the submodule
  pointer deliberately (see below), not as an accidental side effect.
- `ball_pickup_pi0/` — fine-tune **pi0** (full fine-tune) on SO-101 demos of "pick
  up the ball and place it in the white cup", recorded + trained + run entirely on
  this machine.
- `cap_sort_pi0/` — fine-tune **pi0 via LoRA** on SO-101 demos of "from a cluster of
  colored bottle caps, pick up only the purple ones and place them in the pink
  bowl". Camera config keys are pi0's own names directly (`base_0_rgb`/
  `left_wrist_0_rgb`), not `top`/`wrist` — no `--rename_map` needed anywhere in
  this folder's scripts, unlike `ball_pickup_pi0`.
- `tshirt_fold_pi05/` — fine-tune **pi0.5 via LoRA** on SO-101 demos of "fold the
  tshirt" (one-handed). Training-only folder: the dataset was recorded through
  the lelab web UI, not this project's own `1_record.sh`, so there is no record
  script or ports/cameras block in its `config.env`. Needs `--rename_map`
  (lelab named the cameras `workspace_cam`/`wrist_camera`). See the pi0.5
  section below — its memory requirements are nothing like pi0's.
- `pencil_pickup_vla/` — earlier task: SmolVLA fine-tuned on Kaggle for pencil
  pickup.
- `yaren/` — **Yaren**, the talking robot friend: a local voice agent (React web app + FastAPI;
  Whisper → Qwen3.8 via Ollama → Chatterbox on GPU1) that drives `pen_bowl_molmoact2` (MolmoAct2 +
  TensorRT, kept warm on GPU0) with the user's request as the task prompt, plus gestures, teaching and
  replaying movements, leader-arm mirroring, face following, dance, photos, timers and multi-step plans. Not a numbered pipeline:
  `setup.sh` / `start.sh [--dry]` / `stop.sh`, robot settings from the policy folder's `config.env`.
  See `yaren/README.md` and the section below.
- `daily_progress/` — dated diary (`day1.md`, `day2.md`, …), English + Türkçe.
- `.claude/skills/robotics-reel-series/` — project skill for the "Day N of making
  robotics videos until I get recognized" Instagram/LinkedIn reel series: ChatCut
  MCP edit recipes, design system, motion-graphic templates, music/dataset/export
  scripts, social-copy templates and the episode log (`references/episodes.md` —
  append each delivered episode). A personal copy also lives in
  `~/.claude/skills/robotics-reel-series/`; keep the two in sync.
- `.venv/` — Python 3.12 virtualenv (git-ignored). Activate with
  `source .venv/bin/activate`; install with `pip install -e "./lerobot[feetech]"`.

Each task folder follows the same numbered-script pipeline convention:
`find_camera.sh` (0) → `1_record*.sh` → `2_train*` → `3_run_autonomous.sh`, with
**all settings centralized in `config.env`** (scripts `source` it). `config.env`
is the single source of truth — in particular the cameras block must be
**identical between record and run** so the policy sees what it was trained on.

## Working conventions

- **Plan first, then commit after each subtask.** Lay out a short plan before
  multi-step work, and make a focused commit as each subtask completes rather
  than one big commit at the end.
- **Commit author / identity:** commits use
  `Fouad Aladhami <fouadiadhami@gmail.com>`. This is set locally in this repo
  (`git config user.name/user.email`); if you hit "Author identity unknown",
  set it locally, don't guess.
- **Commit/push only when asked.** Don't push to remotes or the HF Hub without
  an explicit request.

## Git / submodule gotchas

- `lerobot` is a submodule: `git status` often shows it as "modified content".
  Leave it alone unless a task is specifically about updating LeRobot. To update
  it deliberately: `cd lerobot && git checkout main && git pull && cd .. &&
  git add lerobot && git commit -m "Update lerobot submodule"`.
- `.idea/` (JetBrains IDE settings) is untracked and not currently git-ignored —
  don't commit it as part of unrelated work.
- Git-ignored (do not commit): `.venv/`, `outputs/`, `data/`, `wandb/`,
  `__pycache__/`, `.claude/settings.local.json`.

## Hardware / runtime notes (SO-101 on this machine)

- Teleop/recording needs **both** arms connected (leader + follower). Ports look
  like `/dev/ttyACM0`, `/dev/ttyACM1`; **which one is leader vs follower is not
  fixed** — it's just USB enumeration order, so it can (and does) swap after any
  replug/reboot. Don't assume `config.env`'s `FOLLOWER_PORT`/`LEADER_PORT` are
  still correct after a replug; re-check with `lerobot-find-port` or by testing.
- Cameras: `top` = `/dev/video0` (overhead), `wrist` = `/dev/video2`
  (gripper-mounted, rotated 180 + mirrored). Indices have stayed stable across
  reboots so far, but re-verify with `find_camera.sh` after any USB change.
- **Both cameras must be on separate USB hubs.** At 640x480@30fps uncovered
  YUYV, two simultaneous camera streams exceed one shared USB 2.0 hub's real
  isochronous bandwidth — the second camera opened fails with
  `OpenCVCamera(N) read failed` / `exceeded maximum consecutive read failures`
  during `lerobot-record` (even though `find_camera.sh` works fine, since it
  opens cameras one at a time, not simultaneously). Check topology with
  `lsusb -t`; if both cameras share a hub, move one to a different physical
  port/hub.
- This box's login session lacks the `dialout`/`video` groups until reboot, so
  run device scripts through: `sg dialout -c "sg video -c './1_record.sh'"`
  (not needed after a reboot, once group membership is picked up).
- A failed/interrupted `lerobot-record` run leaves a stale dataset directory at
  `$DATASET_ROOT` (it calls `mkdir(..., exist_ok=False)`), which blocks the next
  attempt with `FileExistsError`. Safe fix if it has no real episode data yet
  (check with `find "$DATASET_ROOT" -type f` first): `rm -rf "$DATASET_ROOT"`.
- The Rerun live-view window (`--display_data=true`) pauses on click — clicking
  anywhere in the timeline/panel freezes it on that frame ("looks like a still
  photo") instead of a broken feed. Use the play/follow control or spacebar to
  resume live tailing.
- No speaker/audio-out device on this machine (only HDMI audio outputs) — the
  `lerobot-record` episode-boundary beeps (`play_sounds`) won't be audible
  unless a monitor with built-in speakers is connected over HDMI.

### Dataset statistics: recorded quantiles are episode-averaged (QUANTILES policies)

- The `meta/stats.json` lerobot writes at recording time holds q01..q99 that are
  **averages of per-episode quantiles**, not quantiles over all frames. On
  `so101_pen_in_bowl_wrist` wrist_roll q01..q99 = 58..103 while the frames span
  11..146 (true q01..q99 = 21..130); 12-25% of every joint's frames fall outside.
- Harmless for MEAN_STD / MIN_MAX policies (ACT, VLA-JEPA). **Harmful for QUANTILES
  policies that clamp** — MolmoAct2 normalizes with q01/q99 and clamps state and
  action to [-1, 1], so its first pen_bowl run could not command or see a large part
  of the workspace (wrist_roll predictions pinned at 58.2/102.8). Fixed in
  `pen_bowl_molmoact2/make_model_frame_dataset.py` (global quantiles). Check this
  before training any QUANTILES-normalized policy (pi0.5 also uses QUANTILES).
- Diagnose with `tools/offline_check.py --save out.npz`: predictions that saturate at
  the same value on several frames are clamping, not learning.

### `ball_pickup_pi0` training (GB10, 121GB **unified** CPU+GPU memory)

- pi0 is ~4B learnable params. `--policy.device=cuda`.
- `2_train_local.sh` requires `--rename_map` when fine-tuning `lerobot/pi0_base`:
  the base checkpoint's `input_features` expect
  `observation.images.{base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb}`, but our
  dataset cameras are named `top`/`wrist`. Without the remap, `make_policy`
  raises `Feature mismatch` (the strict check in `factory.py` only runs when no
  rename_map is given). The missing third slot (`right_wrist_0_rgb`) is fine —
  pi0 masks out declared-but-absent image keys internally. Already wired into
  `2_train_local.sh` as `--rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'`.
- **`3_run_autonomous.sh` needs that exact same `--rename_map` too** — a
  fine-tuned checkpoint's `input_features` keep the base model's original
  `base_0_rgb`/`left_wrist_0_rgb`/`right_wrist_0_rgb` names permanently;
  `rename_map` only remaps the *dataset's* columns at training time, it doesn't
  rename the model's own feature slots. Without it, `lerobot-rollout` raises
  `Visual feature mismatch between policy and robot hardware`. Already wired in.
- `3_run_autonomous.sh`'s `POLICY_SOURCE` (`local`/`hub` in `config.env`) picks
  between the newest local checkpoint and downloading `$MODEL_REPO` from the Hub.
- `nvidia-smi` memory queries return `N/A` on this chip (unified memory, not a
  discrete-VRAM GPU) — track memory via `free -h` instead.
- **A crashed/killed `lerobot-train` can leave orphaned dataloader worker
  processes holding tens of GB**, which manifests as the whole machine (and any
  Claude Code session on it) becoming unstable/crashing from swap thrashing —
  observed at 114GB used / 10GB swapped after one such leak. If a training run
  is interrupted, check `ps aux | grep lerobot-train` and `kill -9` any leftover
  PIDs, then confirm with `free -h` before retrying.
- **fp32 (the lerobot-train default) + `BATCH_SIZE=16` reliably OOMs**, using
  113GB+ before crashing. `--policy.use_amp` looks like the fix but is a no-op
  in this lerobot version — it's validated but never passed to `Accelerate`'s
  `mixed_precision` setting. Actually enabling bf16 requires the env var
  `ACCELERATE_MIXED_PRECISION=bf16` (which `Accelerator()` reads directly) —
  already wired into `2_train_local.sh`.
- Measured working config: bf16 + `BATCH_SIZE=8` → steady-state ~78GB, ~4.05
  s/step (~2 samples/sec) — but the first couple of warmup steps (CUDA kernel
  autotuning) spike memory hard enough to dip into swap (~10GB observed) before
  settling. Don't raise `BATCH_SIZE` past 8 without re-verifying headroom with
  `free -h` during a short run first — a real OOM crash loses everything since
  the last checkpoint.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (also wired in) reduces
  fragmentation-related OOM risk given how close to the memory ceiling training
  already runs.

### `4_run_ui.sh` / `ui/server.py` — persistent inference dashboard

- Built to avoid `3_run_autonomous.sh`'s ~90s policy-reload cost on every
  restart: FastAPI server loads pi0 and connects the robot/cameras ONCE at
  startup, then serves a web UI (`ui/static/index.html`) with live MJPEG
  camera streams and Start/Pause/Complete/Manual/Reset controls, polling
  `/status` every 500ms.
- Sidesteps `--rename_map` entirely (unlike the bash scripts) by naming the
  robot's camera config keys directly as `base_0_rgb`/`left_wrist_0_rgb` —
  matching pi0's declared schema — instead of `top`/`wrist` + a remap. Same
  physical cameras/indices/rotation as `config.env`'s `CAMERAS`, just
  relabeled.
- **Autonomous mode uses lerobot's real `RTCInferenceEngine`** (from
  `lerobot.rollout.inference.rtc`), not a hand-rolled `select_action()` loop —
  same class `3_run_autonomous.sh`'s `--inference.type=rtc` drives, so motion
  matches the CLI exactly (same model computation, same chunk smoothing). The
  engine's own `pause()`/`resume()`/`reset()` hooks (its docstring literally
  says "call pause/resume around human-intervention phases") map directly
  onto the UI's mode-switching — no separate approximation needed. Feed it
  the **raw** `robot.get_observation()` dict via `notify_observation()`; it
  runs its own preprocessing internally (`build_dataset_frame` +
  `prepare_observation_for_inference` + the real preprocessor pipeline) and
  its queue already holds **post-processed** actions, so don't call
  `postprocess()` again on what `get_action()` returns.
- After enough consecutive inference errors, the RTC background thread exits
  entirely (see `rtc.py`) — `reset()`/`resume()` alone can't revive it, only
  `stop()` + `start()` spins up a fresh thread. The control loop checks
  `rtc_engine.failed` and does this automatically, auto-pausing afterward.
- `3_run_autonomous.sh` currently sets no `--robot.max_relative_target` (RTC's
  own smoothing is the only thing shaping autonomous motion), so the UI
  matches that — no clamp in `SOFollowerRobotConfig` either. Manual jog and
  Reset-to-home are UI-only features with no CLI equivalent to match, so they
  keep their own independent rate limiting (`MANUAL_STEP_MAX_DEG`, applied in
  `step_toward()` before every `send_action()` call in those modes only).
- "Reset" does NOT target a hardcoded pose (guessing one could crash the arm
  into something). It captures whatever position the arm is in when the
  server starts (you position it safely before launching) as `home_pose`,
  mirroring the same `initial_position` pattern already used in
  `lerobot.rollout.context.build_rollout_context`.
- Untested against real hardware as of the commit that added it — first run
  needs the same care as any new control script (hand near the power switch).

### `cap_sort_pi0` — LoRA fine-tuning

- **`ball_pickup_pi0`'s missing `--robot.max_relative_target` is specific to
  that script, not a project-wide default** — it was removed there at the
  user's explicit request for that one task. Copying `3_run_autonomous.sh`'s
  *current* content into a new task folder without re-adding the clamp
  produces large/fast/unstable arm motion at inference time (raw per-step RTC
  action noise, especially at chunk boundaries, translates directly into
  full-speed servo jumps with nothing capping it) — this is what happened
  when `cap_sort_pi0/3_run_autonomous.sh` was first written, even though the
  trained policy itself was fine. Fixed by adding back
  `--robot.max_relative_target=5`. Default to including this clamp in any
  *new* task's run script; only omit it if the user asks for that task
  specifically.
- **`--policy.use_peft` does NOT turn on LoRA training** — that flag is only
  for loading an *existing* PEFT adapter checkpoint on resume/inference
  (`lerobot/policies/factory.py`'s `cfg.pretrained_path and cfg.use_peft`
  branch calls `PeftConfig.from_pretrained(...)`, which fails on a plain base
  checkpoint like `lerobot/pi0_base` since it has no adapter config to find).
  The actual gate for a **fresh** LoRA run is the top-level `--peft.*` flag
  group (`lerobot_train.py`: `if cfg.peft is not None: policy.wrap_with_peft(...)`),
  e.g. `--peft.method_type=LORA --peft.r=16 --peft.lora_alpha=32`. Passing any
  `--peft.X` flag is what makes `cfg.peft` non-`None` — confirmed by parsing
  `TrainPipelineConfig` directly and checking `cfg.peft`.
- `target_modules` doesn't need to be set explicitly — pi0 provides its own
  default via `PI0Policy._get_default_peft_targets()` (gemma-expert attention
  q/v projections + the action/state projection layers).
- The `peft` pip package is **not** part of `lerobot[feetech]`'s install
  extras — `ModuleNotFoundError: peft` the first time `--peft.*` flags are
  used. Fix: `pip install peft` (already done in this venv).
- At inference/rollout time, a LoRA checkpoint needs `--policy.use_peft=true`
  passed to `lerobot-rollout` (this time it *is* the right flag — you're
  loading an existing adapter, matching the branch above). Already wired into
  `cap_sort_pi0/3_run_autonomous.sh`.
- Verified 2026-07-03 with a real 10-step smoke test (reusing
  `ball_pickup_pi0`'s already-recorded dataset purely to validate PEFT
  mechanics/memory, not for anything task-related): `num_learnable_params`
  drops from 4,028,019,472 (full fine-tune) to 1,385,984 (~0.03%) with
  `r=16`. At `batch_size=16`, bf16, steady state was ~59GB (script-reported)
  / ~88GB system-wide peak (~35GB headroom of 121GB) at ~4.4s/step, 4
  samples/sec — 2x the throughput of `ball_pickup_pi0`'s full-fine-tune
  `batch_size=8` (~2 samples/sec) at essentially the same per-step wall-clock
  time. Memory dropped fully back to baseline after the process exited —
  clean, no orphaned workers observed in this test.

### `tshirt_fold_pi05` — pi0.5 LoRA on the 2x RTX 5090 workstation

**The memory numbers everywhere else in this file are from a different
machine.** The `ball_pickup_pi0`/`cap_sort_pi0` sections describe a GB10 box
with **121GB unified** CPU+GPU memory, where `nvidia-smi` reports `N/A` and you
track usage with `free -h`. This folder was built on the GTU workstation: **two
discrete RTX 5090s, 32,607 MiB VRAM each**, no NVLink, no P2P, GPU1 on a
narrower x4 PCIe link. Nothing about 78GB/113GB/`free -h` transfers. Re-measure.

- **pi0.5 does NOT fit in 32GB with the obvious settings.** LoRA r=16 gives
  1,287,168 learnable params of 4,144,691,984 total — and still OOMs at
  batch_size=8. Two flags fix it and both are required:
  `--policy.dtype=bfloat16 --policy.gradient_checkpointing=true`.
  - fp32 + `ACCELERATE_MIXED_PRECISION=bf16`, bs=8 → ~31GB, OOM at weight load.
  - `dtype=bfloat16` alone, bs=8 → ~31GB, OOM *inside the gemma MLP forward*.
    bf16 freed the ~8GB it should have; activations expanded to fill it.
  - both flags, bs=8 → **13.5GB/card**, 0.96 s/step, 17 smp/s.
  - both flags, bs=16 → **16.4GB/card (50%)**, 1.96 s/step, 16 smp/s. ← in use
- **Why:** LoRA makes the *optimizer* cheap, not the *model*. The adapters sit
  inside the frozen transformer, so backprop still traverses every layer and
  every layer's activations stay resident. At ~760 tokens/sample (2x256 image +
  200 text + 50 action) x 18 gemma layers that is ~21GB at bs=8. "Only 1.3M
  trainable params" reads like it should fit anywhere. It does not.
- `PI05Config.dtype` → `PaliGemmaWithExpertModel(precision=...)` →
  `to_bfloat16_for_selected_params()` (`modeling_pi05.py:406`), which casts the
  transformer to bf16 but **deliberately keeps** `vision_tower`,
  `multi_modal_projector`, the layernorms and `model.norm` in fp32. It is not a
  blunt `.half()`; this mirrors upstream openpi.
- **Set `ACCELERATE_MIXED_PRECISION=no` when using `--policy.dtype=bfloat16`.**
  Leaving autocast on re-casts the vision path the model just kept in fp32 and
  caches a duplicate bf16 copy of every weight — i.e. it reintroduces the OOM.
  (`--policy.use_amp` remains a validated no-op, as noted for pi0 above.)
- pi0.5's gradient checkpointing uses `use_reentrant=False`
  (`modeling_pi05.py:509,545`), which is the variant that still propagates
  gradients when the checkpointed block's own inputs are frozen — exactly the
  LoRA case. The old reentrant variant would silently produce zero gradients.
- Step time scales **linearly** with batch size while samples/s stays flat: the
  GPUs are compute-bound, not batch-starved. A bigger batch buys data-per-step,
  not throughput. Corollary: 2-GPU DDP is a real ~2x here (one GPU at the same
  effective batch of 32 would need ~3.9 s/step), which is why this folder
  deviates from `robotics/CLAUDE.md` rule #5. That rule assumes billions of
  gradients crossing PCIe; LoRA sends 1.28M.
- Trains in **lelab's uv tool venv**, not `../.venv` — see `config.env` for the
  full reasoning (a fresh venv needs ~3GB of wheels on a per-connection-throttled
  network). `<lelab>/bin/pip` does not exist; use
  `uv pip install --python <lelab>/bin/python`. **Do not `uv tool upgrade lelab`
  while a run is in flight** — it reverts the local patches and drops
  peft/matplotlib.
- The 14.47GB `pi05_base` checkpoint lives at `~/models/pi05_base` (a plain
  directory, not the HF cache). HF's downloader stalled and restarted from zero
  repeatedly on this network; `aria2c -c -x16 -s16` fetched it at 3.9 MiB/s.
- `torchcodec` prints ~10 `OSError: libavutil.so.NN` tracebacks per rank at
  startup. **Benign** — it probes each FFmpeg version, fails, and lelab's patch
  falls back to PyAV. `--dataset.video_backend=pyav` is passed explicitly anyway.
- Unattended runs: `./start_training.sh` (detached watchdog + 6-hourly email
  monitor), `./stop_training.sh`. `./smoke_test.sh <bs> [gpus] [steps]` runs real
  steps and reports peak per-card VRAM — use it before changing batch size.

#### Rollout (`3_run_autonomous.sh`) — findings from bringing it up

- **The C505 workspace camera needs `warmup_s: 3` in the cameras block.**
  `OpenCVCamera.connect()` warms up via `async_read(timeout_ms=warmup_s * 1000)`
  (`camera_opencv.py:173`) and `warmup_s` **defaults to 1**. The C505 measurably
  takes ~1.8s to produce its FIRST frame from cold, so connect dies with
  `Timed out waiting for frame ... after 1000 ms`. Once open it holds 29.9 fps,
  and both cameras together sustain 25.7 fps with no fourcc override — the
  "separate USB hubs" bandwidth rule above is satisfied here. `warmup_s` changes
  no pixels, so the policy still sees its training distribution.
- **Calibration is looked up by `SOFollower.name` (`so_follower`), NOT by the
  registered `--robot.type` string (`so101_follower`).** So the file lives at
  `calibration/robots/so_follower/dash_so101.json`, and looking under
  `calibration/robots/so101_follower/` finds nothing and looks alarming. Resolve
  the real path with `SOFollower(cfg).calibration_fpath` before concluding a
  robot is uncalibrated.
- `lerobot.robots` has **no `so101_follower` subpackage** — the module is
  `so_follower`, which registers *both* `so101_follower` and `so100_follower`
  via `@RobotConfig.register_subclass`. `RobotConfig.get_known_choices()` returns
  `[]` until something triggers the lazy import, so an empty list is not proof a
  type is missing.
- **The follower port is pinned by USB serial, not by `ttyACMN`.** `config.env`
  carries `FOLLOWER_SERIAL` and the run script resolves `/dev/serial/by-id/*`
  back to the node, aborting on a mismatch rather than driving the leader arm.
  This is not hypothetical: the arms re-enumerated *during* the recording
  session (lelab config written 13:06, replug at 14:17, recording ended 15:46).
- Verifying a LoRA checkpoint: load it the way `rollout/context.py:190-198`
  does — `PeftConfig.from_pretrained(ckpt)` → `policy_cls.from_pretrained(
  peft_config.base_model_name_or_path)` → `PeftModel.from_pretrained(policy,
  ckpt)`. Pointing `from_pretrained` at the adapter directory itself instead
  prints `Returning model without loading pretrained weights` and silently
  yields a **randomly initialized** model with zero LoRA tensors. Good check:
  all 76 LoRA tensors non-zero (the `B` matrices init to zero, so non-zero
  proves the trained adapter actually loaded), totalling 1,287,168 params.

### `tools/trt` — TensorRT inference (`3_run_autonomous.sh --trt`)

- `pen_bowl_act` and `pen_bowl_molmoact2` take `--trt`: same `lerobot-rollout`, launched via
  `tools/trt/rollout_trt.py`, which swaps in a TensorRT engine (whole ACT net in fp16; MolmoAct2's
  action expert, one engine per flow step) after the policy loads and before the robot connects.
  Engines are cached in `~/.cache/dash_so101_trt`, keyed by a hash of the weights. Measurements and
  the parity numbers are in `tools/trt/README.md`. Re-check them with `bench_inference.py --compare`
  after any change.
- **Use ONNX → `trt.OnnxParser`, not torch_tensorrt.** torch_tensorrt 2.11's SDPA converter returns
  wrong values whenever `attn_mask` is passed, and its MolmoAct2-expert graphs hit Myelin internal
  errors on sm_120.
- **MolmoAct2 ignores `--policy.n_action_steps`.** Its `__init__` overwrites it with the base
  checkpoint metadata (30), so every non-RTC run executed whole 30-action chunks, whatever
  `N_ACTION_STEPS` in `config.env` says. `rollout_trt.py --trt-n-action-steps N` really sets it.
- Under `--rtc` the MolmoAct2 expert stays in PyTorch, because RTC guidance backpropagates through it.
- LoRA (`use_peft`) checkpoints are not covered.

### `yaren/` — voice agent (two GPUs, two venvs, its own Ollama)

- **Split by GPU on purpose:** `robot_server.py` (lelab venv) holds MolmoAct2 + TensorRT on GPU0
  (15.2 GB). Whisper + Chatterbox + Qwen3.8-27B share GPU1 (24.7 GB of 32 GB). The voice side has its
  own venv `~/envs/yaren`, because Chatterbox 0.1.7 pins transformers 5.2 and lerobot pins 5.5.
- **Ollama ≥0.40 also enumerates GPUs through Vulkan, which ignores `CUDA_VISIBLE_DEVICES`.** Qwen
  then landed on GPU0, and the robot server died loading its engine
  (`could not deserialize …molmoact2_flowstep_….plan`: GPU0 was full). Always start it with
  `OLLAMA_VULKAN=0`, `GGML_VK_VISIBLE_DEVICES=<gpu>` and `CUDA_DEVICE_ORDER=PCI_BUS_ID` (GPU0 = PCI
  01:00.0, GPU1 = 08:00.0, the x4 link). `start.sh` does this.
- Yaren runs its own Ollama 0.40.9 from `~/opt` on port 11435. The system Ollama
  (`/usr/local/bin`, :11434) is 0.32.9 and refuses Qwen3.8 ("requires a newer version"). Leave the
  system one alone.
- `robot_server.py` reuses lerobot's `build_rollout_context` through `tools/trt/rollout_trt.install()`
  and the policy folder's `config.env`, so clamp, cameras, serial check and start pose match
  `3_run_autonomous.sh --trt`. Every policy folder's `lib.sh` `rollout_running()` counts
  `robot_server.py` as a rollout. `--dry` replays a recorded episode instead of connecting the arm.
  Test new agent behaviour with that first.
- Measured LLM failure modes and their guards (details in `yaren/README.md`): Qwen sometimes
  announces an action without calling `do_task` (1/20) and, far more often, says it deleted a memory
  without calling `forget` (14/20). Chatterbox appends hallucinated tails after the sentence.
- Long-term memory (`yaren/agent/memory.py`, Hermes Agent style): `people.md` and `notes.md`
  (1500 chars each, written by the LLM's `remember`/`forget` tools) and `runs.jsonl` (logged
  automatically from robot_server's `recent_results`). All live in `MEMORY_DIR` (default
  `run/memory/`, git-ignored) and are read once per conversation into the system prompt. Point an
  exported `MEMORY_DIR` at a scratch folder when testing, so tests don't write into the real memory.
- **Qwen3.8 copies prompt tags into replies** ("[ear] <the user's words>", "[tool] [tool use]", ~1 in 20
  first turns), and once one is in the history every later turn copies it and calls no tool - it looks
  like "she stopped answering". `brain.TAG` strips them from speech and history, and an empty/parroted
  reply is resampled once. Anything starting with "[" in a user-role message is a body event or timer,
  not the user (`user_turn`); treat it that way in new code.
- **robot_server keeps only the newest request**: a second request before the control thread picks up
  the first replaces it ("replaced by a newer request"). Body-language `express` never replaces a real
  request, and moves made in one LLM turn wait for the previous move (`until_still`). Keep both when
  adding motion tools, or a tool call made in the same breath as a nod silently fails.
- Return plain Python types from robot_server endpoints: a numpy bool in a plan summary made `/replay`
  answer 500. `test_motions.py` json-dumps the replay summary for that reason.
- The leader arm (mirror, teaching with it) comes from `LEADER_SERIAL`/`LEADER_ID` in the policy
  folder's `config.env`; calibration `calibration/teleoperators/so_leader/dash_so101.json`.
- In `--dry` the joints replay a recorded episode in a loop, so `at_home` only flickers and "stand up to
  chat" cannot be tested there; `posture_step` is unit-tested in `test_agent.py` instead.
- **Stall protection** (`motions.StallWatch`, wired in `robot_server.py`) uses limits from the demos'
  tracking error (1.5x the largest error a still joint held for 0.2 s; never fires on any of the 100
  demos). It is off in `--dry`: replayed joints never follow commands, so every move would look stalled.
  Test the reaction there with `POST /stall {"test": "<joint>"}`. Every command passes the
  `send_action` wrapper in `Body._wrap_send`; keep it that way, since the stall watch and the third
  hand's grip both depend on it.
- Dry tests with the LLM write to Yaren's **real** memory unless `MEMORY_DIR` points elsewhere: on
  2026-10-28 a dry teach test saved "Fouad taught Yaren a movement named selam" (deleted). Export
  `MEMORY_DIR` to a scratch folder before `./start.sh --dry` for tests.
- Test mode's spoken questions must not contain the answer words (`agent/trials.py`): an answer said
  while she still talks goes through the echo check, and "oldu" right after "Sence oldu mu?" is dropped
  as her echo. `test_agent.py` checks this.
- Warming Ollama with the session's system prompt does **not** speed up the first turn with
  qwen3.8 on Ollama 0.40.9: only a request that extends the previous one reuses the cache. Measured
  ~400 ms prefill for 1,666 tokens either way, ~155 ms for follow-up turns. Rerun `yaren/selftest.py` against `./start.sh --dry` after changing the
  persona, tools or voice code.

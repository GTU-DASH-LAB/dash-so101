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
- `ball_pickup_pi0/` — fine-tune **pi0** on SO-101 demos of "pick up the ball and
  place it in the white cup", recorded + trained + run entirely on this machine.
- `pencil_pickup_vla/` — earlier task: SmolVLA fine-tuned on Kaggle for pencil
  pickup.
- `daily_progress/` — dated diary (`day1.md`, `day2.md`, …), English + Türkçe.
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

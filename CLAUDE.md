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
  like `/dev/ttyACM0` (follower), `/dev/ttyACM1` (leader); confirm with
  `lerobot-find-port`.
- Cameras: `top` = `/dev/video0` (overhead), `wrist` = `/dev/video2`
  (gripper-mounted, rotated 180 + mirrored).
- This box's login session lacks the `dialout`/`video` groups until reboot, so
  run device scripts through: `sg dialout -c "sg video -c './1_record.sh'"`.
- `ball_pickup_pi0` trains locally on GPU (`--policy.device=cuda`); pi0 is ~3B
  params, so training is heavier/slower than SmolVLA.
- Publishing to the HF Hub is gated by `PUSH_DATASET_TO_HUB` /
  `PUSH_MODEL_TO_HUB` in `config.env`; local copies are always saved regardless.
  Hub pushes need `hf auth login`.

# cap_cup_sort_pi0

Fine-tune **pi0** on your own SO-101 demonstrations of *"pick up the pink bottle cap
and the purple cups (ignoring cups of other colors), and place them in the pink bowl"*
— recorded, trained (via **LoRA** this time), and run entirely on **this machine**.

## Pipeline

| Step | Script | What it does |
| ---- | ------ | ------------ |
| 0 | `find_camera.sh` | List cameras to confirm the `base_0_rgb` (/dev/video0) and `left_wrist_0_rgb` (/dev/video2) indices. |
| 1 | `1_record.sh` | Teleoperate (move leader, follower mirrors) and record **40** demos with both cameras. |
| 2 | `2_train_local.sh` | LoRA fine-tune `lerobot/pi0_base` locally on the GPU — 20000 steps. |
| 3 | `3_run_autonomous.sh` | Run your trained LoRA policy on the real arm. |

All settings live in **`config.env`** — the cameras block there is the single source of
truth shared by recording and running, so the policy always sees what it was trained on.

## Different from `ball_pickup_pi0` on purpose

- **Camera names match pi0 directly.** `config.env`'s `CAMERAS` keys are
  `base_0_rgb`/`left_wrist_0_rgb` (pi0_base's own declared feature names), not `top`/`wrist`.
  Same physical cameras/indices/rotation, just named so the recorded dataset's columns
  already match what pi0 expects. This means **neither `2_train_local.sh` nor
  `3_run_autonomous.sh` need `--rename_map` at all** — one less place for a
  feature-mismatch bug to hide.
- **LoRA instead of full fine-tuning.** `2_train_local.sh` passes `--peft.method_type=LORA`
  (plus `--peft.r`/`--peft.lora_alpha` from `config.env`), freezing ~99% of pi0's 4B
  params. Much lower memory footprint than the full fine-tune in `ball_pickup_pi0` — see
  the `BATCH_SIZE` comment in `config.env` for the measured number on this machine.
  Inference needs the matching `--policy.use_peft=true` flag too (already wired into
  `3_run_autonomous.sh`) — a LoRA checkpoint is an *adapter*, not a full model, and
  loading it without that flag fails.
- **40 demos, not 15.** `ball_pickup_pi0`'s first policy underfit/undertrained, plausibly
  in part from too little data for the task's difficulty. This task is also harder: it
  needs **color discrimination** (purple cups yes, other-colored cups no), which needs
  negative examples, not just positive ones — see recording tips below.
- **STEPS=20000** (vs. `ball_pickup_pi0`'s 10000) — explicitly requested, and reasonable
  given more demos and a harder task.

## Before you start

1. **Both arms connected.** Teleop needs the leader *and* follower. Ports drift after
   any USB replug — don't trust `config.env`'s `FOLLOWER_PORT`/`LEADER_PORT` blindly,
   re-check with:
   ```bash
   source ../.venv/bin/activate
   lerobot-find-port
   ```

2. **Run these with the right device groups** (only needed if you haven't rebooted since
   last using the cameras/arms):
   ```bash
   sg dialout -c "sg video -c './1_record.sh'"
   ```

3. **Logged in to the Hub** (needed for publishing — both `1_record.sh` and
   `2_train_local.sh` check this upfront and fail fast if not):
   ```bash
   hf auth login
   ```

## Publishing to Hugging Face

`PUSH_DATASET_TO_HUB` / `PUSH_MODEL_TO_HUB` in `config.env` are both `true` by default:

- `1_record.sh` pushes the finished dataset to `https://huggingface.co/datasets/$DATASET_REPO`.
- `2_train_local.sh` pushes the trained LoRA adapter to `https://huggingface.co/$MODEL_REPO`
  once training completes.

Everything is saved locally regardless — the Hub push is in addition to, not instead of,
the local copy.

## Recording tips (for good demos)

- **Include distractor cups of other colors in most episodes.** The model needs to see
  "here are 3 cups, only the purple one gets picked" repeatedly to learn the color
  discrimination — episodes with only purple cups and no distractors don't teach it what
  to *avoid*.
- Vary: cap-only episodes, purple-cups-only episodes, and mixed episodes (cap + purple
  cups + distractor cups together).
- Vary cup/cap position and the number/arrangement of distractor cups between episodes.
- Do the *whole* motion each time: approach → grasp → lift → move over bowl → release.
- Use the arrow keys (see `1_record.sh` header) to redo a bad take.

## Notes

- Dataset is stored locally at `~/.cache/huggingface/lerobot/<user>/so101_cap_cup_sort`
  and training reads it directly from there (the Hub copy is a publish, not the
  source of truth for training).
- If the trained policy still struggles with color discrimination after 20000 steps,
  the likely fix is more demos with more distractor-cup variety, not more steps —
  the model can't learn a distinction it was never shown negative examples of.

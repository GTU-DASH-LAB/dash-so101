# ball_pickup_pi0

Fine-tune **pi0** on your own SO-101 demonstrations of *"pick up the ball and place it in
the white cup"* — recorded, trained, and run entirely on **this machine** (no Kaggle).

## Pipeline

| Step | Script | What it does |
| ---- | ------ | ------------ |
| 0 | `find_camera.sh` | List cameras to confirm the `top` (/dev/video0) and `wrist` (/dev/video2) indices. |
| 1 | `1_record.sh` | Teleoperate (move leader, follower mirrors) and record **15** demos with both cameras. |
| 2 | `2_train_local.sh` | Fine-tune `lerobot/pi0_base` locally on the GPU. |
| 3 | `3_run_autonomous.sh` | Run your trained policy on the real arm. |

All settings live in **`config.env`** — the cameras block there is the single source of
truth shared by recording and running, so the policy always sees what it was trained on.

## Before you start

1. **Both arms connected.** Teleop needs the leader *and* follower. Right now only the
   follower (`/dev/ttyACM0`) is plugged in. Plug in the leader, then:
   ```bash
   source ../.venv/bin/activate
   lerobot-find-port         # identify the leader's /dev/ttyACM* port
   ```
   Set `LEADER_PORT` in `config.env` accordingly.

2. **Run these with the right device groups** (this machine's login session doesn't carry
   the `dialout`/`video` groups until a reboot), e.g.:
   ```bash
   sg dialout -c "sg video -c './1_record.sh'"
   ```

3. **Logged in to the Hub** (needed for publishing — both `1_record.sh` and
   `2_train_local.sh` check this upfront and fail fast if not):
   ```bash
   hf auth login
   ```

## Publishing to Hugging Face

Controlled by `PUSH_DATASET_TO_HUB` / `PUSH_MODEL_TO_HUB` in `config.env` (both `true`
by default):

- `1_record.sh` pushes the finished dataset to `https://huggingface.co/datasets/$DATASET_REPO`.
- `2_train_local.sh` pushes the trained model to `https://huggingface.co/$MODEL_REPO`
  once training completes.

Set either flag to `false` to keep that artifact purely local (e.g. while iterating
before you're ready to publish). Either way, everything is saved locally regardless —
the Hub push is in addition to, not instead of, the local copy.

## Recording tips (for good demos)

- Keep the scene like the real task: ball + white cup, minimal clutter in the `top` view.
- **Vary** ball position between episodes so the policy generalizes.
- Do the *whole* motion each time: approach → grasp → lift → move over cup → release.
- Use the arrow keys (see `1_record.sh` header) to redo a bad take.

## Notes

- `pi0` is a ~3B-param model. Training is heavier and slower than SmolVLA; the
  `BATCH_SIZE` / `STEPS` in `config.env` are conservative starting points — tune them.
- Dataset is stored locally at `~/.cache/huggingface/lerobot/<user>/so101_ball_pickup`
  and training reads it directly from there (the Hub copy is a publish, not the
  source of truth for training).
- 15 demos is a small dataset — treat the first trained policy as an experiment. If it
  underperforms, the usual fixes are: more/cleaner demos, or more training steps.

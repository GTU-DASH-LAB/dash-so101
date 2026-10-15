# pen_bowl_vla_jepa

**Task:** pick up the pen and put it in the bowl. **Policy:** VLA-JEPA, a vision-language-action
model with a built-in world model. **Camera:** only the gripper (wrist) camera. **Data:** the
same 100 demos as `../pen_bowl_act` (`fouad1233/so101_pen_in_bowl_wrist`).

All settings live in `config.env`. Run every script from this folder:

```bash
cd ~/robotics/dash-so101/pen_bowl_vla_jepa
```

## What the model is

| Part | Size | Role |
|---|---|---|
| Qwen3-VL-2B | ~2.4B | Reads the wrist image + the task sentence |
| DiT action head (flow matching) | small | Turns Qwen's features into 30 joint targets (1 s of motion) |
| V-JEPA2 encoder + predictor | 0.3B frozen + predictor | **Training only:** predicts the wrist camera's future frames (as V-JEPA latents). This "world model" loss shapes Qwen's features; it is not used on the robot |

Fine-tuned from `lerobot/VLA-JEPA-SimplerEnv`, the single-camera checkpoint (OXE Bridge/RT-1).

## Steps

| # | Command | What it does |
|---|---------|--------------|
| 2 | `./2_train_local.sh` | Fine-tune with a live loss/ETA dashboard (http://127.0.0.1:8789) |
| 2b | `./2b_publish_model.sh` | Push a checkpoint + model card to the HF Hub |
| 3 | `./3_run_autonomous.sh` | Run it on the arm |

`./smoke_test.sh <batch> [gpus] [steps]` runs a few real steps and prints memory and speed; use it
before changing `BATCH_SIZE`/`NUM_GPUS`.

## Adapting the base model to this robot (`prepare_init.py`, `train_vla_jepa.py`)

- **6 joints, joint state as input:** the base model has a 7-D action and no state input. The three
  layers whose shape depends on that start fresh (`reinit_modules`); everything else keeps its
  pretrained weights. The state is normalized with this dataset's statistics.
- **Continuous gripper:** the base model snaps its gripper to open/closed; that is switched off.
- **30 Hz:** chunk of 30 actions (1 s) instead of 7, and the world model's 8 frames are taken every
  4th frame so they also span ~1 s.
- **Two training-time fixes over lerobot 0.6.0**, applied by `train_vla_jepa.py` (inference needs
  neither, so the published model runs on stock lerobot):
  1. the action head was given the joint state from the *end* of the 8-frame video window,
     i.e. the future; it now gets the current state, as on the robot;
  2. images are resized to 224x224 in training, as the policy already does at inference.

## Running the policy (step 3)

```bash
./3_run_autonomous.sh --list          # checkpoints, epoch, validation loss
./3_run_autonomous.sh                 # newest checkpoint, until Ctrl+C
./3_run_autonomous.sh best            # lowest validation loss
./3_run_autonomous.sh hub             # the published model
```

Put the follower in your demo start pose first. Each joint moves at most 5° per tick
(`MAX_RELATIVE_TARGET`). The policy predicts 30 actions and executes `N_ACTION_STEPS` (15) of
them before predicting again; lower it to react faster.

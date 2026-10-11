# pen_bowl_act

**Task:** pick up the pen and put it in the bowl. **Policy:** ACT. **Camera:** only the
gripper (wrist) camera. **Data:** 100 teleoperated demos, recorded and trained on this machine.

All settings live in `config.env`. Run every script from this folder:

```bash
cd ~/robotics/dash-so101/pen_bowl_act
```

## Steps

| # | Command | What it does |
|---|---------|--------------|
| 0 | `./find_camera.sh` | Lists cameras, saves a snapshot from each, marks the configured wrist camera |
| 0 | `./0_teleoperate.sh` | **Practice** teleoperation. Nothing is saved. Ctrl+C to finish |
| 1 | `./1_record.sh` | Record demos (continues where you left off, any time) |
| 1 | `./manage_data.sh` | Progress, review videos, delete bad demos/sessions, undo |
| 1b | `./1b_publish_dataset.sh` | Merge all sessions, verify every demo, push to the HF Hub |
| 2 | `./2_train_local.sh` | Train ACT on both GPUs with a live loss/ETA dashboard |
| 3 | `./3_run_autonomous.sh` | Run the trained policy on the arm |

Every script checks the hardware first, read-only. It finds the leader and follower
by **USB serial** (so a replug that swaps `/dev/ttyACM0/1` is harmless), checks that the
motors answer and the power is on, compares the calibration stored in the motors
with the calibration files, and opens the camera.

## Recording (step 1)

```bash
./1_record.sh
```

Keys (keep the **terminal** focused: arrow keys in the rerun window move its timeline):

| Key | During a demo | During the reset pause |
|-----|---------------|------------------------|
| `→` | demo done, save it | start the next demo now |
| `←` | redo this demo | throw away the demo you just did |
| `Esc` | stop (this demo is thrown away) | save it and stop |

- Demos are recorded in **sessions of 10**. Each session is its own small dataset, so an
  accident can only ever affect one session. Between sessions you get a safe point to
  take a break. Running `./1_record.sh` later always continues where you stopped.
- **If a cable comes out / the arm loses power:** finished demos are safe. The demo in
  progress is thrown away. A demo you had *just finished* (during the reset pause) is
  saved. Plug it back in, press Enter, and recording continues in the same session.
- Demos shorter than 3 s (an accidental double key press) are discarded automatically.
- When a session ends or you press Esc, you get a pause to **lower the arm to rest** before
  the motors relax.

### Reviewing and deleting

```bash
./manage_data.sh                    # progress per session, total/100
./manage_data.sh review 3           # every demo of session 3 as its own video
./manage_data.sh review 3 7         # just demo 7 of session 3
./manage_data.sh delete-episodes 3 7 9
./manage_data.sh delete-session 3   # whole session
./manage_data.sh trash              # deleted things (nothing is gone until empty-trash)
./manage_data.sh restore <name>     # undo
```

Demo numbers inside a session shift down after a delete, so list them again
(`./manage_data.sh episodes 3`) before deleting more.

### Recording tips for a wrist-camera-only policy

The policy sees **only** the wrist camera plus the joint angles. So:

1. **Start every demo from the same start pose**, high enough that the wrist camera
   sees the whole area where the pen can lie.
2. Move so that the **bowl comes into the camera view** before you release the pen. If
   the camera never sees the bowl, the policy cannot learn where it is.
3. **Vary** the pen position and angle, and the bowl position, across the whole
   workspace. Do not record 100 copies of one layout.
4. Be **smooth and decisive**. Hesitations and corrections get learned too. Press `←`
   on any demo you are unhappy with.
5. Keep the lighting and the background the same as they will be when the policy runs.

## Training (step 2)

Measured on this machine (ACT, 640x480 wrist images, bf16):

| Setup | samples/s |
|---|---|
| 1 GPU | ~468 |
| 2 GPUs, lerobot's default DDP | 636 |
| **2 GPUs, `train_act.py` (bf16 gradient sync)** | **798 (1.7x)** |

```bash
./2_train_local.sh            # start; opens the live dashboard (http://127.0.0.1:8787)
./2_train_local.sh status     # the same numbers in the terminal
./2_train_local.sh stop       # stop; the newest checkpoint is kept
./2_train_local.sh            # resume a stopped or crashed run
./2_train_local.sh --fresh    # start over after changing config.env
```

- **Plan for this dataset (100 demos, 65,550 frames):** 90 training / 10 validation demos.
  The batch is 64 per GPU × 2 = 128, which gives 455 steps per epoch. **60 epochs = 27,300
  steps, about 75 minutes.** Validation runs every epoch, and a checkpoint is saved every 5.
- The **dashboard** shows train loss and validation loss per epoch, progress, remaining time
  and the finish time, GPU load, and every checkpoint with its validation loss.
- Training runs **detached**, so closing the terminal is fine. If it crashes, the watchdog
  resumes from the last checkpoint by itself.
- Validation loss = action error on 10 demos the model never sees. If it rises while train
  loss keeps falling, the model is over-fitting. That is a hint, not a verdict: in behaviour
  cloning, lowest validation loss is only loosely tied to success on the robot. Try `best`
  and `last` on the arm.
- The run settings come from measurements on this machine; see the comments in
  `config.env`. Recorded in h264, decoded with torchcodec, no CPU augmentation, bf16,
  lr 1e-4. `train_act.py` sends gradients between the GPUs in bf16, which took 2-GPU
  throughput from 636 to 798 samples/s.

## Running the policy (step 3)

```bash
./3_run_autonomous.sh --list          # checkpoints, epoch, validation loss
./3_run_autonomous.sh                 # newest checkpoint, until Ctrl+C
./3_run_autonomous.sh best            # lowest validation loss
./3_run_autonomous.sh --trials 10     # 10 timed attempts; the arm returns home in between
```

Put the follower in your demo start pose first. Each joint moves at most 5° per tick
(`MAX_RELATIVE_TARGET`). Actions use ACT's temporal ensembling (`ACTION_MODE=ensemble`,
smooth, re-plans every tick). `ACTION_MODE=chunk` is the alternative if motion is too
hesitant. Ctrl+C returns the arm to where it started before the motors relax.

## Tests

`tests/run_recording_tests.sh` drives the recorder with fake arms and a synthetic camera
(nothing moves) and checks every failure path. Each scenario re-opens the dataset and
decodes every frame: unplug mid-demo, camera lost during the reset pause, resume after
both, `←` redo, double press, `Esc` in both phases, Ctrl+C, and `kill -9`.

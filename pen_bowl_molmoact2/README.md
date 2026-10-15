# pen_bowl_molmoact2

**Task:** pick up the pen and put it in the bowl. **Policy:** MolmoAct2 (Ai2, 5.4B), fine-tuned from
`allenai/MolmoAct2-SO100_101`, which was already trained on community SO-100/101 data.
**Camera:** only the wrist camera. **Data:** the same 100 demos as `../pen_bowl_act`.

```bash
cd ~/robotics/dash-so101/pen_bowl_molmoact2
./2_train_local.sh                 # train on GPU_IDS (config.env: GPU 1); dashboard http://127.0.0.1:8790
./2_train_local.sh status | stop
./2b_publish_model.sh [last|best|<step>] [--dry-run]
./3_run_autonomous.sh [last|best|<step>|hub] [--trials N]
./smoke_test.sh <batch> 1 [steps]  # memory/speed check before changing BATCH_SIZE
```

## Recipe (docs/source/molmoact2.mdx, "datasets under 200 demos")

LoRA on the vision-language model, the flow-matching action expert fully trainable, bf16, gradient
checkpointing, image augmentation on, continuous actions, 30-action chunks (1 s). One RTX 5090.
The docs measure 26.8 GiB at batch 16 with LoRA, and 12.1 GiB for inference.

## The joint-convention fix, done properly

The checkpoint learned SO-100/101 joints in the **pre-LeRobot-0.5 calibration**:
`model = sign * arm + offset` with signs `[1,-1,1,1,1,1]` and offsets `[0,90,90,0,0,0]`
(shoulder_lift flipped, shoulder_lift and elbow_flex shifted 90°). lerobot's `joint_signs` /
`joint_offsets` flags convert only the *state*, before normalizing with this dataset's
arm-frame statistics, and never the action targets. For fine-tuning that would leave two joints
normalized ~90° off and state/action in different conventions. Instead:

1. `make_model_frame_dataset.py` writes a copy of the dataset in the model's convention: state,
   action, and their statistics transformed exactly; videos shared through a symlink. Verified on
   sample frames, and the converted ranges match the pretraining data (shoulder_lift 59..194
   here vs 85..184 in the SO-101 community data).
2. Training runs on that copy with the flags **off**.
3. `deploy_config.py` makes a deployable copy of a checkpoint with the flags **on** (state into the
   model frame, actions back to the arm frame). `3_run_autonomous.sh` and `2b_publish_model.sh`
   use it, so the published model runs directly on a currently calibrated SO-101.

## Note

The policy always builds itself from the base checkpoint (`checkpoint_path`) before loading the
fine-tuned weights. Locally that is `~/models/molmoact2/MolmoAct2-SO100_101`; the published
config names `allenai/MolmoAct2-SO100_101` (21.8 GB, fetched once on another machine).

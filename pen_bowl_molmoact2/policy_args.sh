# shellcheck shell=bash
# MolmoAct2 policy + dataset arguments, shared by watchdog_train.sh and smoke_test.sh
# (sourced after lib.sh). Follows docs/source/molmoact2.mdx's SO-100/101 fine-tuning
# recipe; for datasets < 200 demos: LoRA on the VLM, action expert fully trainable.
# joint_signs/joint_offsets stay OFF here: training data is already in the model frame
# (make_model_frame_dataset.py); deploy_config.py switches them on for the robot.
POLICY_ARGS=(
  --policy.type=molmoact2 --policy.checkpoint_path="$BASE_CHECKPOINT"
  --policy.device=cuda --policy.push_to_hub=false
  --policy.action_mode=continuous --policy.inference_action_mode=continuous
  --policy.chunk_size="$CHUNK_SIZE" --policy.n_action_steps="$N_ACTION_STEPS"
  --policy.model_dtype=bfloat16 --policy.gradient_checkpointing=true
  --policy.enable_lora_vlm=true
  --policy.image_keys='["observation.images.wrist"]'
  --policy.norm_tag="$NORM_TAG" --policy.setup_type="$SETUP_TYPE" --policy.control_mode="$CONTROL_MODE"
  --dataset.repo_id="$MODEL_FRAME_DATASET_REPO" --dataset.root="$MODEL_FRAME_DATASET_ROOT"
  --dataset.video_backend=torchcodec --dataset.image_transforms.enable="$AUGMENT"
)

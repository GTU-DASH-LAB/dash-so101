# pen_bowl_flux3

**Task:** pick up the pen and put it in the bowl. **Policy:** FLUX 3 Action (Black Forest Labs, 7B
"world action model": it denoises the next 32 actions together with the next video frames), LoRA
fine-tune of `black-forest-labs/flux-3-action-so101`. **Camera:** only the wrist camera.
**Trained on** AWS (one H100 80 GB), **run on** this PC's RTX 5090.

## Steps

| # | Command | What it does |
|---|---------|--------------|
| 0 | `./0_setup_env.sh` | `~/envs/lerobot-main`: lerobot from GitHub main (FLUX 3 is not in 0.6.0) + NATTEN. Done on this PC; the same script runs on the AWS machine |
| 2 | `./aws_train.sh start` | Launch the H100 (asks first), set it up, measure 20 steps, start training on the machine |
| 2 | `./aws_train.sh watch` | Pull checkpoints every 15 min until training ends |
| 2 | `./aws_train.sh finish` | Final pull, then terminate the machine |
| 2 | `./aws_train.sh resume` | After a spot interruption: new machine, push the last checkpoint, continue |

Needs: the H100 quota (`../aws/check_quota_requests.sh`), `aws login`, and a **read-only** Hugging Face
token in `~/.config/hf-aws/token` (the weights are gated; it is the only credential copied to AWS).
The machine deletes itself after `MAX_HOURS` (`../aws/gpu_box.env`) whatever happens.

## Adapting the SO-101 checkpoint (`prepare_init.py`)

- **Wrist only:** the checkpoint composes scene|wrist side by side (each 256x256). Layout `single`
  on a 256x256 canvas gives the wrist image exactly the resolution it had in pretraining.
- **Joint-state normalization:** FLUX 3 keeps its own q01/q99 statistics (lerobot's dataset-stat
  override does not reach them). They come from SO-101 data in the pre-0.5 calibration
  (shoulder_lift 85..184 deg there, -104..61 here), so the state quantiles are replaced by this
  dataset's. Action quantiles (per-step command deltas) already cover our data and are kept.

## Verified on this PC

`0_setup_env.sh`: torch 2.11+cu128 on the RTX 5090 (sm_120), NATTEN 0.21.6 `na2d` runs on the GPU,
`lerobot.policies.flux3` imports. Inference memory: 13.9 GB model + 8.9 GB text encoder + 2.5 GB
video VAE in bf16 (the text encoder can be moved to the CPU if needed).

# WM-ACT — a lightweight "more than VLA" policy for the SO-101

## Honest framing first

This is **not** a never-before-seen architecture, and nobody should claim to have invented
one in an afternoon. WM-ACT is a deliberate composition of the two strongest ideas in the
post-VLA literature, sized for exactly this setup (1 camera, 50 demos, SO-101, GX10):

- **Action chunking** (ACT, Zhao et al. 2023) — still the strongest imitation learner for
  single-task real-robot manipulation with ~50 demos, at a fraction of VLA size.
- **Latent world models** (JEPA / V-JEPA 2-AC, TD-MPC, Dreamer) — the ingredient the
  "robots need more than VLA" papers keep pointing at: a model that learns
  *action-conditioned dynamics*, not just imitation-relevant appearance.

The critique of VLAs the PDF makes — no world model, no physical grounding, pure
imitation, heavyweight language tower that a single-task policy never uses — is addressed
here in the cheapest form that actually works.

## Architecture

```
                     ┌───────────────────────────── training only ─────────────┐
                     │                                                          │
 camera (front) ──► ResNet18 ──► feature map ──► ACT transformer ──► action     │
                     │   ▲          │              encoder/decoder    chunk     │
 joint state ────────┼───┘          │ mean-pool         ▲            (100 st.)  │
                     │              ▼                   │                │      │
                     │        current latent ──► WM predictor MLP ◄── actions   │
                     │                                  │ predicts              │
 camera at t+15 ──► EMA copy of ResNet18 ──► future latent  ◄── cosine loss ────┘
                    (stop-gradient)
```

- The **world-model head** gets the pooled current visual latent + the action chunk and
  must predict the pooled visual latent **15 frames (0.5s) in the future**.
- The target comes from an **EMA copy** of the backbone (stop-gradient) — the standard
  BYOL/JEPA recipe that prevents the latents from collapsing to a constant.
- Total loss: `L1(actions) + kl_weight·KLD + wm_loss_weight·(1 − cos(pred, target))`.
- **At inference the WM head is never called.** Runtime is byte-for-byte plain ACT.

Why this answers "more than VLA", lightweight edition:

| | SmolVLA (current baseline) | WM-ACT |
|---|---|---|
| Trainable params | ~450M (+2GB base VLM download) | **52.7M**, no base model |
| World model / dynamics grounding | none | JEPA-style latent dynamics |
| Inference on the Mac (MPS) | too slow for 30Hz → needs RTC | **sync loop at 30Hz** |
| Language tower | yes (unused for a single fixed task) | none — that's the point |
| Camera rename gotcha (Day 3) | needs `--rename_map` | trained natively on `front` |

## Files

- [configuration_wmact.py](configuration_wmact.py) — `WMACTConfig`, subclasses LeRobot's `ACTConfig`
- [modeling_wmact.py](modeling_wmact.py) — `WMACTPolicy`, subclasses `ACTPolicy` (~60 lines of new logic)
- [smoke_test.py](smoke_test.py) — runnable check: forward/backward, EMA update, inference shapes
- [train_gx10.sh](train_gx10.sh) / [run_robot.sh](run_robot.sh)

It's a **LeRobot plugin**: `--policy.discover_packages_path=wm_act` imports this package,
which registers policy type `wm_act`. Zero changes inside the lerobot submodule; all of
`lerobot-train` / `lerobot-rollout` (checkpointing, normalization, hub push, cameras) is reused.

## Train (on the GX10)

```bash
git clone --recurse-submodules <this repo> && cd fouad_so101
python -m venv .venv && source .venv/bin/activate
pip install -e ./lerobot            # on GB10 use an aarch64 CUDA PyTorch build (NGC container works)
huggingface-cli login
./wm_act/train_gx10.sh
```

60k steps at batch 64 on the 43k-frame dataset ≈ 1–2h on the GX10. Checkpoints land in
`outputs/train/wmact_pencil_pickup` and push to `fouad1233/wmact_pencil_pickup`.

Sanity check before burning GPU time (runs on the Mac in ~30s):

```bash
PYTHONPATH=. .venv/bin/python wm_act/smoke_test.py
```

## Run on the robot (on the Mac)

```bash
./wm_act/run_robot.sh                       # pulls fouad1233/wmact_pencil_pickup from the Hub
./wm_act/run_robot.sh outputs/train/wmact_pencil_pickup/checkpoints/last/pretrained_model  # or local
```

## Evaluate — the experiment that actually matters

Run the same protocol for both policies: 10 trials each, same 10 pen start positions
(mark them on the desk), count successes.

1. `cd pencil_pickup_vla && ./3_run_autonomous.sh` → SmolVLA success rate
2. `./wm_act/run_robot.sh` → WM-ACT success rate
3. The clean ablation: retrain with `--policy.wm_loss_weight=0` (= plain ACT) to measure
   what the world-model loss itself buys.

## Knobs

- `--policy.wm_future_offset=15` — prediction horizon in frames (0.5s @ 30fps)
- `--policy.wm_loss_weight=1.0` — `0` disables the world model → plain ACT ablation
- `--policy.wm_ema_momentum=0.996`
- `--policy.chunk_size=100 --policy.n_action_steps=100` — inherited ACT knobs

## Upgrade paths (deliberately not built — add when the eval says so)

- **Planning**: use the WM head at inference to score candidate action chunks (sample K
  chunks from the VAE, pick the one whose predicted future latent is closest to a goal
  latent). That's TD-MPC-style MPC — only worth it if plain WM-ACT plateaus.
- **Better eyes**: swap ResNet18 for frozen DINOv2-S if generalization to new pen
  positions/lighting is the bottleneck.
- **Memory**: `n_obs_steps > 1` needs upstream ACT changes; only for occlusion-heavy tasks.

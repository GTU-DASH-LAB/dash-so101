# tools/trt — TensorRT inference for the trained policies

| File | What |
|---|---|
| `trt_accel.py` | `accelerate(policy)`: replaces the heavy network of a loaded policy with a TensorRT engine, in place |
| `rollout_trt.py` | `lerobot-rollout` with `trt_accel` applied (same CLI). Used by `3_run_autonomous.sh --trt` |
| `bench_inference.py` | speed + timing stability of a policy on held-out dataset frames, no arm; `--save`/`--compare` check that actions match |

```bash
cd pen_bowl_act        && ./3_run_autonomous.sh --trt [--trials N] [checkpoint]
cd pen_bowl_molmoact2  && ./3_run_autonomous.sh --trt [--trials N] [checkpoint]
```

The first run of a checkpoint builds the engine (ACT 28 s, MolmoAct2 78 s on the 5090) after the
policy loads and **before the robot connects**; later runs load it from `~/.cache/dash_so101_trt`
in ~1 s. Robot code, `max_relative_target`, strategies and keys are lerobot's own and unchanged.

## Measured (RTX 5090, episode 95 of so101_pen_in_bowl_wrist, 300 ticks paced at 30 Hz)

Each tick = `postprocessor(select_action(preprocessor(obs)))`, exactly what `lerobot-rollout`'s
sync engine runs. "Diff" = actions vs the PyTorch run on the same frames and seed.

| | tick mean | p99 | std | diff vs PyTorch (max / mean) |
|---|---|---|---|---|
| ACT PyTorch | 4.12 ms | 4.82 | 0.29 | — |
| ACT TRT fp32 | 2.63 ms | 3.05 | 0.13 | 0.046° / 0.007° |
| **ACT TRT fp16** (default) | **1.84 ms** | 2.22 | 0.13 | 0.110° / 0.014° |

MolmoAct2 predicts a 30-action chunk on 1 tick of 30, and pops the queue on the other 29:

| | chunk tick mean (std) | other ticks p50 | diff vs PyTorch (max / mean) |
|---|---|---|---|
| MolmoAct2 PyTorch | 127.2 ms (4.75) | 11.5 ms | — |
| + TRT action expert, O(1) eval | 88.3 ms (0.77) | 6.1 ms | 0.335° / 0.039° |
| **+ queue ticks skip the preprocessor** (`rollout_trt.py`) | **87.8 ms (0.60)** | **0.25 ms** | 0.000° vs the line above |

The chunk tick is still longer than one 33 ms control period, so the arm still pauses at each chunk
boundary — ~88 ms instead of ~127 ms. The rest of a chunk: ViT ~14 ms, LLM prefill ~33 ms, expert
~30 ms (was ~72 ms). `--rtc` hides the pause, but RTC's guidance differentiates through the expert,
so under `--rtc` the expert stays in PyTorch.

**MolmoAct2 ignores `--policy.n_action_steps`**: its `__init__` replaces it with the base
checkpoint's metadata (30). Every non-RTC run executed all 30 actions per chunk, whatever
`N_ACTION_STEPS` said. `rollout_trt.py` warns and keeps 30; `--trt-n-action-steps N` really
executes N.

## Why ONNX → TensorRT and not torch_tensorrt

- torch_tensorrt 2.11's `scaled_dot_product_attention` converter gives wrong values whenever
  `attn_mask` is passed (error ~3 on unit-scale outputs, even an all-zero mask, every dtype). The
  MolmoAct2 expert's cross-attention is therefore written out by hand (fp32 scores + softmax).
- Its graphs of the MolmoAct2 expert hit TensorRT 10.15 Myelin internal errors on sm_120 in every
  variant tried; the fused qkv projection also had to be split into q/k/v.
- `torch.onnx.export(dynamo=True)` + `trt.OnnxParser` builds the same graph cleanly and accurately.

## Not covered

- LoRA (`use_peft`) checkpoints, e.g. `tshirt_fold_pi05`: plain PyTorch, with a warning.
- Every other policy type (VLA-JEPA, ...): `accelerate()` raises NotImplementedError; `rollout_trt.py`
  then runs plain PyTorch.

Install (lelab venv): TensorRT 10.15.1.29 (+ torch_tensorrt 2.11.0+cu128, which pins TRT
≥10.15.1,<10.16), `onnx`, `onnxscript`, installed `--no-deps`. The libs wheel is on
pypi.nvidia.com only (pypi.org has an sdist stub); fetch it with `aria2c`.

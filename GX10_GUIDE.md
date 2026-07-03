# GX10 Guide — train, test, and benchmark the WM-ACT family

Everything to run on the ASUS GX10 (GB10 Grace Blackwell, 128GB, aarch64 Linux),
plus the Mac-side testing steps. Follow top to bottom the first time.

## 0. One-time setup (GX10)

```bash
git clone --recurse-submodules <this-repo-url> && cd fouad_so101
python -m venv .venv && source .venv/bin/activate
# PyTorch for GB10: use an aarch64 CUDA build (the NGC PyTorch container also works)
pip install -e "./lerobot[libero]"        # lerobot + LIBERO sim (Linux-only extra)
huggingface-cli login                     # needed to pull datasets / push models
export MUJOCO_GL=egl                      # headless rendering for LIBERO
export PYTHONPATH="$PWD:$PYTHONPATH"      # makes the wm_act/point_act plugins importable
```

Optional (for the language planner + CoT diagnosis, used by the Mac at runtime —
running ollama on the GX10 and pointing the Mac at it keeps the laptop light):

```bash
ollama pull qwen2.5vl:7b                  # then on the Mac:
# export SO_BRAIN_LLM_ENDPOINT=http://<gx10-ip>:11434/v1
```

Sanity checks (fast, no robot):

```bash
python wm_act/smoke_test.py
python point_act/smoke_test.py
python so_brain/planner.py --selftest
python so_brain/memory.py
```

## 1. Train on your own dataset

### 1a. WM-ACT (single task, no language — your SmolVLA replacement)

```bash
./wm_act/train_gx10.sh
```

Trains on `fouad1233/so101_pencil_pickup`, 60k steps, batch 64 (~1–2h), pushes to
`fouad1233/wmact_pencil_pickup`. For a new dataset, edit `DATASET_REPO` in
`pencil_pickup_vla/config.env` (or copy the script and change the two repo ids).

### 1b. Point-ACT (the zero-shot stack's motor policy)

```bash
./point_act/train_gx10.sh
```

Step 1 auto-labels every episode with the grounder (object + destination points from
frame 0 → `outputs/point_labels_pencil.json`); step 2 trains with markers burned into
the images. For a **new multi-object dataset** (the one that unlocks real zero-shot —
see `so_brain/README.md`, "data recipe"): record with a descriptive task string per
object session, then relabel without fixed phrases so each episode's own task string
is parsed:

```bash
python so_brain/relabel.py --repo-id fouad1233/so101_pickplace_multi --out outputs/point_labels_multi.json
# then train with --policy.point_labels_path=$PWD/outputs/point_labels_multi.json
```

`SO_BRAIN_GROUNDER=nvidia` before relabel switches to LocateAnything-3B (better
phrases understanding; first use downloads ~6GB; non-commercial license).

### 1c. Verify the physics (world model) actually learned

```bash
python wm_act/selfcheck.py \
  --policy-path outputs/train/pointact_pickplace/checkpoints/last/pretrained_model \
  --repo-id fouad1233/so101_pencil_pickup --episode 0
```

Read the gap: **true-action surprise clearly below shuffled-action surprise** means the
model predicts consequences of actions, not just visual continuity. The printed p95 is
the `SurpriseMonitor` threshold. If the gap is ~0, raise `--policy.wm_loss_weight` or
train longer, and re-check.

### 1d. Making the model bigger (when generalization needs it)

Bigger helps **only when the dataset is diverse enough to feed it** — scale the data
first (multi-object recordings), then the model. All knobs are CLI flags; no code changes:

| Preset | Flags | Params | Use when |
|---|---|---|---|
| base (default) | — | ~53M | single task, ≤100 episodes |
| large | `--policy.vision_backbone=resnet34 --policy.pretrained_backbone_weights=ResNet34_Weights.IMAGENET1K_V1 --policy.dim_model=768 --policy.n_encoder_layers=6 --policy.dim_feedforward=4096 --policy.wm_hidden_dim=768` | ~95M | multi-object dataset, 150+ episodes |
| XL vision | large flags but `resnet50` / `ResNet50_Weights.IMAGENET1K_V2` | ~120M | visually varied scenes/lighting |

Append the flags to any `lerobot-train` call (or to the train scripts). Rule of thumb
on the GX10: batch 64 fits all presets easily; larger presets want ~2× the steps.

## 2. Test on the robot (Mac)

```bash
./wm_act/run_robot.sh                                   # single-task WM-ACT
./so_brain/run.sh "put the pen on the black mouse pad"  # full zero-shot stack
./so_brain/run.sh "..." --dry-run                       # perception only, no robot
```

The so_brain loop retries up to `--attempts 3`: verify (detector) → CoT diagnosis →
corrective retry → episodic memory (`so_brain/episodic_memory.jsonl`).

**Success-rate protocol** (fill this into the diary): mark 10 start positions on the
desk; for each policy run 10 trials from the same positions.

| Policy | Trained object | Novel object 1 | Novel object 2 | Novel destination |
|---|---|---|---|---|
| SmolVLA (day 3 baseline) | /10 | — | — | — |
| WM-ACT | /10 | — | — | — |
| Point-ACT + brain | /10 | /10 | /10 | /10 |

Only Point-ACT gets the novel columns — that's the zero-shot claim under test.

## 3. LIBERO benchmark (GX10, simulation — no robot needed)

LIBERO is LeRobot's standard sim benchmark (130 tasks in 5 suites). Reference: π0.5
scores 97.5% average across the 4 standard suites — a 3B-param VLA; don't expect a
53M policy to match it. What this benchmark is actually for: **comparing your own
variants under identical conditions** (act vs wm_act vs wm_act_large), i.e. measuring
what the world-model loss and scale buy.

These policies are language-blind, so the protocol is **per task**: train on one
task's episodes, evaluate on that task (`--env.task_ids`). One command per run:

```bash
export MUJOCO_GL=egl
python benchmark/libero_bench.py --suite libero_object --task-id 0
# more variants / longer:
python benchmark/libero_bench.py --suite libero_object --task-id 0 \
    --variants act,wm_act,wm_act_large --steps 50000
# see the exact commands without running:
python benchmark/libero_bench.py --suite libero_spatial --task-id 3 --dry-run --episodes 0,1
```

What it does per variant: resolves the task's language string via the LIBERO package →
selects matching episodes from `HuggingFaceVLA/libero` → `lerobot-train` → `lerobot-eval`
(10 episodes, 256×256 to match the dataset) → appends to `benchmark/results.json` and
prints a markdown table you can paste into the diary.

Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` (long-horizon),
`libero_90`. A decent self-benchmark: task 0–4 of `libero_object` + `libero_spatial`,
3 variants — 30 runs, each ~1h train + ~10min eval on the GX10; start with one run
end-to-end to shake out the environment.

Expected reading of results: `wm_act > act` on contact-rich tasks is the signature of
the world-model loss working; `wm_act_large > wm_act` only where episodes are plentiful
(LIBERO has 50/task — scale may not pay; that's a finding, not a failure).

## 4. pi-WM: the "beat pi0.5" experiment (GX10)

The strongest model in this repo: frozen `pi05_libero_finetuned` (3B) + a 12.5M
world-model scorer doing best-of-N action selection. Full instructions, honest
expectations, and the results table live in [pi_wm/README.md](pi_wm/README.md).
Short version:

```bash
python pi_wm/make_checkpoint.py --out pi_wm_checkpoint            # ~7GB download
python pi_wm/train_scorer.py --pi05-path pi_wm_checkpoint --out outputs/scorer_libero.pt
# then the two lerobot-eval commands in pi_wm/README.md: n_samples=1 (baseline) vs 8
```

Run the baseline first — reproducing pi0.5's published 97.5 validates the whole
setup before the experiment costs you a day of eval time.

For **unseen / long-horizon tasks** (`libero_10`, where pi0.5 degrades), run the
hierarchical harness — CoT decomposition + grounded verification + episodic memory
around the frozen policy, with a built-in 4-way ablation. Commands and expectations:
[pi_wm/README.md](pi_wm/README.md), "long-horizon experiment" section.

## 5. Troubleshooting

- `PluginLoadError ... wm_act` → `PYTHONPATH` must include the repo root.
- MuJoCo render errors → `export MUJOCO_GL=egl` (headless) before train/eval.
- LIBERO eval loads but success is ~0 for all variants → check `--env.control_mode`
  (dataset uses `relative`, the default) and that eval resolution stays 256×256.
- `libero` import fails → the extra is Linux-only: `pip install -e "./lerobot[libero]"`.
- Anything downloading on the Mac by accident → it belongs on the GX10; models
  lazy-load on first use, so just don't call them locally.

# pi-WM — beating pi0.5 on LIBERO with a world-model judge

## The claim, stated honestly

Nobody beats pi0 "in the world" from a single GX10 — pi0.5 was trained on ~10k hours
of multi-robot data. What *is* achievable, and what this package is built to do:
**beat pi0.5's published LIBERO score (97.5% average) under the identical evaluation
protocol**, by attacking its residual ~2.5% failures with test-time compute.

The mechanism has published precedent (V-GPS value-guided steering, RoboMonkey
test-time scaling for VLAs, TD-MPC-style model-based reranking):

```
observation ──► pi0.5 (frozen) ──► N action chunks sampled from N noise draws
                                        │
              world-model scorer (12.5M): predicts each chunk's visual future
              and how far along the task that future is (progress)
                                        │
                        execute the chunk with max predicted progress
```

pi0.5's flow model is stochastic — different noise gives genuinely different plans.
Most samples are good; the failures come from occasionally committing to a bad one.
The scorer never needs to *act* — it only needs to *rank* the base model's own
proposals, a far easier problem, learnable from LIBERO's demos on one GX10.

Why the scorer can rank: every demo episode succeeds, so frame t sits at task
progress t/(T−1). The scorer learns (JEPA-style, EMA target) to imagine the latent
future of an action chunk and reads off its progress. Chunks that knock the bowl
over don't advance progress in imagination.

## Expected outcome — decide by measurement

Realistic range: **+0.5 to +2 points** over the 97.5% baseline if the scorer ranks
well; **0** if it's miscalibrated (then more scorer training / bigger N, or accept
the negative result — it's still a benchmark finding). The A/B is one flag:
`--policy.n_samples=1` *is* pi0.5, byte for byte.

## Run it (GX10, all steps)

```bash
export MUJOCO_GL=egl PYTHONPATH="$PWD:$PYTHONPATH"

# 1. local checkpoint with type rewritten to pi_wm (~7GB download)
python pi_wm/make_checkpoint.py --out pi_wm_checkpoint

# 2. train the scorer on LIBERO demos (~1-2h)
python pi_wm/train_scorer.py --pi05-path pi_wm_checkpoint --out outputs/scorer_libero.pt

# 3. BASELINE: plain pi0.5 (n_samples=1), full 4-suite protocol, 10 eps/task
lerobot-eval \
  --policy.discover_packages_path=pi_wm \
  --policy.path=pi_wm_checkpoint \
  --policy.n_samples=1 \
  --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 --eval.n_episodes=10 --env.max_parallel_tasks=1 \
  --output_dir=benchmark/eval/pi05_baseline

# 4. pi-WM: best-of-8 with the scorer
lerobot-eval \
  --policy.discover_packages_path=pi_wm \
  --policy.path=pi_wm_checkpoint \
  --policy.n_samples=8 \
  --policy.scorer_path=$PWD/outputs/scorer_libero.pt \
  --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 --eval.n_episodes=10 --env.max_parallel_tasks=1 \
  --output_dir=benchmark/eval/pi_wm_n8
```

Step 3 should reproduce ≈97.5 (their published number, sanity check for the setup).
Step 4 is the experiment. `--policy.n_action_steps=10` matches the published protocol;
one scoring decision per 10 env steps keeps the N× inference cost off the critical path.

## Results table (fill in)

| Policy | Spatial | Object | Goal | Long (10) | Average |
|---|---|---|---|---|---|
| pi0.5 published | 97.0 | 99.0 | 98.0 | 96.0 | **97.5** |
| pi0.5 reproduced (n=1) | | | | | |
| pi-WM n=4 | | | | | |
| pi-WM n=8 | | | | | |
| pi-WM n=16 | | | | | |

## Knobs

- `--policy.n_samples` — compute vs quality; try 4/8/16 (GB10's 128GB takes batch-16
  pi0.5 inference comfortably)
- `train_scorer.py --future-offset` — how far the WM imagines (default 10 frames)
- `train_scorer.py --steps` — scorer training length; underfit scorer = no gain

## Files

- [configuration_piwm.py](configuration_piwm.py) / [modeling_piwm.py](modeling_piwm.py) —
  PI05Policy subclass; only `predict_action_chunk` is overridden. Weights-compatible
  with any pi05 checkpoint (the scorer ships in its own file).
- [scorer.py](scorer.py) — 12.5M world-model + progress scorer, JEPA training losses
- [train_scorer.py](train_scorer.py) / [make_checkpoint.py](make_checkpoint.py) — GX10 steps 1-2
- [smoke_test.py](smoke_test.py) — offline checks (run on the Mac, no downloads)

## The long-horizon / unseen-task experiment (eval_hierarchical.py)

Best-of-N can't fix LIBERO-Long: on unseen *compositions*, all N samples are bad.
But pi0.5 has seen every atomic skill — it's the composition that's out of
distribution. [eval_hierarchical.py](eval_hierarchical.py) closes that gap by
composing this repo's full stack around the frozen policy:

```
long instruction ─► CoT decompose (VLM): atomic sub-commands in training phrasing
       │                    (with lessons from episodic memory injected)
       ▼
pi-WM executes sub-command k as its task string   [best-of-N per decision]
       │         action queue flushed on every sub-goal switch
       ▼
verify every 40 steps: grounder evidence (object→destination distance)
       + CoT VLM check ─► advance / retry once / fail
       ▼
failures logged to memory ─► episode ep+1 plans around them  [lifelong learning]
```

```bash
export MUJOCO_GL=egl      # ollama must be reachable for the CoT layers
# the experiment:
PYTHONPATH=. python pi_wm/eval_hierarchical.py --policy-path pi_wm_checkpoint \
    --scorer-path outputs/scorer_libero.pt --suite libero_10 --n-episodes 10 --grounder
# baselines under the IDENTICAL loop:
#   plain pi0.5:            --no-plan --n-samples 1
#   +best-of-N only:        --no-plan
#   +hierarchy, no memory:  --no-memory
```

That's a 4-row ablation on `libero_10` isolating each contribution: best-of-N,
CoT decomposition, and memory. `benchmark/hier_results.json` accumulates rows;
`benchmark/libero_memory.jsonl` is the memory (delete it to reset lifelong state;
keep it to measure cross-episode learning — LIBERO's actual theme).

Honest expectations: the decomposition helps exactly where pi0.5's long-task
failures are compositional (published hierarchical-VLA results suggest double-digit
gains on long suites); memory helps only when failures repeat across episodes;
neither helps when an atomic skill itself is broken — that residual is what the
scorer's best-of-N attacks. Caveat for the diary: memory-on runs are a *lifelong*
protocol, not comparable to single-episode published numbers — report both.

## Grasp affordances and grip effort in pi-WM

Same semantics as the real-robot stack (see `so_brain/README.md`), adapted to a
language-conditioned policy:

- **Effort**: `batch["effort"]` (preferred; the hierarchical harness sets it from the
  decompose plan) or an `effort@X` task tag — which the policy parses and **strips
  before pi0.5's tokenizer sees the text**, so the language input stays in
  distribution. The selected chunk's gripper channel is scaled around its first step
  (`g' = g₀ + effort·(g − g₀)`); candidates are scored *unscaled* so the scorer stays
  in the demo distribution. Honest note: LIBERO's sim gripper is forgiving — expect
  effort to matter mostly when pi-WM runs on real hardware, not in the benchmark.
- **Grasp part**: pi0.5 takes no points, so the grasp phrase steers through
  *language*: when a grasp sub-goal fails once, the retry is rephrased with the
  decompose plan's grasp hint ("…, gripping it by the handle of the frying pan").
  First attempts keep training-distribution phrasing; only retries deviate — the
  plain phrasing already failed, so the OOD risk is worth it.

## The committee — "Pragmatic Chaos" for manipulation (committee.py)

The Netflix Prize ensemble won by *blending* predictions — and Netflix never
deployed it (engineering cost beat the gain). Two lessons carried into
[committee.py](committee.py):

1. **Select, don't average.** Ratings are scalars, errors cancel; actions are
   multimodal — the mean of "around the left" and "around the right" hits the
   obstacle. So the committee is heterogeneous best-of-N: every member policy
   (pi0.5, MolmoAct2, VLA-JEPA, … any LeRobot-loadable checkpoint fine-tuned on the
   benchmark) proposes chunks; a sanity filter vetoes degenerate ones (the "IK
   reflex" — bounds/finiteness in action space; on the real SO-101, lerobot's
   `RobotKinematics` FK adds workspace checks); ONE env-space world-model judge
   (`train_scorer.py --env-space`) picks the winner. Errors decorrelate across
   architectures exactly as in Netflix — via argmax instead of averaging.
2. **Measure whether the ensemble pays.** K members cost K inferences. The printed
   per-member pick tally is the committee's "blend weights" — a member that never
   wins gets dropped.

```bash
python pi_wm/train_scorer.py --env-space --out outputs/scorer_env.pt   # judge in env action space
python pi_wm/committee.py --member pi_wm_checkpoint:8 --member <other_ft_checkpoint>:4 \
    --scorer outputs/scorer_env.pt --suite libero_object --n-episodes 10
```

Weight-merging across architectures is impossible (no shared parameter space between
a PaliGemma and a Qwen backbone); the true "all models in one" is **distillation** —
train one student on the committee's selected outputs — worthwhile only after the
tally proves the committee actually beats its best member.

The full "merge everything" picture, for the record — each part already exists here,
composed rather than fused: LLM/VLM reasoning = the CoT planner/diagnoser; grounding
= OWLv2/LocateAnything; action generation = the committee; physics judgment = the
world-model scorer; kinematics = the reflex filter (+ FK/IK primitives on the real
arm); learning from failure = episodic memory. Composition is what today's evidence
supports; distillation into one network is the research step after.

## Upgrade paths, in order of expected value

1. **Bigger N + temperature on the noise** — pure compute, zero code.
2. **Scorer sees the wrist camera too** — second image key, concat latents.
3. **Fine-tune pi0.5 itself with the JEPA auxiliary loss** (the wm_act recipe at 3B
   scale) — the full "world-model VLA"; costs real GX10 training time but attacks the
   failures best-of-N can't fix (when *all* N samples are bad).

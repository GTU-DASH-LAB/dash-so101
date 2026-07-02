# SO-Brain — language-driven, zero-shot control for the SO-101

Say **"put the red pen in the cup"** — any phrasing, objects never seen in any demo —
and the arm does it. This works by *factoring* generalization instead of training it:

```
 "put the red pen in the cup"
        │
        ▼
 ① PLANNER — open VLM (qwen2.5vl via ollama), frozen        [language understanding]
        │   {"object": "red pen", "destination": "cup"}
        ▼
 ② GROUNDER — OWLv2 open-vocab detector, frozen             [zero-shot object generalization]
        │   pick point (0.71, 0.80) · place point (0.39, 0.82)
        ▼
 ③ HANDS — Point-ACT, 52M params, trained on YOUR demos     [the robot-specific motion]
        │   markers drawn into the image: green = pick, blue = place
        ▼
 30Hz joint control on the SO-101 (via the standard lerobot-rollout)
```

**The key idea:** the policy never learns *what* objects are — it learns "grasp whatever
is under the green marker, place it at the blue marker." Object identity and language
live entirely in the frozen web-scale models, which are zero-shot by construction. Your
50 demos only have to teach the *motion*. This is the SayCan / MOKA / RoboPoint /
Helix-style System-2/System-1 factorization, in its lightest practical form.

Verified on this repo's real scene image (no robot needed):
`"a pen"` → (0.711, 0.803), `"a black mouse pad"` → (0.395, 0.817) — both within ~1% of
ground truth. See `./run.sh --dry-run` below to reproduce.

## Components

| File | Role | Model | Runs on |
|---|---|---|---|
| [planner.py](planner.py) | instruction → object/destination phrases | any OpenAI-compatible VLM (default ollama `qwen2.5vl:7b`); regex fallback for offline | Mac or GX10 |
| [grounding.py](grounding.py) | phrase → image point | `google/owlv2-base-patch16-ensemble` (~600MB, frozen) | Mac (MPS/CPU, ~1s) |
| [../point_act/](../point_act/) | points → 30Hz motor control | Point-ACT (subclasses WM-ACT: keeps the world-model loss) | Mac MPS |
| [relabel.py](relabel.py) | auto-label training episodes with points | OWLv2 | GX10 |
| [run.py](run.py) / [run.sh](run.sh) | glue: capture → plan → ground → rollout | — | Mac |

## Setup

```bash
# LLM planner (optional but recommended; regex fallback works for simple phrasings):
brew install ollama && ollama pull qwen2.5vl:7b       # or run ollama on the GX10 and set
export SO_BRAIN_LLM_ENDPOINT=http://<gx10-ip>:11434/v1  # to use its GPU from the Mac
```

## Train (GX10)

```bash
./point_act/train_gx10.sh
```

This first runs [relabel.py](relabel.py) (OWLv2 finds the pen + mouse pad in frame 0 of
each of the 50 episodes → `outputs/point_labels_pencil.json`), then `lerobot-train` with
`--policy.type=point_act`. During training the policy sees each episode's image with the
markers burned in, so it learns marker-relative behavior.

## Run (Mac, robot connected)

```bash
./so_brain/run.sh "put the pen on the black mouse pad"            # full pipeline
./so_brain/run.sh "..." --dry-run                                 # no robot: prints points,
                                                                  #   saves so_brain/last_plan.png
./so_brain/run.sh "..." --no-llm                                  # regex planner, no ollama
./so_brain/run.sh --object "a pen" --place "a cup" --dry-run      # bypass planner
```

## The data recipe that unlocks real zero-shot

With only pen episodes, the policy can cheat: the pen's appearance and the green marker
are perfectly correlated, so it may key on the pen and ignore the marker. The fix is
data, not architecture — record **one new session (~60 episodes) with a different random
object each few episodes** (eraser, cube, spoon, bottle cap, USB stick…), varied positions,
varied destinations. Then the markers are the *only* invariant signal and the policy is
forced to follow them. After that retrain, any object OWLv2 can name becomes pickable —
that's the zero-shot claim, and `relabel.py` auto-labels the new session too (it parses
each episode's task string for the object phrase).

Success-rate protocol: 10 trials × {trained object, 3 novel objects} × {trained
destination, novel destination}, count successes, compare against SmolVLA given the
same novel instructions.

## Honest limits

- **Skill class is fixed**: tabletop pick-and-place. "Pour the water" or "open the
  drawer" needs new demos of those *motions* — no architecture fixes that at this scale.
- **Grounding is static per rollout**: points are computed once from the pre-rollout
  frame. Fine for static scenes; re-grounding mid-rollout is the upgrade path if
  objects get moved mid-task.
- **Detector misses**: OWLv2 confidence on unusual phrasings/objects can dip (threshold
  0.1 default). The annotated preview (`last_plan.png`) exists so you catch bad
  groundings before the arm moves.
- **The planner is as good as the VLM**: qwen2.5vl handles pick-and-place extraction
  easily; complex multi-step instructions ("stack all three cubes") need a loop over
  sub-goals — deliberately not built yet.

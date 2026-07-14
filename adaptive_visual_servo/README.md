# adaptive_visual_servo — uncalibrated adaptive visual servoing for SO-101

Pick-and-place with **no camera calibration, no hand-eye calibration, no accurate
kinematic model and no markers**. The controller estimates the image Jacobian
(the "parameters") online while working — motor babbling bootstraps it, Broyden
updates track it — and closes the loop on a single fixed RGB camera
(eye-to-hand). Runs in a self-contained simulation (`run_sim.py`) and has a
real-hardware mode (`run_real.py`, untested until tried on the arm).

## Assumptions (verify on hardware)

- One RGB webcam, **fixed** eye-to-hand (overhead-ish, like this project's `top`
  camera), static background, static lighting during an episode.
- Objects sit on a known table plane; a colored drop pad/container is in view.
- The arm is the only thing that moves while the system runs.

## Why this design (vs. the original phase draft)

| Original draft | Problem | This design |
|---|---|---|
| Faz 0: before/after frame diff, largest blob centroid = EE | One diff blob spans **both** arm poses → centroid ≈ midpoint of motion, a biased EE estimate; whole-arm motion also biases it toward the forearm, not the tip | **Gripper blink**: open/close the gripper a few mm between two frames. The diff isolates only the fingers → precise, markerless EE pixel. Used to bootstrap and to re-anchor whenever tracking degrades |
| Faz 1: HSV threshold | Fails for gray/black/multicolor objects — not "every object possible" | **Background subtraction** vs. a one-time empty-workspace frame → any new blob is an object regardless of color/shape. HSV is an optional *selector* ("pick the red one"). `detector` is a plug point — a learned model (e.g. YOLO on the GX10) can replace it without touching the control code |
| Faz 2: Δq = −λ Ĵ⁺e, Broyden | Plain pinv blows up near singular configs; unconditional Broyden updates corrupt Ĵ with noise when steps are tiny; nothing detects divergence | **Damped (Tikhonov) pseudo-inverse**, per-step joint clamp, Broyden gated on min ‖Δq‖, **divergence watchdog**: error grows K steps → re-blink to re-anchor the EE; still failing → local re-babble |
| Faz 3: align XY once, then blind Z descend | With a tilted camera, image alignment at approach height has a **parallax offset** on the table plane (≈ Δh·tanθ — easily >1 cm) | **Interleaved descend**: drop 2 cm open-loop (nominal FK), re-servo XY visually, repeat. Parallax → 0 as the gripper reaches object height. Model error in Z is tolerated because vision keeps fixing XY |
| Faz 4: close gripper, read load | fine | Same. Sim: capture radius. Real: stop closing on Present_Load spike + timeout |
| Faz 5: servo to drop zone | EE can't blink while holding the object | **Patch tracker** (template matching in a small ROI, re-initialized after grasp) with open-loop prediction Ĵ·Δq as fallback for the carry phase; pad located once at start via HSV (the pad is ours, so its color is known) |

Plus a retrying **state machine** over all phases:
`DETECT → BABBLE → SERVO_XY → DESCEND(interleaved) → GRASP → LIFT → TRANSPORT → RELEASE → HOME`.

## What makes it "adaptive control"

The image Jacobian **is** the parameter set being estimated. It absorbs, in one
2×N linear map: camera intrinsics/extrinsics, link-length errors, servo gain
errors, mounting offsets. Babbling = initial identification (least squares over
random (Δq, Δs) probe pairs); Broyden = recursive estimation during work
(`Ĵ += β(Δs − ĴΔq)Δqᵀ/‖Δq‖²`). The only place a kinematic model is used at all
is the open-loop Z descend — and it's allowed to be wrong by design.

## Simulation honesty rules

The sim executes with **true** link lengths, true camera, gaussian pixel noise,
joint actuation noise. The controller gets:

- rendered images (that's all the "camera" is),
- a **nominal model with ±5% link-length error** (descend only),
- joint commands it issued (no encoder ground truth used for control).

Ground truth (true FK, true projection) is used **only** by tests to validate
the babbled Jacobian and to score grasp/place success. Nothing in `control.py`
touches `SimWorld`'s true state.

## Files

- `config.py` — every tuning knob: sim geometry/noise, servo gains, real-robot ports/limits.
- `sim.py` — 3-DOF (pan/shoulder/elbow) + gripper arm, pinhole camera, cv2 renderer, attach-on-grasp physics.
- `perception.py` — blobs, background-subtraction detector, HSV selector, blink locator, patch tracker.
- `control.py` — babbling, Broyden servo, interleaved descend, episode state machine. Robot-agnostic: anything with `set_q/get_q/set_gripper/gripper_contact/read frame` works.
- `run_sim.py` — CLI: run N randomized episodes, report success rate, optionally dump annotated frames.
- `run_real.py` — SO-101 adapter (lerobot `SO101Follower` + cv2 camera). **Untested on hardware.**
- `tests/` — per-phase assert scripts; run any with `.venv/bin/python adaptive_visual_servo/tests/test_X.py` (pytest-compatible too).

## Run

```bash
source .venv/bin/activate
python adaptive_visual_servo/run_sim.py --episodes 10 --seed 0
python adaptive_visual_servo/run_sim.py --episodes 1 --save-frames /tmp/avs_frames  # visual debug
for t in adaptive_visual_servo/tests/test_*.py; do python "$t"; done
```

## Real hardware (later)

`run_real.py --port /dev/ttyACM0 --camera 0`. Read its safety notes first:
per-step clamp (`MAX_STEP_DEG`), soft joint limits, gripper load threshold, and
a REQUIRED first run with the arm's motion range verified by hand. Ports swap
after replug (see repo CLAUDE.md); re-check with `lerobot-find-port`.

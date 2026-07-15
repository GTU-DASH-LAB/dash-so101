# adaptive_visual_servo — uncalibrated adaptive visual servoing for SO-101

Pick-and-place with **no camera calibration, no hand-eye calibration, no accurate
kinematic model and no markers**. The controller estimates the image Jacobian
(the "parameters") online while working — motor babbling bootstraps it, Broyden
updates track it — and closes the loop on a single fixed RGB camera
(eye-to-hand). Two simulation backends (`run_sim.py`'s fast toy 3-DOF world,
`run_pb_sim.py`'s pybullet 6-DOF world over the real downloaded URDF, watchable
live with `--gui`) plus a real-hardware mode (`run_real.py`, untested until
tried on the arm) — all three share the exact same `control.py`/`perception.py`
via one duck-typed rig interface.

## Assumptions (verify on hardware)

- One RGB webcam, **fixed** eye-to-hand (overhead-ish, like this project's `top`
  camera), static background, static lighting during an episode.
- Objects sit on a known table plane; a colored drop pad/container is in view.
- The arm is the only thing that moves while the system runs.

## Why this design (vs. the original phase draft)

| Original draft                                             | Problem                                                                                                                                                              | This design                                                                                                                                                                                                                                                                                                  |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Faz 0: before/after frame diff, largest blob centroid = EE | One diff blob spans**both** arm poses → centroid ≈ midpoint of motion, a biased EE estimate; whole-arm motion also biases it toward the forearm, not the tip | **Gripper blink**: open/close the gripper a few mm between two frames. The diff isolates only the fingers → precise, markerless EE pixel. Used to bootstrap and to re-anchor whenever tracking degrades                                                                                               |
| Faz 1: HSV threshold                                       | Fails for gray/black/multicolor objects — not "every object possible"                                                                                               | **Background subtraction** vs. a one-time empty-workspace frame → any new blob is an object regardless of color/shape. HSV is an optional *selector* ("pick the red one"). `detector` is a plug point — a learned model (e.g. YOLO on the GX10) can replace it without touching the control code |
| Faz 2: Δq = −λ Ĵ⁺e, Broyden                           | Plain pinv blows up near singular configs; unconditional Broyden updates corrupt Ĵ with noise when steps are tiny; nothing detects divergence                       | **Damped (Tikhonov) pseudo-inverse**, per-step joint clamp, Broyden gated on min ‖Δq‖, **divergence watchdog**: error grows K steps → re-blink to re-anchor the EE; still failing → local re-babble                                                                                         |
| Faz 3: align XY once, then blind Z descend                 | With a tilted camera, image alignment at approach height has a**parallax offset** on the table plane (≈ Δh·tanθ — easily >1 cm)                           | **Interleaved descend**: drop 2 cm open-loop (nominal FK), re-servo XY visually, repeat. Parallax → 0 as the gripper reaches object height. Model error in Z is tolerated because vision keeps fixing XY                                                                                              |
| Faz 4: close gripper, read load                            | fine                                                                                                                                                                 | Same. Sim: capture radius. Real: stop closing on Present_Load spike + timeout                                                                                                                                                                                                                                |
| Faz 5: servo to drop zone                                  | EE can't blink while holding the object; template matching on the carried object drifts as its viewing angle rotates over a long transport (confirmed: false convergence on a ~200px move, and separately locks onto a distractor object) | **Background-subtraction locator** (`locate_by_diff`): re-run the same bg-sub used for Faz 1 in a small ROI around the kinematic prediction Ĵ·Δq. Invariant to the object's appearance/rotation (only asks "differs from the empty-workspace photo", not what it looks like); a jump gate vs. the prediction rejects a distractor (e.g. another not-yet-picked object) landing in the ROI. Pad located once via HSV (the pad is ours, so its color is known) |

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
- `sim.py` — fast toy 3-DOF (pan/shoulder/elbow) + gripper arm, analytic FK, cv2 renderer,
  attach-on-grasp physics. Used by most of the test suite (hundreds of episodes need to run fast).
- `pb_sim.py` — the **real SO-101 URDF** (`assets/SO101/`) loaded into pybullet: genuine 6-DOF
  kinematics, physics stepping, and a rendered camera. `--gui` opens a live 3D window. Same rig
  interface as `sim.py`'s `SimWorld`, so `control.py`/`perception.py` don't change between them.
- `lerobot_ik.py` — `PlacoModel`, the open-loop-descend model for `pb_sim.py`/`run_real.py`,
  backed by lerobot's own `RobotKinematics` (placo) over the real URDF — not a hand-approximated
  link-length model like `sim.py`'s `NominalModel`. Only used for the Z move; XY stays visually closed.
- `nanodet_detector.py` — `NanodetDetector`, a real trained object detector (NanoDet-Plus, ONNX
  inference only — no PyTorch training deps) plugged in through `run_episode`'s `detector` argument.
  **The default detector in `run_real.py`** (real cameras see real COCO objects: ball, bottle, cup,
  fruit, ...); its blobs carry mean-color too, so `--hue` selection works with it. Learned detections
  are always cross-checked against background subtraction (`perception.filter_by_background`): a
  pickable object must also differ from the empty-workspace photo — this drops detections of the
  robot arm itself, which otherwise becomes the "largest object" and the servo chases its own arm
  (observed live). The sim's abstract painted shapes don't resemble any COCO class, so background
  subtraction stays the sim's default.
- `perception.py` — blobs, background-subtraction detector, HSV selector, blink locator, bg-diff carry-phase locator.
- `control.py` — babbling, Broyden servo, interleaved descend, episode state machine. Robot-agnostic
  and dimension-agnostic: works with any `set_q/get_q/set_gripper/gripper_contact/read` rig, any DOF count.
- `run_sim.py` — CLI: run N randomized episodes on the toy sim, report success rate, dump frames.
- `run_pb_sim.py` — CLI: same, on the real-URDF pybullet backend. `--gui` to watch it live.
- `run_real.py` — SO-101 hardware adapter (lerobot `SOFollower` + `OpenCVCamera`), same duck-typed rig
  interface so `control.py`/`perception.py` are unchanged between sim and real. Drives all 6 real
  motors (5 arm joints directly through the visual servo, gripper separately).
  **Untested on hardware** — the arm/camera aren't connected to this machine. Includes
  `--calibrate-gripper` to find real `gripper_open_pos`/`gripper_closed_pos`/`load_threshold` values.
- `assets/` — downloaded, not authored here: `SO101/` is the real URDF + meshes
  (TheRobotStudio/SO-ARM100), `nanodet/` is the NanoDet-Plus ONNX checkpoint (RangiLyu/nanodet).
- `tests/` — per-phase assert scripts; run any with `.venv/bin/python adaptive_visual_servo/tests/test_X.py` (pytest-compatible too).

## Run

```bash
source .venv/bin/activate
python adaptive_visual_servo/run_sim.py --episodes 10 --seed 0
python adaptive_visual_servo/run_sim.py --episodes 1 --save-frames /tmp/avs_frames  # visual debug
python adaptive_visual_servo/run_pb_sim.py --gui --episodes 2 --seed 9             # watch it live
for t in adaptive_visual_servo/tests/test_*.py; do python "$t"; done
```

## Known limitations

`run_pb_sim.py` / `pb_sim.py` is **not yet at the toy sim's reliability** (which is 100% on its
own randomized batch). Grasping is **contact-triggered** (a jaw must physically touch the object
during the closing sweep; no proximity shortcut), so every cm of perception error directly costs
grasps. Two real adaptive-control defects were found and fixed by tracing failures on this backend:

- **Single-moving-jaw aim bias**: the SO-101 has one moving jaw, so the blink diff tracks the *jaw*,
  not the grasp point where the jaws meet — a systematic 25-48px aim error the Jacobian can't fix
  because it's inside the measurement. Fixed by `perception.calibrate_grasp_frame`: once per
  episode, in free air, a near-closed blink measures the true closure point directly and stores its
  offset in the jaw-sweep direction frame; `locate_grasp_point` then reconstructs it from safe
  open-range blinks (measured 2-10px residual across poses, vs 25-48px uncorrected). The toy sim's
  symmetric gripper degrades gracefully to a zero offset.
- **z-hold runaway near full extension**: the per-step height-hold correction was uncapped while the
  visual servo is clamped at `dq_max` — near the arm's reach limit the IK returns huge joint deltas
  for small height errors, and the hold was observed dragging a nearly-converged EE monotonically
  off-target for 10+ steps. Fixed by clamping the hold term to the servo's own authority.

These moved failures downstream (approach servo used to fail outright; episodes now regularly reach
grasp attempts) but a 3-seed dev batch still delivered 0/3: the remaining gap is last-centimeter
grasp mechanics — fine-servo tolerance (12px ≈ 1cm) plus calibration residual is marginal against
~1cm objects, and a striking jaw can knock the object away (mitigated with heavier/damped objects
and faster weld latching, not eliminated). `tests/test_pb_e2e.py` is scoped to what's actually
solid: every episode returns a well-formed result without crashing or hanging, within a bounded
step/retry budget (also guards two earlier bugs: an unbounded descend loop that spun 29+ minutes
when placo's IK stalled, and an uncaught exception crashing episodes when babble recovery failed).
Closing the delivery gap is real follow-up work, not something faked here.

## Real hardware (later)

```bash
python adaptive_visual_servo/run_real.py --calibrate-gripper   # find real gripper values first
python adaptive_visual_servo/run_real.py --port /dev/ttyACM0 --camera 0 --episodes 5
python adaptive_visual_servo/run_real.py --classes "sports ball,cup"   # narrow nanodet's classes
python adaptive_visual_servo/run_real.py --detector bgsub              # any-object fallback
```

Read `run_real.py`'s module docstring first: per-step clamp (`RealConfig.max_step_deg`),
soft joint limits, gripper load threshold, and a REQUIRED first run with the
arm's motion range verified by hand (hand near the power switch). Ports swap
after replug (see repo CLAUDE.md); re-check with `lerobot-find-port`. Descend uses
`lerobot_ik.PlacoModel` against the real downloaded URDF — no per-arm length
calibration needed there; expect the same real-mesh perception reliability
gap noted above until `blink_locate` is retuned against the actual camera.

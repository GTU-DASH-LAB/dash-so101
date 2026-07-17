# Adaptive Visual Servo — Mathematical Reference

Complete equation reference for the `adaptive_visual_servo` system. Every equation
is tagged with where it lives in the code and when/why it fires at runtime.

---

## Table of Contents

1. [Image Jacobian & Adaptive Estimation](#1-image-jacobian--adaptive-estimation)
2. [Damped Pseudo-Inverse (Tikhonov Regularization)](#2-damped-pseudo-inverse-tikhonov-regularization)
3. [Visual Servo Control Law](#3-visual-servo-control-law)
4. [Broyden Rank-1 Update](#4-broyden-rank-1-update)
5. [Forward Kinematics](#5-forward-kinematics)
6. [Inverse Kinematics (Damped Least Squares)](#6-inverse-kinematics-damped-least-squares)
7. [Camera Resectioning (Normalized DLT)](#7-camera-resectioning-normalized-dlt)
8. [Gauss-Newton Tool Offset Fitting](#8-gauss-newton-tool-offset-fitting)
9. [Fused Tracker — Projection & Jacobian](#9-fused-tracker--projection--jacobian)
10. [Blink Localization Geometry](#10-blink-localization-geometry)
11. [Grasp-Frame Calibration](#11-grasp-frame-calibration)
12. [Background Subtraction & Object Detection](#12-background-subtraction--object-detection)
13. [Carry-Phase Tracking (locate_by_diff)](#13-carry-phase-tracking-locate_by_diff)
14. [Cross-Validated Reprojection Error Gate](#14-cross-validated-reprojection-error-gate)
15. [Episode Pipeline Summary](#15-episode-pipeline-summary)
16. [AruCo Marker Localization & Tracking](#16-aruco-marker-localization--tracking)
17. [Parametric Trajectory Kinematics (Shapes)](#17-parametric-trajectory-kinematics-shapes)
18. [Recursive Base-Offset Estimation (EKF)](#18-recursive-base-offset-estimation-ekf)
19. [Null-Space Secondary Objectives](#19-null-space-secondary-objectives)

---

## 1. Image Jacobian & Adaptive Estimation

### Definition

The **image Jacobian** (also called the *interaction matrix*) `J ∈ ℝ^(2×N)` maps
infinitesimal joint-space velocity to image-plane velocity:

```
ds = J · dq
```

where:
- `s ∈ ℝ²` is the end-effector pixel position `(u, v)` in the image
- `q ∈ ℝ^N` is the joint angle vector (N = 3 for the toy sim, 5 for the real arm)
- `ds = s_new − s_old` is the observed pixel displacement
- `dq = q_new − q_old` is the executed joint displacement

**Key insight**: `J` encapsulates *everything* — camera intrinsics, extrinsics,
arm geometry, mounting offsets — in one 2×N matrix that can be identified purely
from (dq, ds) observations, with no explicit model of any of those components.

### Motor Babbling (Initial Identification)

**Code**: `control.babble()` (line 38)

Given `M` usable probe pairs `{(dq_k, ds_k)}`, stack into matrices:

```
Q = [dq_1, dq_2, ..., dq_M]ᵀ  ∈ ℝ^(M×N)
S = [ds_1, ds_2, ..., ds_M]ᵀ  ∈ ℝ^(M×2)
```

The Jacobian estimate minimizes the total squared prediction error:

```
Ĵ = argmin_J  Σ_k ‖ds_k − J · dq_k‖²
```

This is a standard linear least-squares problem. Taking the derivative w.r.t. J
and setting to zero:

```
Ĵᵀ = (QᵀQ + εI)⁻¹ QᵀS
```

where `ε = 10⁻⁹` is a tiny regularizer for numerical stability (the matrix is
almost always well-conditioned because babble probes span the joint space — an
explicit rank check guards this).

**Step-by-step in the code**:
1. Start at current pose `q₀`, blink-locate the EE → `s₀`
2. For each of `babble_probes` random steps:
   - Generate `q_target = q₀ + U(-1.5, 1.5) · amp` (leash: stay near `q₀`)
   - Clamp step to `±amp`, execute `rig.set_q(q + step)`
   - Read actual `q` post-clamp, compute `dq = q_after − q_before`
   - Blink-locate → `s_new`; if None or jumped >120px, skip (glitch)
   - Record `(dq, s_new − s)`; update `s ← s_new`
3. Require `M ≥ 2N` usable pairs and `rank(Q) = N`
4. Solve via `numpy.linalg.solve(QᵀQ + εI, QᵀS)` → `Jᵀ`
5. Return `J = Jᵀ.T` and the final `s`

---

## 2. Damped Pseudo-Inverse (Tikhonov Regularization)

**Code**: `control.damped_pinv()` (line 22)

The naive pseudo-inverse `J⁺ = Jᵀ(JJᵀ)⁻¹` blows up when `J` is near-singular
(arm near a kinematic singularity, or Broyden corruption). Tikhonov damping
stabilizes it:

```
J⁺_μ = Jᵀ(JJᵀ + μI)⁻¹
```

where the damping factor `μ` is **relative to J's scale**:

```
μ = λ_damp · tr(JJᵀ) / m + ε
```

- `λ_damp = 0.01` (`ServoConfig.damping`)
- `m = 2` (image dimensions)
- `ε = 10⁻⁹` prevents division by zero if `J = 0`

**Why relative damping**: `J`'s entries are px/rad in the toy sim (~100s) but
px/deg on the real arm (~1s). Scaling `μ` by `tr(JJᵀ)` makes the same
`damping=0.01` work on both without retuning.

**Effect**: near a singularity, instead of commanding infinite joint velocity
(which `dq_max` would clamp to a random direction), the damped inverse
gracefully reduces the step magnitude, keeping the servo well-behaved.

---

## 3. Visual Servo Control Law

**Code**: `control.servo_to()` (line 91), inner loop

The control law at each step:

```
e = s_target − s_ee         (pixel error)
dq = J⁺_μ · (λ(e) · e)      (velocity command)
dq = clip(dq, −dq_max, dq_max)  (per-joint clamp)
dq = clip(dq + (I − J⁺_μ·J) · dq_null, −dq_max, dq_max)   (null-space term, §19)
```

where **`λ(e)` is the adaptive control gain** (error magnitude, stiction, and
tracking-rate terms):

```
λ(e) = λ_min + (0.95 − λ_min) · exp(−‖e‖ / 15.0)
λ(e) ← min(0.95, λ(e) · (1 + 0.15 · stall))
```

- `λ_min = 0.5` (`ServoConfig.lam`) — base fraction of error corrected.
- **Stiction compensation**: as the error `‖e‖` decreases, `λ(e)` smoothly increases towards `0.95`, boosting joint velocity commands to ensure they remain above the stiction/deadband threshold of physical Feetech servos (STS3215), preventing lockup.
- **Tracking-rate term**: `stall` counts consecutive measurements with no
  improvement in `‖e‖`. Stalled measured progress means the commanded velocity
  is being eaten (stiction, under-modeled J), so `λ` is boosted proportionally,
  capped at 0.95.

**With height hold** (`shape_dq`): the image servo controls XY, but a pixel
target is a whole 3D ray — without Z constraint the EE slides down the ray
into the table. A parallel height-hold term adds:

```
dq_total = clip(dq_servo + dq_z_hold, −0.2, 0.2)
```

where `dq_z_hold = clip(model.step_dz(q, 0.6·(z_ref − z_current)), −dq_max, dq_max)`.

**Convergence declared only on fresh measurements**: the loop dead-reckons
`s_pred = s + J·dq` between real measurements (every `measure_every` steps).
Success (`‖e‖ < tol`) requires a real measurement confirming it, never just a
prediction.

---

## 4. Broyden Rank-1 Update

**Code**: `control.broyden_update()` (line 42)

Between babble re-calibrations, the Jacobian is updated recursively using the
Broyden rank-1 formula (a secant method for matrix equations):

```
Ĵ_new = Ĵ + β_adaptive · (ds − Ĵ · dq) · dqᵀ / ‖dq‖²
```

where:
- `ds = s_measured − s_base` (actual pixel displacement since last measurement)
- `dq = Σ dq_actual` (accumulated joint displacement since last measurement)
- `‖dq‖² = dqᵀ · dq` (squared norm)

**`β_adaptive` is the adaptive learning rate** based on prediction error:

```
pred_err = ‖ds − Ĵ · dq‖
rel_err = pred_err / (‖ds‖ + 10⁻⁵)
β_adaptive = clip(β · (0.5 + rel_err), β · 0.5, min(1.6 · β, 0.95))
```

- `β = 0.5` (`ServoConfig.broyden_beta`) — base learning rate.
- **High relative prediction error** (Jacobian is wrong/outdated): `β_adaptive` is boosted up to `1.6 · β = 0.80`, forcing rapid correction.
- **Low relative prediction error** (Jacobian is accurate): `β_adaptive` is lowered to `0.5 · β = 0.25`, filtering out pixel noise and preventing Jacobian corruption.

**Gating**: the update is skipped when `‖dq‖ < min_dq = 0.004` — tiny steps
carry mostly noise and would corrupt `J` via the `1/‖dq‖²` amplification.

**Why Broyden, not re-babble every time**: babbling takes ~22 probes × (blink
time + settle time) — seconds of robot motion per calibration. Broyden gives
free, incremental refinement from the work the servo is already doing. It
drifts slowly (no second-order information), so the escalating recovery
(re-anchor blink → re-babble) catches degradation.

**In fused mode**: Broyden is **completely replaced** by the analytical
Jacobian from the fused camera+FK model — no drift, no corruption, no learning
rate. This is one of the fused tracker's key advantages.

---

## 5. Forward Kinematics

### 5a. Toy Sim (3-DOF): Analytic FK

**Code**: `sim.fk_points()` (line 47)

A planar 2R arm with a pan joint at the base:

```
q = (q₀, q₁, q₂)   — pan, shoulder, elbow

shoulder = (0, 0, L_base)

elbow = shoulder + (r₁·cos(q₀), r₁·sin(q₀), z₁)
  where r₁ = L₁·cos(q₁),  z₁ = L₁·sin(q₁)

EE = elbow + (r₂·cos(q₀), r₂·sin(q₀), z₂)
  where r₂ = L₂·cos(q₁ + q₂),  z₂ = L₂·sin(q₁ + q₂)
```

The **grip center** (tool center point) hangs below the EE by half the
gripper finger length:

```
grip_center = EE − (0, 0, L_gripper/2)
```

### 5b. Real URDF (5-DOF): Placo FK

**Code**: `lerobot_ik.PlacoModel.ee()` (line 29)

Uses lerobot's `RobotKinematics` (placo library) to compute the full
homogeneous transform from the URDF kinematic chain:

```
T = FK(q) ∈ SE(3)     — 4×4 homogeneous transform

position: p = T[0:3, 3]
rotation: R = T[0:3, 0:3]
```

Joint angles are stored in **radians** internally but placo works in degrees:

```
T = kin.forward_kinematics(degrees(q))
p = T[:3, 3]
```

The 5 joints are: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll`.

---

## 6. Inverse Kinematics (Damped Least Squares)

### 6a. Toy Sim: Analytic 3×3 IK

**Code**: `sim.NominalModel.step_dz()` (line 90)

For the 3-DOF toy arm, the position Jacobian is 3×3 (square), computed by
central finite differences:

```
J_pos[i] = (ee(q + εeᵢ) − ee(q − εeᵢ)) / (2ε)
```

Then damped least-squares for a pure-Z target:

```
dq = (JᵀJ + μI)⁻¹ Jᵀ · [0, 0, dz]ᵀ
```

with `μ = 10⁻⁶`.

### 6b. Real URDF: Placo IK with Soft Orientation

**Code**: `lerobot_ik.PlacoModel.step_dz()` (line 33)

Uses placo's constrained IK solver, with a desired pose target that is the
current pose shifted by `(0, 0, dz)`:

```
T_target = FK(q)
T_target[2,3] += dz

q_new = IK(q_current, T_target, w_pos=1.0, w_orn=0.05)
dq = q_new − q
```

**Why `w_orn = 0.05`**:
- `w_orn = 0` (position-only): wrist drifts, eventually flips upside down
- `w_orn = 1` (equal weight): 6 constraints (3 pos + 3 orn) on 5 joints —
  generally infeasible, solver oscillates or stalls (29+ minute hang observed)
- `w_orn = 0.05`: soft tie-breaker in the null space. Fixes the drift without
  turning descent into an overconstrained problem.

---

## 7. Camera Resectioning (Normalized DLT)

**Code**: `fused_tracking.fit_projection()` (line 50)

Given `N ≥ 6` pairs of 3D world points and their 2D image projections
`{(Xᵢ, xᵢ)}`, find the 3×4 projection matrix `P` such that:

```
x ~ P · X̃     (projective equality, up to scale)
```

where `X̃ = [X, 1]ᵀ` is the homogeneous 3D point.

### Normalization (Hartley & Zisserman)

Raw DLT is numerically unstable. Standard practice: normalize both 2D and 3D
points to have zero mean and unit average distance from origin.

**2D normalization** (`_normalize_2d`):
```
c = mean(x)
s = √2 / mean(‖x − c‖)
T = [[s, 0, −s·cₓ], [0, s, −s·c_y], [0, 0, 1]]
x̃ = T · [x, 1]ᵀ
```

**3D normalization** (`_normalize_3d`):
```
c = mean(X)
s = √3 / mean(‖X − c‖)
U = [[s, 0, 0, −s·cₓ], [0, s, 0, −s·c_y], [0, 0, s, −s·c_z], [0, 0, 0, 1]]
X̃ = U · [X, 1]ᵀ
```

### DLT System

Each point pair gives 2 equations (the third is redundant due to projective
equivalence). For each pair `(X̃ᵢ, x̃ᵢ)`:

```
Row 1: [0ᵀ,  −x̃ᵢ₃·X̃ᵢᵀ,  x̃ᵢ₂·X̃ᵢᵀ]
Row 2: [x̃ᵢ₃·X̃ᵢᵀ,  0ᵀ,  −x̃ᵢ₁·X̃ᵢᵀ]
```

Stack all 2N rows into matrix `A ∈ ℝ^(2N×12)`. The solution is the right
singular vector of `A` corresponding to the smallest singular value:

```
SVD(A) = UΣVᵀ   →   p = V[-1, :]   (last row of Vᵀ)
P_norm = reshape(p, 3, 4)
```

### Denormalization

```
P = T⁻¹ · P_norm · U
P = P / ‖P[2, 0:3]‖    (normalize so depth scale ≈ 1)
```

### Projection

```
h = P · [X, 1]ᵀ     ∈ ℝ³
pixel = h[0:2] / h[2]
```

---

## 8. Gauss-Newton Tool Offset Fitting

**Code**: `fused_tracking.FusedTracker._fit_pd()` (line 115)

The blink measurement tracks the **moving jaw**, not the jaw-closure point
(the true TCP). The constant offset `d ∈ ℝ³` absorbs this systematic bias:

```
predicted_pixel = P · [FK(q) + d, 1]ᵀ
```

### Alternating Optimization

For `gn_iters = 3` iterations:

1. **Fit P** given current `d`:
   ```
   P, _ = fit_projection(X + d, px)
   ```

2. **Gauss-Newton step on d** given current `P`:
   - Compute residuals: `r = (project(P, X + d) − px).ravel()  ∈ ℝ^(2N)`
   - Compute Jacobian w.r.t. d by forward differences:
     ```
     J_d[:, k] = (project(P, X + d + ε·eₖ) − px).ravel() − r) / ε
     ```
     for `k = 0,1,2` and `ε = 10⁻⁴`
   - Gauss-Newton update:
     ```
     step = (J_dᵀ J_d)⁻¹ J_dᵀ · (−r)    (least-squares)
     d ← d + step
     ```

### Usage at Runtime

- **Fitting**: `d` soaks the blink-vs-TCP bias (the observations track the
  moving jaw, but we want to aim at the jaw-closure point)
- **Tracking**: `project(P, FK(q))` uses `d = 0` — the URDF's gripper frame
  IS the jaw-closure point; the offset was only needed during calibration
  because the *blink measurements* are biased, not the FK frame.
- **track_px**: `project(P, FK(q) + d)` — matches the blink semantics exactly,
  used for consistency with the proven blink mode.

---

## 9. Fused Tracker — Projection & Jacobian

**Code**: `fused_tracking.FusedTracker` (line 90)

### EE Pixel Prediction (Free, Instantaneous)

```
s = P · [FK(q), 1]ᵀ                  — ee_px (d=0, bare FK frame)
s = P · [FK(q) + d, 1]ᵀ             — track_px (with fitted offset)
```

Cost: microseconds. No robot motion, no camera frames. Replaces the blink
measurement for per-step tracking.

### Image Jacobian from the Fused Model

```
J_fused[:, i] = (ee_px(q + ε·eᵢ) − ee_px(q − ε·eᵢ)) / (2ε)
```

for each joint `i`, with `ε = 10⁻⁴`.

This is the **exact** numerical derivative of the calibrated projection
pipeline. Unlike the babbled `J` (which is a noisy linear fit over a
small neighbourhood) or the Broyden-updated `J` (which drifts over time),
this Jacobian is:
- **Pose-exact**: fresh at every configuration, not a stale neighbourhood fit
- **Drift-free**: no recursive updates that can corrupt it
- **Consistent**: the same model that predicts position also gives velocity

### Residual Correction

At runtime, a small image-space residual `c ∈ ℝ²` is added:

```
s_corrected = track_px(q) + c
```

`c` absorbs the pose-dependent part of FK model error that a constant `d`
cannot. It is re-anchored by one real blink measurement at each phase
transition (approach start, each descend stage):

```
c = blink_locate() − track_px(q)
```

Between re-anchors, `c` is constant (no decay in the current implementation).

---

## 10. Blink Localization Geometry

**Code**: `perception.blink_measure()` (line 98)

### Markerless EE Localization

The gripper is the only object that changes **exactly when commanded** and
**in both directions** at the **same pixel location**. This temporal structure
uniquely identifies it against any background clutter.

### Blink Protocol (5-frame, 2 independent diffs)

```
Frame sequence:
  a₁ (open) → a₂ (open)    — null pair (no command)
  [set_gripper(g_mid)]
  b₁ (mid) → b₂ (mid)      — same position, different frame
  [set_gripper(g_open)]
  c (open)

Masks:
  m_null = diff(a₁, a₂)     — in-place flicker exclusion
  m_fwd  = diff(a₂, b₁)     — first gripper sweep
  m_rev  = diff(b₂, c)       — second gripper sweep (independent)

  m = m_fwd AND m_rev        — only pixels that moved in BOTH sweeps
  m[dilate(m_null)] = 0      — exclude anything that flickered without a command
```

**Why 2 independent diffs**: a single (before, after) diff picks up codec
speckles, monitor flicker, and passing people. The AND of two *disjoint*
pairs (no shared frame) rejects all one-time transients. A moving person's
leading/trailing edges land at different pixels in each pair → AND is empty.

### Centroid

```
weights = [area₁, area₂, ...]    (each finger blob)
centers = [c₁, c₂, ...]
EE_pixel = Σ(wᵢ · cᵢ) / Σwᵢ
```

Area-weighted because each finger's blink contribution should be proportional
to how much it moved (farther finger travel → larger diff blob).

---

## 11. Grasp-Frame Calibration

**Code**: `perception.calibrate_grasp_frame()` (line 189)

### Problem

The SO-101 has a **single moving jaw**. The blink centroid tracks the moving
jaw, not the jaw-closure point (where the jaws meet) — a systematic 25-48px
aim error.

### Calibration Procedure (in free air)

```
Blink at BLINK_HI = (1.0, 0.8) → c₁  (jaw at ~90% open)
Blink at BLINK_LO = (0.6, 0.4) → c₂  (jaw at ~50% open)
Blink at BLINK_CLOSURE = (0.12, 0.0) → c_close  (near-closed, true grasp point)
```

### Jaw-Sweep Direction Frame

The vector `d = c₂ − c₁` defines the jaw's sweep direction in image space
(perpendicular to the jaw axis). Express the closure-point offset in this frame:

```
M = [d | d⊥]    — columns: sweep direction, perpendicular
d⊥ = (−d_y, d_x)

[a, b]ᵀ = M⁻¹ · (c_close − c₂)
```

`(a, b)` is the offset of the true grasp point from the `c₂` blink, expressed
in units of `d` and `d⊥`.

### Reconstruction at Any Pose

At a new pose, blink at HI and LO to get new `c₁', c₂'` and compute new `d'`:

```
grasp_point = c₂' + a · d' + b · d'⊥
```

This transfers across arm poses because the jaw's image-plane sweep direction
rotates and scales with the arm, and the offset stays proportional to the
sweep magnitude.

**Measured accuracy**: 2-10px residual across poses, vs 25-48px uncorrected.

---

## 12. Background Subtraction & Object Detection

**Code**: `perception.diff_mask()` (line 23), `perception.detect_objects()` (line 51)

### Diff Mask

```
d = |frame − background|_max     (per-pixel max across BGR channels)
d = d − median(d)                 (cancel global exposure shift)
d = GaussianBlur(d, 5×5)         (noise suppression)
mask = (d > threshold)
```

**Why median subtraction**: real webcams auto-adjust exposure between captures.
The median of the whole-frame diff is dominated by unchanged pixels (the moving
object covers a small fraction), so it estimates the global exposure pedestal.
Removing it keeps thresholding about *motion*, not *lighting*.

### Object Detection

Any connected component in the diff mask with `area ≥ min_area` pixels and
`area ≤ max_area` is a candidate object. Components are sorted by area
(largest first) and optionally filtered by HSV hue for color-selective picking.

### NanoDet Cross-Check

When using the learned NanoDet detector, detections are cross-checked against
background subtraction:

```
keep = {blob ∈ detector(frame) : ∃ bg_blob with ‖blob.center − bg_blob.center‖ < 45px}
```

This rejects detections of the robot arm itself (always present in the
background photo → no bg-diff).

---

## 13. Carry-Phase Tracking (locate_by_diff)

**Code**: `perception.locate_by_diff()` (line 237)

While carrying an object, the gripper can't blink (that would drop the object).
Instead, bg-subtraction in a local ROI tracks the gripper+object:

```
ROI = [center_x ± roi/2, center_y ± roi/2]
mask = diff_mask(frame[ROI], background[ROI])
blobs = connected_components(mask, min_area)
best = argmin_b ‖b.center − ROI_center‖     (nearest blob to prediction)
```

**Jump gate**: if `‖best.center − ROI_center‖ > max_jump`, return None
(a distractor blob, not the carried object — the object's per-step motion is
bounded by the joint-step clamp).

**Advantages over template matching**: invariant to appearance changes during
long transports (rotation, viewing angle). Only asks "did this pixel change
from the empty background", not what the object looks like.

---

## 14. Cross-Validated Reprojection Error Gate

**Code**: `fused_tracking.FusedTracker.fit()` (line 131)

### Accept/Reject Decision

The fused tracker only activates if its projection generalizes to unseen poses:

```
Train on even-indexed pairs: P_even, d_even = fit(X[::2], px[::2])
Validate on odd-indexed pairs:  rms_val = reproj_rms(P_even, X[1::2] + d_even, px[1::2])

Accept if rms_val ≤ max_rms (10.0 px)
```

**Why cross-validation, not in-sample RMS**: DLT with 6+ points can achieve
sub-pixel in-sample RMS on a degenerate configuration (e.g., near-coplanar
points) while producing wildly wrong projections at new poses. The held-out
set tests exactly what matters — generalization to poses the arm will visit
during the episode.

### Reprojection RMS

```
e_i = project(P, Xᵢ) − px_i
rms = √(mean(‖eᵢ‖²))
```

### Degeneracy Guard

```
SVD(X − mean(X)) = UΣV'
reject if σ₃ / √N < 0.002   (thinnest axis < 2mm std → coplanar)
```

The DLT becomes numerically degenerate when the 3D calibration points lie
near a plane (the projection from 3D to 2D is inherently ambiguous for
in-plane points). Babble that never varied height produces exactly this
configuration.

---

## 15. Episode Pipeline Summary

The full pick-and-place episode, with equations annotated:

```
1. DETECT
   obj_px = bg_sub(frame, background) or nanodet(frame)
   pad_px = hsv_threshold(frame, pad_hue)

2. BABBLE (if no reusable J or fused tracker)
   {dq_k, ds_k} ← random probes + blink_locate
   Ĵ = (QᵀQ + εI)⁻¹ QᵀS                              [Eq. §1]
   If fuse=True: fit P, d from (FK(q), blink_px) pairs  [Eq. §7, §8]

3. SERVO_XY (approach height)
   repeat:
     e = obj_px − s_ee                                   [Eq. §3]
     dq = J⁺_μ · (λ · e)                                [Eq. §2, §3]
     shape_dq = step_dz(q, 0.6·(z_ref − z))             [Eq. §6]
     execute dq + shape_dq
     measure (blink or fused track_px)                   [Eq. §9, §10]
     J ← broyden_update(J, dq, ds) [or fused.jac(q)]   [Eq. §4, §9]
   until ‖e‖ < tol_coarse

4. DESCEND (interleaved)
   repeat:
     dq_z = step_dz(q, −approach_dz)                    [Eq. §6]
     execute dq_z
     SERVO_XY with tol_fine                              [Eq. §3]
     re-anchor via blink (grasp-frame corrected)         [Eq. §11]
   until z ≤ grasp_z

5. GRASP
   set_gripper(0.0)
   check gripper_contact (load + stall on real hw)

6. LIFT
   dq_z = step_dz(q, lift_z − z)                        [Eq. §6]
   track via locate_by_diff                              [Eq. §13]

7. TRANSPORT
   SERVO_XY toward pad_px at lift height                 [Eq. §3]
   locate via bg-diff + (optional) fused model           [Eq. §9, §13]

8. LOWER + RE-SERVO
   descend to release_z, servo to pad_px                 [Eq. §3, §6]

9. RELEASE
   set_gripper(1.0)

10. HOME
    interpolate q → home_q in 5 steps
```

---

## 16. AruCo Marker Localization & Tracking

### Mathematical Model

Instead of active sync-blink wiggling or color-sensitive HSV segmentation, standard automation platforms (e.g. NVIDIA, Boston Dynamics, FANUC) use square binary fiducial markers called **AruCo markers** or **AprilTags** for robust hand-eye tracking.

An AruCo marker of size $L$ has 4 outer corners $p_1, p_2, p_3, p_4$ in the marker frame:
$$
P_{marker} = \left\{\left[-\frac{L}{2}, \frac{L}{2}, 0\right]^\top, \left[\frac{L}{2}, \frac{L}{2}, 0\right]^\top, \left[\frac{L}{2}, -\frac{L}{2}, 0\right]^\top, \left[-\frac{L}{2}, -\frac{L}{2}, 0\right]^\top\right\}
$$

When projected onto the camera frame, the corners are identified at pixel coordinates $u_i = (x_i, y_i)^\top$ via local adaptive thresholding and polygonal contour retrieval:
1. **Adaptive Thresholding**: Converts the BGR image to grayscale and applies local thresholding to find candidate contours.
2. **Contour Extraction**: Identifies 4-corner polygons.
3. **Template Decoding**: Decodes the internal black/white grid code against a predefined dictionary (e.g. `DICT_4X4_50`).
4. **2D Centroid Extraction**: The image-plane tracking coordinate $s$ is the arithmetic mean of the projected corners:
$$
s = \frac{1}{4} \sum_{i=1}^4 u_i \quad \in \mathbb{R}^2
$$

This extraction is invariant to global lighting shifts and does not require active wiggling of the gripper to calibrate offsets.

---

## 17. Parametric Trajectory Kinematics (Shapes)

To draw geometric patterns in the space (like a child drawing with their hand in the air), we generate parametric path coordinates $p(t) = [x(t), y(t), z(t)]^\top$ in a vertical plane that directly faces the viewer (perpendicular to the pointing direction of the arm) and solve the Joint Position Inverse Kinematics at each step.

### 1. Sagittal Vertical Drawing Plane
The drawing is centered directly on the robot's default home posture $q_{home}$. At this configuration, the end effector is located at $p_c \in \mathbb{R}^3$. Let $\theta = q_{home}[0]$ be the shoulder pan angle (which is $0^\circ$ for forward pointing). We define a vertical drawing plane at $p_c$ with orthonormal basis vectors:
- **Horizontal basis vector** (pointing right/left relative to the arm): $v = [-\sin(\theta), \cos(\theta), 0]^\top$
- **Vertical basis vector** (pointing straight up): $w = [0, 0, 1]^\top$

### 2. Circle Trajectory
To trace a circle of radius $R$ in this vertical space:
$$
p(t) = p_c + (R \cos(t)) \cdot v + (R \sin(t)) \cdot w \qquad t \in [0, 2\pi]
$$

### 3. Heart Trajectory (Cardioid Variant)
To trace a symmetric heart of scales $S_h, S_v$ in this vertical space:
$$
\begin{aligned}
h(t) &= S_h \cdot \sin^3(t) \\
v(t) &= S_v \cdot \frac{13 \cos(t) - 5 \cos(2t) - 2 \cos(3t) - \cos(4t) + 2.5}{15.0} \\
p(t) &= p_c + h(t) \cdot v + v(t) \cdot w \qquad t \in [0, 2\pi]
\end{aligned}
$$

### 4. Position-Only Inverse Kinematics Mapping
Since the SO-101 has only 5 Degrees of Freedom, it cannot simultaneously track a 3D position and a 3D orientation. Trying to enforce orientation constraints near workspace boundaries locks joints and causes solver divergence. To allow the arm to trace the vertical shape cleanly, we set the orientation weight to zero. The joint target $q_{next}$ is solved using Placo:
$$
q_{next} = \text{IK}(q_{curr}, T_{target})
$$
where $T_{target}$ is the 4x4 target transformation matrix containing the position $p(t)$, and the orientation task has a weight of $0.0$:
$$
T_{target} = \begin{bmatrix} 
R_{home} & p(t) \\ 
0 & 1 
\end{bmatrix}, \qquad w_{position} = 1.0, \quad w_{orientation} = 0.0
$$
This allows the wrist and hand joints to adjust naturally to trace the vertical trajectory without hitting physical joint limits.

---

## 18. Recursive Base-Offset Estimation (EKF)

**Code**: `fused_tracking.AnalyticalFusedTracker.update()`

The analytical fused tracker's robot→workspace translation `t ∈ ℝ³` starts
from a ONE-SHOT wrist-marker calibration. Every real marker measurement the
episode takes afterwards (approach anchors, ArUco carry tracking) is a fresh
`(q, pixel)` pair that refines `t` recursively with a 3-state EKF:

```
Measurement model:  h(t) = project(K, dist, R_ws, t_ws, FK(q) + t)   ∈ ℝ²
Innovation:         ν = pixel_measured − h(t̂)
Gate:               reject if ‖ν‖ > gate_px (80) — marker misdetection

Linearization:      H = ∂h/∂t  ∈ ℝ^(2×3)   (forward differences, ε = 10⁻⁴ m)
Predict:            P ← P + Q               (random-walk offset, Q = drift_std² I)
Update:             S = H P Hᵀ + R          (R = meas_px_std² I₂)
                    K = P Hᵀ S⁻¹
                    t̂ ← t̂ + K ν
                    P ← (I − K H) P
```

**Why EKF, not plain RLS**: the pixel is a *nonlinear* function of `t`
(perspective division + lens distortion), so each update linearizes `h` at
the current estimate; RLS assumes a linear regressor.

**Division of labor with the anchor residual `c` (§9)**: the EKF absorbs the
constant + slowly varying 3D part of the offset error (its random-walk `Q`
lets it track pose-dependent FK/mount bias); `c` remains the fast per-pose
image-space correction, recomputed against the *updated* model at each anchor.
`anchor_fine` measurements (jaw-closure point — a different physical point
than the marker) deliberately do **not** feed the filter.

A single measurement observes only 2 of `t`'s 3 dimensions (a pixel is 2D);
diversity of poses across the workspace makes all three observable, exactly
as in the DLT argument of §14. Measured on the synthetic-camera test:
44 mm initial offset error → 1.1 mm after 40 updates with 2 px measurement
noise (58 px → 1.6 px prediction error).

---

## 19. Null-Space Secondary Objectives

**Code**: `control.make_null_fn()`, applied in `control.servo_to()`

The image task uses 2 DOF of the 5-DOF arm; the remaining 3-dimensional
null space of `J ∈ ℝ^(2×5)` is spent on two secondary objectives without
disturbing the pixel target to first order:

```
dq = J⁺_μ (λ e) + (I − J⁺_μ J) · dq_null
dq_null = dq_lim + dq_vis
```

### Joint-limit avoidance (potential field)

With `u = (q − q_mid) / q_half ∈ [−1, 1]` (normalized position within the
URDF limits from `RealConfig.q_min/max_deg`) and deadzone `dz = 0.75`:

```
U(q)   = Σᵢ max(0, |uᵢ| − dz)² / (1 − dz)²
dq_lim = −k_lim · ∇U = −k_lim · sign(u) · ((|u| − dz)⁺ / (1 − dz))²
```

Zero inside the deadzone — mid-range motion is never biased; repulsion grows
quadratically toward each limit, reaching `k_lim` (0.04 rad/step) at the
hard limit.

### Marker visibility (wrist posture attraction)

`q_vis` is the joint pose at the **last real marker sighting** (updated at
every anchor / servo measurement / carry measurement that actually saw the
marker). The wrist joints (flex, roll) are pulled back toward it:

```
dq_vis[j] = k_vis · (q_vis[j] − q[j])     for j ∈ {wrist_flex, wrist_roll}
```

Because the term rides the null-space projector, the wrist re-orients toward
a camera-facing pose while the EE pixel stays on target — replacing the
reactive ~30 s `scan_wrist_for_marker` sweep with continuous prevention.
(The projector preserves the *pixel*, a 2-constraint task; residual height
drift is already handled by the parallel `z_hold` term of §3.)

---

## References

- **Hartley & Zisserman**, *Multiple View Geometry in Computer Vision*, Ch. 7
  (DLT, camera resectioning)
- **Broyden, C.G.** (1965), "A class of methods for solving nonlinear
  simultaneous equations", *Mathematics of Computation*
- **Hutchinson, Hager & Corke** (1996), "A tutorial on visual servo control",
  *IEEE Trans. Robotics & Automation* — image Jacobian, IBVS control law
- **Garrido-Jurado et al.** (2014), "Automatic generation and detection of highly reliable fiducial markers under occlusion", *Pattern Recognition* (AruCo design)
- **Kalib** (arXiv:2408.10562) — markerless hand-eye calibration using
  proprioceptive tracking (similar to our fused tracker approach)
- **Fanello et al.** (2018), "Visual-Inertial Estimation for the iCub
  Humanoid Robot", *Frontiers in Robotics and AI*


*Generated for `adaptive_visual_servo` — update this document when equations change.*

"""All tuning knobs for adaptive_visual_servo. One place, like config.env elsewhere."""

from dataclasses import dataclass, field


@dataclass
class SimConfig:
    # camera (true values; the controller never sees these)
    img_w: int = 640
    img_h: int = 480
    focal: float = 650.0
    cam_pos: tuple = (0.05, -0.12, 0.55)
    cam_target: tuple = (0.14, -0.02, 0.0)
    # true arm geometry (meters, radians)
    link_base: float = 0.05   # table to shoulder pivot
    link1: float = 0.11       # shoulder->elbow
    link2: float = 0.14       # elbow->EE point (includes gripper body)
    q_min: tuple = (-1.6, 0.15, -2.4)
    q_max: tuple = (1.6, 1.45, -0.15)
    home_q: tuple = (-1.1, 0.75, -1.35)
    gripper_len: float = 0.02       # finger length, hangs straight down from EE
    finger_half_gap: float = 0.012  # half distance between fingers at g=1 (open)
    # noise + the model error the controller must absorb
    pixel_noise: float = 1.5
    joint_noise: float = 0.0015
    model_error: float = 0.05  # nominal link lengths off by up to +/-5%
    # world
    pad_center: tuple = (0.10, 0.15)
    pad_radius: float = 0.035
    obj_r_range: tuple = (0.010, 0.016)
    obj_height: float = 0.025
    # where random objects may spawn: reach radius + y band (keeps them off pad/home)
    obj_reach: tuple = (0.13, 0.20)
    obj_y: tuple = (-0.08, 0.10)
    # grasp physics
    capture_radius: float = 0.018  # horizontal EE-object distance for a grab
    grasp_ee_z: float = 0.032      # EE height whose fingers reach the object
    grasp_z_tol: float = 0.022
    grip_close: float = 0.25       # g below this while near object => attached
    grip_open: float = 0.6         # g above this => released


@dataclass
class ServoConfig:
    # babbling
    babble_probes: int = 22
    babble_step: float = 0.12  # max |dq| per joint per probe (any DOF count)
    # servo loop
    lam: float = 0.5           # fraction of error corrected per step
    dq_max: float = 0.06       # per-joint per-step clamp (rad; deg on real)
    damping: float = 1e-2      # Tikhonov mu, in (px/rad)^2 units after scaling
    broyden_beta: float = 0.5
    min_dq: float = 0.004      # skip Broyden update below this
    tol_coarse_px: float = 7.0
    tol_fine_px: float = 4.0
    max_steps: int = 120
    diverge_patience: int = 12
    measure_every: int = 1  # steps between real EE measurements (gripper blinks
                            # cost robot motion; >1 dead-reckons on J between
                            # them -- pb/real configs use 3 for ~3x fewer blinks)
    # descend
    approach_dz: float = 0.02       # descend increment between XY re-servos
    approach_z: float = 0.075       # nominal EE height for the coarse approach
    grasp_z: float = 0.030          # nominal EE height to close the gripper at
    lift_z: float = 0.09
    release_z: float = 0.055
    obj_max_area: int = 3500        # px^2; larger diff blobs are the arm, not objects
    # perception
    blink_dg: tuple = (1.0, 0.75)   # gripper open values toggled to blink
                                    # (g_mid must stay above SimConfig.grip_close)
                                    # smaller sweep measurably reduces blink
                                    # centroid bias on the real URDF gripper
                                    # mesh (pb_sim: ~39px -> ~28px mean, still
                                    # a real bias -- see README calibration note)
    diff_thresh: int = 7            # blink slivers are blur-attenuated. diff_mask
                                    # subtracts the median (noise/exposure pedestal,
                                    # ~3 levels) first, so the floor sits lower than
                                    # the raw-diff era's ~4 -- same margin as the
                                    # old 10 over the old floor
    bg_thresh: int = 28
    min_blob: int = 25
    blink_max_blob: int = 6000      # px^2; a blink diff is just the fingers -- bigger
                                    # blobs are exposure jumps/passers-by, not motion
    blink_null_gap_s: float = 0.0   # spacing of the null (no-command) frame pair.
                                    # Real cameras need ~0.15s so the pair isn't the
                                    # same buffered frame (run_real sets this); sims
                                    # render fresh frames per read, 0 keeps tests fast
    reject_px: float = 28.0         # measurement gate vs. J-predicted EE motion
    max_blind: int = 2              # consecutive steps allowed on prediction only
    obj_min_area: int = 80
    pad_hue: int = 165              # OpenCV hue of the pink drop pad
    hue_tol: int = 14
    track_roi: int = 80              # bg-diff search window during carry (locate_by_diff)
    track_max_jump: float = 25.0     # reject a diff blob this far from the kinematic prediction
    retries: int = 2                # grasp retries per episode
    # encoder+vision fusion (fused_tracking.py): fit the fixed camera
    # projection from babble blinks once, then track the EE by FK projection
    # -- free and instant, no per-step blinking. Falls back to blink mode if
    # the cross-validated fit is worse than fuse_max_rms.
    fuse: bool = True
    fuse_max_rms: float = 10.0      # px, cross-validated reprojection error gate
    # gripper color marker tracking
    gripper_marker_hue: int = None   # OpenCV hue (0-179) of marker tape on gripper. If set, tracks color passively without wiggling.
    gripper_marker_sat_min: int = 60
    gripper_marker_val_min: int = 50
    gripper_marker_area_min: int = 15
    gripper_marker_area_max: int = 5000
    gripper_search_radius: float = 120.0 # max distance (px) allowed between predicted/previous and detected gripper position
    stiction_comp: bool = False
    use_aruco: bool = False
    aruco_id: int = 48
    aruco_wrist_id: int = 49
    # null-space secondary objectives (control.make_null_fn / servo_to):
    # dq = J+(lam*e) + (I - J+J) dq_null. The 2xN image Jacobian on a 5-DOF
    # arm leaves a 3-dim null space -- spent on joint-limit repulsion
    # (potential field) and on holding the wrist at the last pose where the
    # tracking marker was actually seen (so the marker stays camera-facing
    # instead of triggering ~30s wrist scans when it drops out).
    null_k_lim: float = 0.04     # rad/step repulsion at a hard joint limit (0 = off)
    null_k_vis: float = 0.15     # fraction/step of wrist error toward visible pose
    null_deadzone: float = 0.75  # |q-mid| fraction of half-range where repulsion starts
    q_lo: tuple = None           # per-joint soft limits (radians); None = repulsion off
    q_hi: tuple = None           # (real runs wire these from RealConfig.q_min/max_deg)
    wrist_joints: tuple = (3, 4)  # wrist_flex, wrist_roll on the SO-101





@dataclass
class RealConfig:
    """SO-101 hardware mode. UNTESTED until first run on the arm.

    All 6 motors are driven directly now (5 arm joints via the visual
    servo's Jacobian + lerobot's placo IK for descend, gripper separately) --
    earlier this slaved wrist_flex to a formula and fixed wrist_roll, driving
    only 3 joints. Must match lerobot_ik.ARM_JOINTS's order (not imported
    directly to avoid config.py depending on a module that imports lerobot).
    """
    port: str = "/dev/ttyACM0"
    robot_id: str = "avs_follower"
    camera_index: int = 0
    joints: tuple = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    max_step_deg: float = 4.0       # per-command clamp, same spirit as max_relative_target=5
    # settle time scales with how far the command actually moved: most visual-
    # servo steps are tiny (clamped by max_step_deg/dq_max) and don't need a
    # full gripper-sweep's worth of wait. floor = minimum for any move (comms
    # round-trip + a little settle); full_move_s = extra added at the biggest
    # possible single-command move (max_step_deg for arm joints, full open<->
    # closed range for the gripper). Starting guess -- tune against your
    # arm's actual STS3215 speed (this project has never run on real hardware).
    settle_floor_s: float = 0.05
    settle_full_move_s: float = 0.3
    # gripper.pos is lerobot-normalized 0-100 over the range recorded during
    # `lerobot` motor calibration (verified against this machine's
    # avs_follower.json: 2022->3531 ticks, ~133deg of servo travel), so use
    # the WHOLE range. The old placeholders (open=50, closed=5) meant the
    # gripper never opened past HALF -- possibly too narrow to straddle the
    # object during approach -- and "grasp close" from the blink-low position
    # was a barely-visible 18-unit squeeze. lerobot's convention is higher =
    # more open; if YOUR gripper visibly closes when an episode starts
    # (g=1.0 should open it), swap these two values.
    gripper_open_pos: float = 100.0  # gripper joint value at g=1.0
    gripper_closed_pos: float = 0.0
    gripper_settle_full_s: float = 1.2  # full open<->close sweep time. Measured on
                                        # this arm: ~90 normalized-units/s, so 100
                                        # units ~= 1.1s -- at 0.75s it was still only
                                        # at pos 27.8 with a +500 in-motion load
    load_threshold: int = 300       # SIGNED Present_Load indicating the gripper is
                                    # squeezing something (measured: settled on empty
                                    # air = +104..108; in-motion transients hit +-500,
                                    # which is why contact ALSO requires the jaw to
                                    # have stalled short of its commanded position)
    grasp_stall_gap: float = 10.0   # normalized units the jaw must stop short of its
                                    # close command to count as 'object between jaws'
                                    # (empty-air close arrives within ~2 units)
    # soft joint limits (deg), straight from the real URDF's own <limit> tags
    # (assets/SO101/so101_new_calib.urdf) -- authoritative, not guessed.
    q_min_deg: tuple = (-110.0, -100.0, -96.83, -95.0, -157.21)
    q_max_deg: tuple = (110.0, 100.0, 96.83, 95.0, 162.79)
    gripper_marker_hue: int = None
    # ArUco marker calibration
    aruco_ws_marker_size: float = 0.033    # workspace markers (IDs 0-3): 3.3cm side
    aruco_robot_marker_size: float = 0.018  # robot markers (IDs 48,49): 1.8cm side
    workspace_marker_ids: tuple = (0, 1, 2, 3)
    gripper_marker_id: int = 48
    wrist_marker_id: int = 49
    grasp_z: float = 0.018  # default grasp height in meters (1.8cm)
    calibration_file: str = "calibration.json"  # saved intrinsics + extrinsics
    # Default workspace marker 3D positions in workspace frame (meters).
    # Assumes markers at four corners of a rectangular table, origin at marker 0.
    # Override these with your actual measured positions for best accuracy.
    workspace_marker_positions: dict = None  # set in __post_init__

    def __post_init__(self):
        if self.workspace_marker_positions is None:
            # Default: 63cm × 60cm table, markers at corners, Z=0 (table surface)
            # User reports ~10cm max measurement error; calibration absorbs it.
            self.workspace_marker_positions = {
                0: [0.0,  0.0,  0.0],
                1: [0.63, 0.0,  0.0],
                2: [0.63, 0.60, 0.0],
                3: [0.0,  0.60, 0.0],
            }


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
    babble_step: tuple = (0.045, 0.05, 0.05)  # max |dq| per joint per probe
    # servo loop
    lam: float = 0.5           # fraction of error corrected per step
    dq_max: float = 0.06       # per-joint per-step clamp (rad; deg on real)
    damping: float = 1e-2      # Tikhonov mu, in (px/rad)^2 units after scaling
    broyden_beta: float = 0.5
    min_dq: float = 0.004      # skip Broyden update below this
    tol_coarse_px: float = 7.0
    tol_fine_px: float = 4.0
    max_steps: int = 60
    diverge_patience: int = 5
    # descend
    approach_dz: float = 0.02       # descend increment between XY re-servos
    approach_z: float = 0.075       # nominal EE height for the coarse approach
    grasp_z: float = 0.030          # nominal EE height to close the gripper at
    lift_z: float = 0.09
    release_z: float = 0.055
    obj_max_area: int = 3500        # px^2; larger diff blobs are the arm, not objects
    # perception
    blink_dg: tuple = (1.0, 0.4)    # gripper open values toggled to blink
                                    # (g_mid must stay above SimConfig.grip_close)
    diff_thresh: int = 10           # blink slivers are blur-attenuated; noise floor ~4
    bg_thresh: int = 28
    min_blob: int = 25
    reject_px: float = 28.0         # measurement gate vs. J-predicted EE motion
    max_blind: int = 2              # consecutive steps allowed on prediction only
    obj_min_area: int = 80
    pad_hue: int = 165              # OpenCV hue of the pink drop pad
    hue_tol: int = 14
    track_roi: int = 80              # bg-diff search window during carry (locate_by_diff)
    track_max_jump: float = 25.0     # reject a diff blob this far from the kinematic prediction
    retries: int = 2                # grasp retries per episode


@dataclass
class RealConfig:
    """SO-101 hardware mode. UNTESTED until first run on the arm."""
    port: str = "/dev/ttyACM0"
    robot_id: str = "avs_follower"
    camera_index: int = 0
    # our 3 controlled joints -> SO-101 motor names
    joints: tuple = ("shoulder_pan", "shoulder_lift", "elbow_flex")
    # wrist_flex is slaved to keep the gripper pointing down:
    # wrist_flex = wrist_k * (shoulder_lift + elbow_flex) + wrist_offset  (degrees)
    wrist_k: float = -1.0
    wrist_offset: float = 0.0
    wrist_roll: float = 0.0
    max_step_deg: float = 4.0       # per-command clamp, same spirit as max_relative_target=5
    settle_s: float = 0.35
    gripper_open_pos: float = 50.0  # gripper joint value at g=1.0 (calibrate!)
    gripper_closed_pos: float = 5.0
    load_threshold: int = 300       # |Present_Load| indicating contact (calibrate!)
    # soft joint limits (deg) enforced on top of firmware limits
    q_min_deg: tuple = (-90.0, 5.0, -135.0)
    q_max_deg: tuple = (90.0, 85.0, -10.0)
    # nominal link lengths (m) for the open-loop Z descend ONLY -- XY stays
    # visually closed, so like the sim's model_error these are allowed to be
    # wrong. Defaults are the same SO-101-ish guess sim.py's SimConfig uses;
    # measure your actual arm and override here if descend steps feel off.
    link_base: float = 0.05
    link1: float = 0.11
    link2: float = 0.14

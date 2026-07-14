"""FK/descend model backed by lerobot's own SO-101 kinematics (placo), over
the real URDF -- same .ee(q)/.step_dz(q, dz) interface as sim.NominalModel,
so control.py's nominal_z_to/z_hold work unchanged with either backend.
Used for pb_sim.py (5-DOF) and run_real.py; sim.py's toy 3-DOF world keeps
its own NominalModel since that's a different, already-proven kinematic toy,
not the real arm.

Only ever used for the open-loop Z move -- XY stays visually closed via the
babbled/Broyden-updated image Jacobian regardless of how accurate this is.
"""

import os

import numpy as np

from lerobot.model.kinematics import RobotKinematics

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "SO101")
URDF_PATH = os.path.join(ASSETS_DIR, "so101_new_calib.urdf")
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
EE_FRAME = "gripper_frame_link"


class PlacoModel:
    def __init__(self, urdf_path=URDF_PATH, joint_names=ARM_JOINTS, target_frame=EE_FRAME):
        self.kin = RobotKinematics(urdf_path, target_frame_name=target_frame,
                                   joint_names=list(joint_names))

    def ee(self, q):
        T = self.kin.forward_kinematics(np.degrees(q))
        return T[:3, 3]

    def step_dz(self, q, dz):
        """Joint delta moving the EE by (0,0,dz), via lerobot's placo IK
        solver, position-only (orientation unconstrained)."""
        q_deg = np.degrees(q)
        T_target = self.kin.forward_kinematics(q_deg)
        T_target[2, 3] += dz
        q_deg_new = self.kin.inverse_kinematics(q_deg, T_target,
                                                position_weight=1.0, orientation_weight=0.0)
        return np.radians(q_deg_new) - q

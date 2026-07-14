"""PyBullet SO-101 simulation over the real URDF (assets/SO101/).

Same duck-typed rig interface as sim.py's SimWorld (get_q/set_q/set_gripper/
gripper_contact/read/capture_background), so control.py/perception.py are
unchanged -- this is a second, more physically real backend, not a rewrite.
Unlike sim.py's fast headless SimWorld (analytic 3-DOF FK, used for hundreds
of test episodes), this one drives all 5 arm joints through the actual URDF
kinematic chain via pybullet, with a real rendered camera, and --gui shows a
live 3D window so you can watch the arm work.

Grasping: the same proximity-triggered `p.createConstraint` weld as sim.py's
distance-radius heuristic (not real mesh contact/friction) -- general
position-only IK picks an arbitrary wrist orientation, so real contact
detection was unreliable even when ee_world() sat right on top of the object.
# ponytail: proximity weld, not contact/friction-grasp physics -- upgrade to
# orientation-aware IK + real contact if objects need to slip/rotate
# realistically in-hand.
"""

import os

import numpy as np
import pybullet as p
import pybullet_data

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "SO101")
URDF_PATH = os.path.join(ASSETS_DIR, "so101_new_calib.urdf")

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
EE_LINK = "gripper_frame_link"

# camera: fixed eye-to-hand, overhead-ish -- same spirit as sim.py's SimConfig
CAM_POS = (0.05, -0.35, 0.55)
CAM_TARGET = (0.15, 0.0, 0.05)
IMG_W, IMG_H = 640, 480
CAM_FOV = 50.0

BASE_Z = 0.0           # robot base sits on the table plane
PAD_CENTER = (0.20, 0.20)
PAD_RADIUS = 0.045
TABLE_RGBA = (0.80, 0.80, 0.79, 1.0)
GRASP_CAPTURE_RADIUS = 0.035   # horizontal ee_world()-to-object distance for a grab
GRASP_Z_TOL = 0.06              # height tolerance, generous: descend targeting is Task 9


class PyBulletWorld:
    def __init__(self, gui=False, seed=0):
        self.rng = np.random.default_rng(seed)
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(0.6, 50, -35, [0.15, 0.0, 0.05]) if gui else None

        self.table = p.loadURDF("plane.urdf", [0, 0, BASE_Z])
        p.changeVisualShape(self.table, -1, rgbaColor=TABLE_RGBA)
        self.robot = p.loadURDF(URDF_PATH, [0, 0, BASE_Z], useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION)

        self._joint_idx = {}
        self._link_idx = {}
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            self._joint_idx[info[1].decode()] = i
            self._link_idx[info[12].decode()] = i
        self.ee_link = self._link_idx[EE_LINK]
        # anchor for the weld: gripper_link (rigid w.r.t. the arm chain), NOT
        # the moving-jaw link -- that one has its own revolute joint and
        # drifts relative to ee_world() under load if welded to directly.
        self.gripper_link = self._link_idx["gripper_link"]
        self.jaw_link = self._joint_idx[GRIPPER_JOINT]  # moving jaw's own link

        lo, hi = [], []
        for j in ARM_JOINTS:
            info = p.getJointInfo(self.robot, self._joint_idx[j])
            lo.append(info[8]); hi.append(info[9])
        self.q_min, self.q_max = np.array(lo), np.array(hi)
        ginfo = p.getJointInfo(self.robot, self._joint_idx[GRIPPER_JOINT])
        self.g_min, self.g_max = ginfo[8], ginfo[9]

        home = np.array([0.0, -1.2, 1.2, 0.6, 0.0])
        for j, v in zip(ARM_JOINTS, home):
            p.resetJointState(self.robot, self._joint_idx[j], v)
        self._q_cmd = home.copy()
        self.g = 1.0
        p.resetJointState(self.robot, self._joint_idx[GRIPPER_JOINT], self.g_min)

        self.pad = self._make_pad()
        self.objects = []      # list of dict(body, r, attached, constraint)
        self._settle(60)

    # ---------- setup ----------
    def _make_pad(self):
        col = -1
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=PAD_RADIUS, length=0.002,
                                  rgbaColor=(0.95, 0.35, 0.75, 1.0))
        body = p.createMultiBody(0, col, vis, [PAD_CENTER[0], PAD_CENTER[1], 0.001])
        return body

    def spawn_object(self, xy, r=0.013, color=(0.8, 0.15, 0.15, 1.0), shape="box"):
        h = 0.025
        if shape == "cylinder":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=color)
        else:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[r, r, h / 2])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[r, r, h / 2], rgbaColor=color)
        body = p.createMultiBody(0.01, col, vis, [xy[0], xy[1], h / 2 + 0.002])
        p.changeDynamics(body, -1, lateralFriction=1.0)
        obj = dict(body=body, r=r, attached=False, constraint=None)
        self.objects.append(obj)
        return obj

    def spawn_random(self, n=1):
        made = []
        for _ in range(n):
            for _try in range(200):
                reach = self.rng.uniform(0.18, 0.32)
                ang = self.rng.uniform(-0.5, 0.5)
                x, y = reach * np.cos(ang), reach * np.sin(ang)
                if np.hypot(x - PAD_CENTER[0], y - PAD_CENTER[1]) < PAD_RADIUS + 0.04:
                    continue
                if any(np.hypot(x - o["_xy"][0], y - o["_xy"][1]) < 0.05 for o in made):
                    continue
                r = self.rng.uniform(0.010, 0.015)
                hsv_hue = self.rng.uniform(0, 1)
                import colorsys
                rgb = colorsys.hsv_to_rgb(hsv_hue, 0.8, 0.85)
                shape = "cylinder" if self.rng.random() < 0.5 else "box"
                o = self.spawn_object((x, y), r, (*rgb, 1.0), shape)
                o["_xy"] = (x, y)
                made.append(o)
                break
            else:
                raise RuntimeError("could not place object")
        self._settle(30)
        return made

    def _settle(self, n):
        for _ in range(n):
            p.stepSimulation()

    # ---------- rig interface ----------
    def get_q(self):
        return np.array([p.getJointState(self.robot, self._joint_idx[j])[0]
                         for j in ARM_JOINTS])

    def set_q(self, q):
        q = np.clip(q, self.q_min, self.q_max)
        for j, v in zip(ARM_JOINTS, q):
            p.setJointMotorControl2(self.robot, self._joint_idx[j], p.POSITION_CONTROL,
                                    targetPosition=float(v), force=15, maxVelocity=3.0)
        self._settle(24)
        self._q_cmd = self.get_q()
        for o in self.objects:
            if o["attached"]:
                pass  # constraint keeps it welded; physics moves it automatically

    def set_gripper(self, g):
        self.g = float(np.clip(g, 0.0, 1.0))
        # g=1 (open) -> g_min, g=0 (closed) -> g_max: verified empirically,
        # the URDF's joint-limit sign is the opposite of "higher = more open"
        target = self.g_max - self.g * (self.g_max - self.g_min)
        p.setJointMotorControl2(self.robot, self._joint_idx[GRIPPER_JOINT],
                                p.POSITION_CONTROL, targetPosition=target,
                                force=15, maxVelocity=8.0)
        self._settle(60)  # full open<->close sweep is ~1.9rad, needs real settle time
        if self.g < 0.35:
            self._try_grasp()
        elif self.g > 0.85:
            self._release_all()

    def _try_grasp(self):
        # proximity-triggered weld, not mesh contact: general position-only IK
        # (used by callers/tests to reach a target) picks an arbitrary wrist
        # orientation, so the jaws often don't face the object even when
        # ee_world() is right on top of it -- same proven approach as
        # sim.py's SimWorld (distance + height gate), just measured against
        # the real URDF's gripper_frame_link.
        # ponytail: proximity weld, not friction-grasp physics -- upgrade to
        # real contact + orientation-aware IK if objects need to slip/rotate
        # realistically in-hand.
        ee = self.ee_world()
        for o in self.objects:
            if o["attached"]:
                continue
            pos, orn = p.getBasePositionAndOrientation(o["body"])
            close_xy = np.hypot(pos[0] - ee[0], pos[1] - ee[1]) < GRASP_CAPTURE_RADIUS
            close_z = abs(pos[2] - ee[2]) < GRASP_Z_TOL
            if close_xy and close_z:
                link_state = p.getLinkState(self.robot, self.gripper_link)
                inv_pos, inv_orn = p.invertTransform(link_state[4], link_state[5])
                rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, pos, orn)
                cid = p.createConstraint(self.robot, self.gripper_link, o["body"], -1,
                                         p.JOINT_FIXED, [0, 0, 0], rel_pos, [0, 0, 0],
                                         childFrameOrientation=rel_orn)
                p.changeConstraint(cid, maxForce=200)
                o["attached"], o["constraint"] = True, cid

    def _release_all(self):
        for o in self.objects:
            if o["attached"]:
                p.removeConstraint(o["constraint"])
                o["attached"], o["constraint"] = False, None

    def gripper_contact(self):
        return any(o["attached"] for o in self.objects)

    def read(self):
        return self._render()

    def capture_background(self):
        return self._render()

    # ---------- ground truth (tests/scoring only) ----------
    def ee_world(self):
        return np.array(p.getLinkState(self.robot, self.ee_link)[4])

    def grip_px(self):
        return self._project(self.ee_world())

    def object_xy(self, o):
        pos, _ = p.getBasePositionAndOrientation(o["body"])
        return np.array(pos[:2])

    # ---------- rendering ----------
    def _view_proj(self):
        view = p.computeViewMatrix(CAM_POS, CAM_TARGET, [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(CAM_FOV, IMG_W / IMG_H, 0.05, 3.0)
        return view, proj

    def _render(self):
        view, proj = self._view_proj()
        _, _, rgba, _, _ = p.getCameraImage(
            IMG_W, IMG_H, view, proj, renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK)
        rgb = np.reshape(rgba, (IMG_H, IMG_W, 4))[:, :, :3].astype(np.uint8)
        return rgb[:, :, ::-1].copy()  # RGB -> BGR

    def _project(self, world_xyz):
        view, proj = self._view_proj()
        view = np.array(view).reshape(4, 4, order="F")
        proj = np.array(proj).reshape(4, 4, order="F")
        clip = proj @ view @ np.array([*world_xyz, 1.0])
        ndc = clip[:3] / clip[3]
        u = (ndc[0] * 0.5 + 0.5) * IMG_W
        v = (1 - (ndc[1] * 0.5 + 0.5)) * IMG_H
        return np.array([u, v])

    def close(self):
        p.disconnect(self.client)

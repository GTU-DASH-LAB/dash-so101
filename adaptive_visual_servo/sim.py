"""Kinematic SO-101-ish simulator with a synthetic pinhole camera.

The sim runs on TRUE geometry (link lengths, camera pose) and adds pixel +
joint noise. The controller only ever sees rendered frames and its own
commanded joint values; `fk`/`project`/`grip_center` ground truth is for tests
and scoring only. `NominalModel` is the deliberately-wrong model the controller
is allowed to use for open-loop Z moves.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field

from config import SimConfig

UP = np.array([0.0, 0.0, 1.0])


def _norm(v):
    return v / np.linalg.norm(v)


class Camera:
    def __init__(self, cfg: SimConfig):
        self.C = np.array(cfg.cam_pos, float)
        fwd = _norm(np.array(cfg.cam_target, float) - self.C)
        right = _norm(np.cross(fwd, UP))
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd])  # world -> camera rows
        self.f = cfg.focal
        self.cx, self.cy = cfg.img_w / 2.0, cfg.img_h / 2.0

    def project(self, p):
        """World point(s) (...,3) -> pixel (u, v) float array."""
        pc = (np.atleast_2d(p) - self.C) @ self.R.T
        z = np.maximum(pc[:, 2], 1e-6)
        uv = np.stack([self.f * pc[:, 0] / z + self.cx,
                       self.f * pc[:, 1] / z + self.cy], axis=1)
        return uv[0] if np.ndim(p) == 1 else uv

    def px_per_m(self, p):
        """Approximate image scale at world point p."""
        z = max(((p - self.C) @ self.R.T)[2], 1e-6)
        return self.f / z


def fk_points(q, link_base, link1, link2):
    """Base/shoulder/elbow/EE world positions for pan/shoulder/elbow angles."""
    q0, q1, q2 = q
    c0, s0 = np.cos(q0), np.sin(q0)
    shoulder = np.array([0.0, 0.0, link_base])
    r1, z1 = link1 * np.cos(q1), link1 * np.sin(q1)
    r2, z2 = link2 * np.cos(q1 + q2), link2 * np.sin(q1 + q2)
    elbow = shoulder + np.array([r1 * c0, r1 * s0, z1])
    ee = elbow + np.array([r2 * c0, r2 * s0, z2])
    return np.array([0.0, 0.0, 0.0]), shoulder, elbow, ee


class NominalModel:
    """The controller's imperfect kinematic model (link lengths off by ~model_error).

    Used ONLY for open-loop Z moves (descend/lift). XY is closed visually.
    """

    def __init__(self, cfg: SimConfig, rng):
        e = cfg.model_error
        self.lb = cfg.link_base * (1 + rng.uniform(-e, e))
        self.l1 = cfg.link1 * (1 + rng.uniform(-e, e))
        self.l2 = cfg.link2 * (1 + rng.uniform(-e, e))

    def ee(self, q):
        return fk_points(q, self.lb, self.l1, self.l2)[3]

    def jac(self, q, eps=1e-5):
        """3x3 position Jacobian d(xyz)/dq by central differences."""
        J = np.zeros((3, 3))
        for i in range(3):
            d = np.zeros(3); d[i] = eps
            J[:, i] = (self.ee(q + d) - self.ee(q - d)) / (2 * eps)
        return J

    def step_dz(self, q, dz, mu=1e-6):
        """Joint step moving the EE by (0,0,dz) with damped least squares."""
        J = self.jac(q)
        target = np.array([0.0, 0.0, dz])
        return np.linalg.solve(J.T @ J + mu * np.eye(3), J.T @ target)


@dataclass
class Obj:
    pos: np.ndarray          # 3D center (z = height/2 when on table)
    r: float
    color: tuple             # BGR
    rim: np.ndarray          # radius multipliers for the outline polygon
    theta: float = 0.0
    attached: bool = False


def _shape_rim(shape, rng):
    n = 20
    if shape == "circle":
        return np.ones(n)
    if shape == "square":
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return 1.0 / np.maximum(np.abs(np.cos(ang)), np.abs(np.sin(ang))) / np.sqrt(2) * 1.3
    if shape == "triangle":
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return 0.9 / np.cos((ang % (2 * np.pi / 3)) - np.pi / 3) * 0.9
    return rng.uniform(0.65, 1.25, n)  # random blob


SHAPES = ("circle", "square", "triangle", "blob")


class SimWorld:
    """Robot + camera + objects. Controller-facing API: set_q/get_q/set_gripper/
    gripper_contact/read/capture_background. Everything else is ground truth."""

    def __init__(self, cfg: SimConfig = None, seed: int = 0):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(seed)
        self.cam = Camera(self.cfg)
        self._q_cmd = np.array(self.cfg.home_q, float)
        self._q_true = self._q_cmd.copy()
        self.g = 1.0
        self.objects: list[Obj] = []
        self.pad = np.array([*self.cfg.pad_center, 0.0])
        self._pad_color = tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[157, 190, 235]]]), cv2.COLOR_HSV2BGR)[0, 0])
        self._table = self._make_table()

    # ---------- ground truth (tests/scoring only) ----------
    def fk(self, q=None):
        q = self._q_true if q is None else q
        return fk_points(q, self.cfg.link_base, self.cfg.link1, self.cfg.link2)

    def ee(self, q=None):
        return self.fk(q)[3]

    def grip_center(self, q=None):
        """Midpoint of the fingers — the physical 'tool center point'."""
        return self.ee(q) + np.array([0.0, 0.0, -self.cfg.gripper_len / 2])

    def grip_px(self):
        return self.cam.project(self.grip_center())

    # ---------- controller-facing ----------
    def get_q(self):
        return self._q_cmd.copy()

    def set_q(self, q):
        self._q_cmd = np.clip(q, self.cfg.q_min, self.cfg.q_max)
        self._q_true = self._q_cmd + self.rng.normal(0, self.cfg.joint_noise, 3)
        ee = self.ee()
        for o in self.objects:
            if o.attached:
                o.pos = np.array([ee[0], ee[1],
                                  max(ee[2] - self.cfg.gripper_len, self.cfg.obj_height / 2)])

    def set_gripper(self, g):
        self.g = float(np.clip(g, 0.0, 1.0))
        c = self.cfg
        ee = self.ee()
        if self.g < c.grip_close and not self.gripper_contact():
            for o in self.objects:
                near_xy = np.hypot(ee[0] - o.pos[0], ee[1] - o.pos[1]) < c.capture_radius
                near_z = abs(ee[2] - c.grasp_ee_z) < c.grasp_z_tol
                if near_xy and near_z:
                    o.attached = True
                    break
        elif self.g > c.grip_open:
            for o in self.objects:
                if o.attached:
                    o.attached = False
                    o.pos = np.array([o.pos[0], o.pos[1], c.obj_height / 2])

    def gripper_contact(self):
        return any(o.attached for o in self.objects)

    def read(self):
        return self.render(noise=True)

    def capture_background(self):
        """One-time 'empty workspace' photo: arm parked, pad in place, no objects."""
        return self.render(draw_objects=False, noise=True)

    # ---------- world setup ----------
    def spawn_object(self, xy, r=0.013, shape="circle", color=(40, 40, 200), theta=0.0):
        rim = _shape_rim(shape, self.rng)
        o = Obj(np.array([xy[0], xy[1], self.cfg.obj_height / 2]), r, color, rim, theta)
        self.objects.append(o)
        return o

    def spawn_random(self, n=1):
        c = self.cfg
        home_xy = self.ee(np.array(c.home_q))[:2]
        made = []
        for _ in range(n):
            for _try in range(200):
                reach = self.rng.uniform(*c.obj_reach)
                ang = self.rng.uniform(-0.55, 0.55)
                x, y = reach * np.cos(ang), reach * np.sin(ang)
                if not (c.obj_y[0] <= y <= c.obj_y[1]):
                    continue
                r = self.rng.uniform(*c.obj_r_range)
                if np.hypot(x - self.pad[0], y - self.pad[1]) < c.pad_radius + r + 0.02:
                    continue
                if np.hypot(x - home_xy[0], y - home_xy[1]) < 0.06:
                    continue
                if any(np.hypot(x - o.pos[0], y - o.pos[1]) < r + o.r + 0.03
                       for o in self.objects):
                    continue
                if self.rng.random() < 0.25:  # colorless object: bg-sub must still see it
                    v = int(self.rng.integers(55, 120))
                    color = (v, v, v)
                else:
                    hsv = np.uint8([[[self.rng.integers(0, 180), 200, 205]]])
                    color = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
                shape = SHAPES[self.rng.integers(0, len(SHAPES))]
                made.append(self.spawn_object((x, y), r, shape, color,
                                              self.rng.uniform(0, np.pi)))
                break
            else:
                raise RuntimeError("could not place object")
        return made

    # ---------- rendering ----------
    def _make_table(self):
        c = self.cfg
        img = np.full((c.img_h, c.img_w, 3), 205, np.uint8)
        grad = np.linspace(-8, 8, c.img_w, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + grad[None, :, None], 0, 255).astype(np.uint8)
        rng = np.random.default_rng(12345)  # static texture, same every frame
        for _ in range(350):
            x, y = int(rng.integers(0, c.img_w)), int(rng.integers(0, c.img_h))
            shade = int(rng.integers(175, 225))
            cv2.circle(img, (x, y), int(rng.integers(1, 3)), (shade,) * 3, -1)
        return img

    def _poly_px(self, o: Obj):
        ang = np.linspace(0, 2 * np.pi, len(o.rim), endpoint=False) + o.theta
        pts = np.stack([o.pos[0] + o.r * o.rim * np.cos(ang),
                        o.pos[1] + o.r * o.rim * np.sin(ang),
                        np.full(len(o.rim), o.pos[2])], axis=1)
        return self.cam.project(pts).astype(np.int32)

    def render(self, draw_objects=True, noise=False):
        c = self.cfg
        img = self._table.copy()
        # drop pad
        pad_r_px = int(c.pad_radius * self.cam.px_per_m(self.pad))
        pu, pv = self.cam.project(self.pad)
        cv2.circle(img, (int(pu), int(pv)), pad_r_px, self._pad_color, -1)
        # loose objects
        if draw_objects:
            for o in self.objects:
                if not o.attached:
                    cv2.fillPoly(img, [self._poly_px(o)], o.color)
        # arm
        base, shoulder, elbow, ee = self.fk()
        px = self.cam.project(np.stack([base, shoulder, elbow, ee])).astype(int)
        cv2.line(img, tuple(px[0]), tuple(px[1]), (70, 70, 75), 10, cv2.LINE_AA)
        cv2.line(img, tuple(px[1]), tuple(px[2]), (85, 80, 75), 9, cv2.LINE_AA)
        cv2.line(img, tuple(px[2]), tuple(px[3]), (95, 90, 85), 7, cv2.LINE_AA)
        for p in px[1:3]:
            cv2.circle(img, tuple(p), 5, (60, 60, 65), -1)
        # fingers: hang straight down, offset perpendicular to the arm plane
        q0 = self._q_true[0]
        d = np.array([-np.sin(q0), np.cos(q0), 0.0])
        half_gap = 0.003 + (c.finger_half_gap - 0.003) * self.g
        for s in (-1, 1):
            top = ee + s * half_gap * d
            bot = top + np.array([0.0, 0.0, -c.gripper_len])
            t, b = self.cam.project(np.stack([top, bot])).astype(int)
            cv2.line(img, tuple(t), tuple(b), (45, 45, 50), 3, cv2.LINE_AA)
        # carried object drawn last (held at the fingers)
        if draw_objects:
            for o in self.objects:
                if o.attached:
                    cv2.fillPoly(img, [self._poly_px(o)], o.color)
        if noise and c.pixel_noise > 0:
            img = np.clip(img.astype(np.float32) +
                          self.rng.normal(0, c.pixel_noise, img.shape),
                          0, 255).astype(np.uint8)
        return img


def solve_ik_true(world: SimWorld, target_xyz, q0=None, iters=80):
    """Ground-truth damped-LS IK — test/setup helper ONLY, never used by control."""
    q = np.array(q0 if q0 is not None else world.get_q(), float)
    c = world.cfg
    for _ in range(iters):
        err = np.asarray(target_xyz) - world.ee(q)
        if np.linalg.norm(err) < 1e-4:
            break
        J = np.zeros((3, 3))
        for i in range(3):
            d = np.zeros(3); d[i] = 1e-5
            J[:, i] = (world.ee(q + d) - world.ee(q - d)) / 2e-5
        dq = np.linalg.solve(J.T @ J + 1e-6 * np.eye(3), J.T @ err)
        q = np.clip(q + np.clip(dq, -0.2, 0.2), c.q_min, c.q_max)
    return q

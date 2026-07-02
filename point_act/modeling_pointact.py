import json
import logging
import re
from pathlib import Path

import torch
from torch import Tensor

from wm_act.modeling_wmact import WMACTPolicy

from .configuration_pointact import PointACTConfig

# Marker colors in *normalized* image space (images are MEAN_STD-normalized before
# they reach the policy, so we draw values ~2 std out — unmistakable to the network,
# consistent between training and inference by construction).
PICK_COLOR = (-2.0, 2.0, -2.0)  # "green"
PLACE_COLOR = (-2.0, -2.0, 2.0)  # "blue"

TASK_POINT_RE = re.compile(r"(pick|place)@([01]?\.\d+),\s*([01]?\.\d+)")


def parse_task_points(task: str) -> dict[str, tuple[float, float]]:
    """Parse "pick@0.43,0.61 place@0.72,0.35" -> {"pick": (u, v), "place": (u, v)}."""
    return {m[0]: (float(m[1]), float(m[2])) for m in TASK_POINT_RE.findall(task)}


class PointACTPolicy(WMACTPolicy):
    """WM-ACT conditioned on pick/place points via visual prompting.

    The points arrive as normalized (u, v) image coordinates and are drawn onto the
    camera image as solid square markers (green = pick, blue = place) before the
    backbone sees it. The policy therefore learns "grasp what is under the green
    marker, place it at the blue marker" — object identity is irrelevant to it,
    which is what lets frozen open-vocabulary models upstream provide zero-shot
    generalization to objects never seen in the demos.

    Point sources, in order of precedence, per batch:
      1. batch["pick_point"] / batch["place_point"]: (B, 2) float tensors in [0, 1]
      2. batch["task"] strings of the form "pick@0.43,0.61 place@0.72,0.35"
      3. the episode_index -> points JSON sidecar (training; see so_brain/relabel.py)
    """

    config_class = PointACTConfig
    name = "point_act"

    def __init__(self, config: PointACTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._episode_points: dict[str, dict] = {}
        if config.point_labels_path:
            path = Path(config.point_labels_path)
            if path.exists():
                self._episode_points = json.loads(path.read_text())
            else:
                logging.warning(
                    f"point_labels_path '{path}' not found — training batches without explicit "
                    "points will get no markers. (Fine at inference: points are supplied directly.)"
                )

    def _lookup_points(self, batch: dict, b: int, kind: str) -> tuple[float, float] | None:
        key = f"{kind}_point"
        if key in batch:
            u, v = batch[key][b].tolist()
            return float(u), float(v)
        task = batch.get("task")
        if task is not None:
            task_str = task[b] if isinstance(task, (list, tuple)) else task
            pts = parse_task_points(str(task_str))
            if kind in pts:
                return pts[kind]
        if "episode_index" in batch and self._episode_points:
            ep = self._episode_points.get(str(int(batch["episode_index"][b])))
            if ep and kind in ep:
                return tuple(ep[kind])
        return None

    def _draw_markers(self, batch: dict) -> dict:
        """Return a copy of the batch with markers drawn onto every image key."""
        batch = dict(batch)
        first_img = batch[next(iter(self.config.image_features))]
        bsize = first_img.shape[0]
        points = [
            {kind: self._lookup_points(batch, b, kind) for kind in ("pick", "place")}
            for b in range(bsize)
        ]
        if not any(p["pick"] or p["place"] for p in points):
            return batch

        r = self.config.marker_radius
        for key in self.config.image_features:
            if key not in batch:
                continue
            img = batch[key].clone()  # (B, C, H, W) or (B, T, C, H, W) — draw on all frames
            h, w = img.shape[-2:]
            for b in range(bsize):
                for kind, color in (("pick", PICK_COLOR), ("place", PLACE_COLOR)):
                    pt = points[b][kind]
                    if pt is None:
                        continue
                    px, py = int(pt[0] * (w - 1)), int(pt[1] * (h - 1))
                    y0, y1 = max(py - r, 0), min(py + r + 1, h)
                    x0, x1 = max(px - r, 0), min(px + r + 1, w)
                    for c, value in enumerate(color):
                        img[b, ..., c, y0:y1, x0:x1] = value
            batch[key] = img
        return batch

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return super().forward(self._draw_markers(batch))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        return super().predict_action_chunk(self._draw_markers(batch))

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from wm_act.configuration_wmact import WMACTConfig


@PreTrainedConfig.register_subclass("point_act")
@dataclass
class PointACTConfig(WMACTConfig):
    """WM-ACT conditioned on pick/place image points drawn as visual markers.

    Extra args:
        point_labels_path: JSON sidecar mapping episode_index -> {"pick": [u, v],
            "place": [u, v]} (normalized [0,1] image coords), produced by
            so_brain/relabel.py. Used at training time to draw the markers.
            Missing file is tolerated (inference supplies points directly).
        marker_radius: Marker half-size in pixels (at the training resolution).
    """

    point_labels_path: str | None = None
    marker_radius: int = 6

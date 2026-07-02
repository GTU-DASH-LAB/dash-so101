"""Point-ACT: point-conditioned WM-ACT, the trained "hands" of the zero-shot stack.

The policy is conditioned on two image points — pick (green marker) and place
(blue marker) — burned into the camera image as visual prompts. Object identity
never enters the policy, so open-vocabulary generalization comes entirely from
the frozen upstream models (VLM planner + OWLv2 detector) that supply the points.

Registers policy type "point_act" (LeRobot plugin, like wm_act).
"""

from .configuration_pointact import PointACTConfig
from .modeling_pointact import PointACTPolicy

__all__ = ["PointACTConfig", "PointACTPolicy"]

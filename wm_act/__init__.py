"""WM-ACT: world-model-augmented action chunking policy, packaged as a LeRobot plugin.

Importing this package registers the "wm_act" policy type with LeRobot's config
registry, so `lerobot-train --policy.discover_packages_path=wm_act --policy.type=wm_act`
and `lerobot-rollout --policy.discover_packages_path=wm_act --policy.path=...` work
without touching the lerobot submodule.
"""

from .configuration_wmact import WMACTConfig
from .modeling_wmact import WMACTPolicy

__all__ = ["WMACTConfig", "WMACTPolicy"]

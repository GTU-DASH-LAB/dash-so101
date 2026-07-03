"""pi-WM: pi0.5 + world-model best-of-N action selection, as a LeRobot plugin.

The base VLA (lerobot/pi05_libero_finetuned, frozen) samples N action chunks from
its flow model; a small world-model scorer predicts each chunk's visual future and
its task progress; the chunk predicted to make the most progress gets executed.
Test-time scaling for VLAs (V-GPS / RoboMonkey lineage) — the target is beating
pi0.5's published 97.5% LIBERO average under the identical eval protocol.

Registers policy type "pi_wm".
"""

from .configuration_piwm import PiWMConfig
from .modeling_piwm import PiWMPolicy

__all__ = ["PiWMConfig", "PiWMPolicy"]

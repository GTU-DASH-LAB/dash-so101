from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi_wm")
@dataclass
class PiWMConfig(PI05Config):
    """pi0.5 + world-model best-of-N. Weights-compatible with any pi05 checkpoint
    (the scorer lives in its own file, not in the safetensors).

    Extra args:
        n_samples: Candidate action chunks sampled per decision. 1 = plain pi0.5
            (the baseline to compare against). 8 is a good starting point.
        scorer_path: torch checkpoint produced by pi_wm/train_scorer.py. Empty
            disables scoring (falls back to plain pi0.5).
        scorer_image_key: Which camera the scorer judges from (must match training).
        scorer_action_dim: Env action dim; pi05 chunks are sliced to this before scoring.
        effort_action_index: Gripper dim of the action (LIBERO and SO-101: last).
            Grip-effort hints (batch["effort"] or an "effort@X" task tag, stripped
            before the language model sees the text) scale this channel.
    """

    n_samples: int = 8
    scorer_path: str = ""
    scorer_image_key: str = "observation.images.image"
    scorer_action_dim: int = 7
    effort_action_index: int = -1

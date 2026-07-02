from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("wm_act")
@dataclass
class WMACTConfig(ACTConfig):
    """ACT plus a JEPA-style latent world-model auxiliary objective (see wm_act/README.md).

    Extra args:
        wm_future_offset: How many frames ahead the world-model head must predict.
            15 frames = 0.5s at the dataset's 30fps — far enough that predicting it
            requires understanding contact dynamics, near enough to stay predictable.
        wm_loss_weight: Weight of the world-model loss. 0 disables it entirely,
            which makes the policy exactly plain ACT (useful as an ablation).
        wm_ema_momentum: EMA momentum for the target encoder (prevents latent collapse).
        wm_hidden_dim: Hidden width of the dynamics predictor MLP.
    """

    wm_future_offset: int = 15
    wm_loss_weight: float = 1.0
    wm_ema_momentum: float = 0.996
    wm_hidden_dim: int = 512

    @property
    def observation_delta_indices(self) -> list:
        # Ask the dataset for the current frame plus the future frame the
        # world-model head must predict. Every observation key arrives stacked
        # as (B, 2, ...); WMACTPolicy.forward splits them.
        return [0, self.wm_future_offset]

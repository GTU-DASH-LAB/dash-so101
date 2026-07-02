import copy

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION

from .configuration_wmact import WMACTConfig


class WMACTPolicy(ACTPolicy):
    """ACT with a JEPA-style latent world model.

    During training, an auxiliary head must predict the visual latent `wm_future_offset`
    frames into the future, given the current visual latent and the action chunk. The
    prediction target comes from an EMA copy of the vision backbone (stop-gradient), the
    standard recipe to prevent latent collapse. This forces the shared backbone to encode
    action-conditioned dynamics — "what will the world look like after these actions" —
    instead of only imitation-relevant appearance.

    At inference the world-model head is simply never called: select_action /
    predict_action_chunk are inherited from ACT unchanged, so runtime cost is identical
    to plain ACT.
    """

    config_class = WMACTConfig
    name = "wm_act"

    def __init__(self, config: WMACTConfig, **kwargs):
        super().__init__(config, **kwargs)

        feat_dim = self.model.encoder_img_feat_input_proj.in_channels  # 512 for resnet18
        act_dim = config.action_feature.shape[0]
        hidden = config.wm_hidden_dim

        self.wm_action_proj = nn.Linear(config.chunk_size * act_dim, hidden)
        self.wm_predictor = nn.Sequential(
            nn.Linear(feat_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )
        self.wm_target_backbone = copy.deepcopy(self.model.backbone).requires_grad_(False)

    def _pooled_feats(self, backbone: nn.Module, images: list[Tensor]) -> Tensor:
        """(B, feat_dim) mean-pooled layer4 features, averaged over cameras."""
        feats = [backbone(img)["feature_map"].mean(dim=(2, 3)) for img in images]
        return torch.stack(feats).mean(0)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # Training batches deliver each observation key stacked as (B, 2, ...) =
        # (current, future) because of WMACTConfig.observation_delta_indices.
        cur_batch = dict(batch)
        future_imgs: list[Tensor] = []
        future_pad: Tensor | None = None
        for key, feat in self.config.input_features.items():
            if key in batch and batch[key].dim() == len(feat.shape) + 2:
                cur_batch[key] = batch[key][:, 0]
                if key in self.config.image_features:
                    future_imgs.append(batch[key][:, 1])
                    pad_key = f"{key}_is_pad"
                    if future_pad is None and pad_key in batch:
                        future_pad = batch[pad_key][:, 1]

        loss, loss_dict = super().forward(cur_batch)

        if future_imgs:
            cur_imgs = [cur_batch[key] for key in self.config.image_features]
            cur_feat = self._pooled_feats(self.model.backbone, cur_imgs)
            action_emb = self.wm_action_proj(batch[ACTION].flatten(start_dim=1))
            pred = self.wm_predictor(torch.cat([cur_feat, action_emb], dim=-1))
            with torch.no_grad():
                target = self._pooled_feats(self.wm_target_backbone, future_imgs)

            wm_err = 1 - F.cosine_similarity(pred, target, dim=-1)  # (B,)
            if future_pad is not None:
                valid = ~future_pad
                wm_loss = (wm_err * valid).sum() / valid.sum().clamp_min(1)
            else:
                wm_loss = wm_err.mean()

            loss = loss + self.config.wm_loss_weight * wm_loss
            loss_dict["wm_loss"] = wm_loss.item()

            if self.training:
                self._update_wm_target()

        return loss, loss_dict

    @torch.no_grad()
    def _update_wm_target(self):
        m = self.config.wm_ema_momentum
        for p, tp in zip(
            self.model.backbone.parameters(), self.wm_target_backbone.parameters(), strict=True
        ):
            tp.lerp_(p, 1.0 - m)

"""World-model scorer: judges candidate action chunks by their predicted futures.

Given the current image latent and an action chunk, a predictor (JEPA-style,
EMA-target trained) imagines the future latent; a progress head — trained on the
fact that demo episodes monotonically approach success — estimates how far along
the task that future is. Score = predicted progress. Small on purpose (~13M):
it only has to *rank* the base VLA's own proposals, not act.
"""

import copy

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class WorldModelScorer(nn.Module):
    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 50,
        hidden: int = 512,
        pretrained_backbone: bool = False,  # True only when training (downloads weights)
    ):
        super().__init__()
        from torchvision.models import resnet18

        self.action_dim, self.chunk_size = action_dim, chunk_size
        self.backbone = resnet18(weights="IMAGENET1K_V1" if pretrained_backbone else None)
        self.backbone.fc = nn.Identity()  # -> (B, 512) latents
        self.target_backbone = copy.deepcopy(self.backbone).requires_grad_(False)
        self.action_proj = nn.Linear(chunk_size * action_dim, hidden)
        self.predictor = nn.Sequential(
            nn.Linear(512 + hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 512),
        )
        self.progress = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def _prep_chunk(self, chunks: Tensor) -> Tensor:
        """Slice/pad any (…, T, D) chunk to the trained (chunk_size, action_dim)."""
        chunks = chunks[..., : self.action_dim]
        t = chunks.shape[-2]
        if t > self.chunk_size:
            chunks = chunks[..., : self.chunk_size, :]
        elif t < self.chunk_size:
            pad = chunks[..., -1:, :].expand(*chunks.shape[:-2], self.chunk_size - t, self.action_dim)
            chunks = torch.cat([chunks, pad], dim=-2)
        return chunks.flatten(start_dim=-2)

    def predict_future(self, latent: Tensor, chunks: Tensor) -> Tensor:
        return self.predictor(torch.cat([latent, self.action_proj(self._prep_chunk(chunks))], dim=-1))

    @torch.no_grad()
    def score(self, image: Tensor, chunks: Tensor) -> Tensor:
        """image (B, C, H, W); chunks (B, K, T, D) -> scores (B, K): predicted progress."""
        b, k = chunks.shape[:2]
        latent = self.backbone(image)  # (B, 512)
        latent = latent.unsqueeze(1).expand(b, k, -1).reshape(b * k, -1)
        future = self.predict_future(latent, chunks.reshape(b * k, *chunks.shape[2:]))
        return self.progress(future).view(b, k)

    @torch.no_grad()
    def ema_update(self, momentum: float = 0.996):
        for p, tp in zip(self.backbone.parameters(), self.target_backbone.parameters(), strict=True):
            tp.lerp_(p, 1.0 - momentum)

    def save(self, path: str):
        torch.save(
            {"state_dict": self.state_dict(),
             "meta": {"action_dim": self.action_dim, "chunk_size": self.chunk_size}},
            path,
        )

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "WorldModelScorer":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        scorer = cls(**ckpt["meta"])
        scorer.load_state_dict(ckpt["state_dict"])
        return scorer.eval()


def jepa_progress_losses(
    scorer: WorldModelScorer,
    image: Tensor,          # (B, C, H, W) current frame (post pi05 preprocessing)
    future_image: Tensor,   # (B, C, H, W) frame at t+offset (same preprocessing)
    chunks: Tensor,         # (B, T, D) ground-truth action chunk (normalized space)
    progress_now: Tensor,   # (B,) in [0, 1]
    progress_future: Tensor,  # (B,) in [0, 1]
) -> tuple[Tensor, dict]:
    latent = scorer.backbone(image)
    pred_future = scorer.predict_future(latent, chunks)
    with torch.no_grad():
        target_future = scorer.target_backbone(future_image)
    jepa = (1 - F.cosine_similarity(pred_future, target_future, dim=-1)).mean()
    prog_now = F.mse_loss(scorer.progress(latent).squeeze(-1), progress_now)
    prog_fut = F.mse_loss(scorer.progress(pred_future).squeeze(-1), progress_future)
    loss = jepa + prog_now + prog_fut
    return loss, {"jepa": jepa.item(), "prog_now": prog_now.item(), "prog_future": prog_fut.item()}

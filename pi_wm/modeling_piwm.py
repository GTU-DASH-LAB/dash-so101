import torch
from torch import Tensor

from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from .configuration_piwm import PiWMConfig
from .scorer import WorldModelScorer


def pick_best(chunks: Tensor, scores: Tensor, n: int) -> Tensor:
    """chunks (B*n, T, D) grouped per original sample; scores (B, n) -> best chunk (B, T, D)."""
    b = scores.shape[0]
    grouped = chunks.view(b, n, *chunks.shape[1:])
    return grouped[torch.arange(b, device=chunks.device), scores.argmax(dim=1)]


class PiWMPolicy(PI05Policy):
    """pi0.5 with world-model best-of-N action selection.

    Weights-identical to pi05 (the checkpoint loads unchanged); the only override is
    predict_action_chunk: replicate the observation N times, let the flow model sample
    N distinct chunks from N noise draws, score each chunk's predicted future with the
    world-model scorer, and queue the chunk with the highest predicted task progress.
    n_samples=1 or no scorer -> byte-for-byte pi0.5 (the A/B baseline).
    """

    config_class = PiWMConfig
    name = "pi_wm"

    def __init__(self, config: PiWMConfig, **kwargs):
        super().__init__(config, **kwargs)
        # ponytail: a plain list hides the scorer from nn.Module registration, keeping
        # state_dict pi05-compatible so from_pretrained(strict) accepts pi05 checkpoints.
        self._scorer_holder: list[WorldModelScorer] = []

    def _scorer(self, device) -> WorldModelScorer:
        if not self._scorer_holder:
            scorer = WorldModelScorer.load(self.config.scorer_path, map_location=device)
            self._scorer_holder.append(scorer.to(device))
        return self._scorer_holder[0]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        n = self.config.n_samples
        if n <= 1 or not self.config.scorer_path:
            return super().predict_action_chunk(batch, **kwargs)

        replicated = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                replicated[key] = value.repeat_interleave(n, dim=0)
            elif isinstance(value, (list, tuple)):
                replicated[key] = [x for x in value for _ in range(n)]
            else:
                replicated[key] = value

        chunks = super().predict_action_chunk(replicated, **kwargs)  # (B*n, T, D)
        image = batch[self.config.scorer_image_key]
        scores = self._scorer(image.device).score(
            image, chunks.view(image.shape[0], n, *chunks.shape[1:])
        )
        return pick_best(chunks, scores, n)

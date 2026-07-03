import torch
from torch import Tensor

from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from point_act.modeling_pointact import TASK_EFFORT_RE, parse_task_effort

from .configuration_piwm import PiWMConfig
from .scorer import WorldModelScorer


def pick_best(chunks: Tensor, scores: Tensor, n: int) -> Tensor:
    """chunks (B*n, T, D) grouped per original sample; scores (B, n) -> best chunk (B, T, D)."""
    b = scores.shape[0]
    grouped = chunks.view(b, n, *chunks.shape[1:])
    return grouped[torch.arange(b, device=chunks.device), scores.argmax(dim=1)]


def extract_effort_and_clean(batch: dict) -> tuple[dict, list | None]:
    """Pull grip-effort hints out of the batch. pi0.5 reads the task string as
    language, so any "effort@X" tag is STRIPPED before the base model sees the text.
    Sources: batch["effort"] (per-sample floats) beats task-string tags."""
    efforts = None
    if "effort" in batch:
        raw = batch["effort"]
        efforts = [max(0.3, min(1.3, float(e))) for e in raw]
        batch = {k: v for k, v in batch.items() if k != "effort"}
    task = batch.get("task")
    if task is not None:
        strings = [str(s) for s in (task if isinstance(task, (list, tuple)) else [task])]
        parsed = [parse_task_effort(s) for s in strings]
        if any(p is not None for p in parsed):
            if efforts is None:
                efforts = parsed
            batch = dict(batch)
            batch["task"] = [TASK_EFFORT_RE.sub("", s).strip() for s in strings]
    return batch, efforts


def apply_effort(chunk: Tensor, efforts: list | None, gripper_index: int) -> Tensor:
    """Scale only the gripper channel around the chunk's first step:
    g'_t = g_0 + effort * (g_t - g_0). Affine-invariant (valid in normalized space);
    effort < 1 grips more gently, > 1 squeezes deeper than demonstrated."""
    for b, effort in enumerate(efforts or []):
        if effort is not None and effort != 1.0:
            ref = chunk[b, 0, gripper_index]
            chunk[b, :, gripper_index] = ref + effort * (chunk[b, :, gripper_index] - ref)
    return chunk


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
        batch, efforts = extract_effort_and_clean(batch)
        n = self.config.n_samples
        if n <= 1 or not self.config.scorer_path:
            chunk = super().predict_action_chunk(batch, **kwargs)
            return apply_effort(chunk, efforts, self.config.effort_action_index)

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
        # Score in the demo distribution (unscaled), then apply effort to the winner.
        return apply_effort(pick_best(chunks, scores, n), efforts, self.config.effort_action_index)

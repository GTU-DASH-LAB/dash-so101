"""Smoke test for the pi-WM scorer and selection math. No model downloads, no pi05
instantiation (that needs the checkpoint — GX10 territory). Run from the repo root:

    PYTHONPATH=. .venv/bin/python pi_wm/smoke_test.py
"""

import tempfile

import torch

from pi_wm.modeling_piwm import pick_best
from pi_wm.scorer import WorldModelScorer, jepa_progress_losses


def main():
    b, k, t, d = 2, 4, 50, 7
    scorer = WorldModelScorer(action_dim=d, chunk_size=t, pretrained_backbone=False)

    # Training losses: finite, and gradients reach predictor + progress head, not the EMA target.
    image = torch.rand(b, 3, 224, 224)
    loss, parts = jepa_progress_losses(
        scorer, image, torch.rand(b, 3, 224, 224), torch.randn(b, t, d),
        torch.rand(b), torch.rand(b),
    )
    assert torch.isfinite(loss), parts
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in scorer.predictor.parameters())
    assert all(p.grad is None for p in scorer.target_backbone.parameters())

    # EMA moves after the backbone changes.
    before = next(scorer.target_backbone.parameters()).clone()
    with torch.no_grad():
        next(scorer.backbone.parameters()).add_(1.0)
    scorer.ema_update()
    assert not torch.equal(before, next(scorer.target_backbone.parameters()))

    # Scoring: shapes, and chunks longer/shorter/wider than trained are handled.
    scores = scorer.score(image, torch.randn(b, k, t, d))
    assert scores.shape == (b, k) and torch.isfinite(scores).all()
    assert scorer.score(image, torch.randn(b, k, t + 10, d + 25)).shape == (b, k)  # pi05 pads dims
    assert scorer.score(image, torch.randn(b, k, t - 20, d)).shape == (b, k)

    # Selection picks exactly the argmax chunk per sample.
    chunks = torch.arange(b * k, dtype=torch.float32).view(b * k, 1, 1).expand(b * k, t, d)
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.3], [0.5, 0.1, 0.8, 0.2]])
    best = pick_best(chunks, scores, k)
    assert best[0, 0, 0] == 1.0 and best[1, 0, 0] == 6.0, best[:, 0, 0]

    # Save/load round trip.
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        scorer.save(f.name)
        loaded = WorldModelScorer.load(f.name)
        assert loaded.chunk_size == t and loaded.action_dim == d

    n_params = sum(p.numel() for p in scorer.parameters() if p.requires_grad)
    print(f"smoke test OK (scorer: {n_params / 1e6:.1f}M trainable params)")


if __name__ == "__main__":
    main()

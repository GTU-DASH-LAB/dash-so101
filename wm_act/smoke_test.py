"""Smoke test for WM-ACT: forward/backward on a fake training batch, EMA update,
and single-frame inference. Run from the repo root:

    PYTHONPATH=. .venv/bin/python wm_act/smoke_test.py
"""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature

from wm_act.configuration_wmact import WMACTConfig
from wm_act.modeling_wmact import WMACTPolicy


def main():
    b, chunk, state_dim = 2, 20, 6
    cfg = WMACTConfig(
        chunk_size=chunk,
        n_action_steps=chunk,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 128)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(state_dim,))},
    )
    policy = WMACTPolicy(cfg)
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.1f}M")

    train_batch = {
        "observation.state": torch.randn(b, 2, state_dim),
        "observation.images.front": torch.rand(b, 2, 3, 96, 128),
        "observation.images.front_is_pad": torch.tensor([[False, False], [False, True]]),
        "action": torch.randn(b, chunk, state_dim),
        "action_is_pad": torch.zeros(b, chunk, dtype=torch.bool),
    }

    policy.train()
    loss, loss_dict = policy.forward(train_batch)
    assert torch.isfinite(loss), loss
    assert {"l1_loss", "kld_loss", "wm_loss"} <= loss_dict.keys(), loss_dict
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.wm_predictor.parameters())
    assert all(p.grad is None for p in policy.wm_target_backbone.parameters())

    # EMA: after an optimizer step moves the backbone, the next forward must move the target.
    tgt_before = next(policy.wm_target_backbone.parameters()).clone()
    opt = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=1e-2)
    opt.step()
    policy.forward(train_batch)
    assert not torch.equal(tgt_before, next(policy.wm_target_backbone.parameters())), "EMA did not update"

    # Inference: single-frame observations, inherited ACT path.
    policy.reset()
    obs = {
        "observation.state": torch.randn(1, state_dim),
        "observation.images.front": torch.rand(1, 3, 96, 128),
    }
    action = policy.select_action(obs)
    assert action.shape == (1, state_dim), action.shape

    print("smoke test OK:", {k: round(v, 4) for k, v in loss_dict.items()})


if __name__ == "__main__":
    main()

"""Smoke test for Point-ACT: marker drawing from all three point sources, training
forward/backward, and point-sensitivity of the policy output. Run from the repo root:

    PYTHONPATH=. .venv/bin/python point_act/smoke_test.py
"""

import json
import tempfile

import torch

from lerobot.configs.types import FeatureType, PolicyFeature

from point_act.configuration_pointact import PointACTConfig
from point_act.modeling_pointact import PointACTPolicy, parse_task_points


def make_policy(point_labels_path=None):
    b_state = 6
    cfg = PointACTConfig(
        chunk_size=20,
        n_action_steps=20,
        point_labels_path=point_labels_path,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(b_state,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 128)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(b_state,))},
    )
    return PointACTPolicy(cfg)


def main():
    assert parse_task_points("pick@0.43,0.61 place@0.72,0.35") == {
        "pick": (0.43, 0.61),
        "place": (0.72, 0.35),
    }

    # Sidecar labels for episodes 0 and 1.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"0": {"pick": [0.25, 0.5], "place": [0.75, 0.5]}, "1": {"pick": [0.5, 0.5]}}, f)
        labels_path = f.name

    policy = make_policy(labels_path)
    b, chunk, state_dim = 2, 20, 6
    train_batch = {
        "observation.state": torch.randn(b, 2, state_dim),
        "observation.images.front": torch.rand(b, 2, 3, 96, 128),
        "action": torch.randn(b, chunk, state_dim),
        "action_is_pad": torch.zeros(b, chunk, dtype=torch.bool),
        "episode_index": torch.tensor([0, 1]),
    }

    # Markers land in the image: sidecar pick point for episode 0 is (0.25, 0.5).
    drawn = policy._draw_markers(train_batch)
    img = drawn["observation.images.front"]
    px, py = int(0.25 * 127), int(0.5 * 95)
    assert img[0, 0, 1, py, px] == 2.0 and img[0, 0, 0, py, px] == -2.0, "pick marker not drawn"
    assert img[0, 1, 1, py, px] == 2.0, "marker missing on the future frame"
    assert train_batch["observation.images.front"][0, 0, 1, py, px] != 2.0, "input batch was mutated"

    policy.train()
    loss, loss_dict = policy.forward(train_batch)
    assert torch.isfinite(loss) and "wm_loss" in loss_dict
    loss.backward()

    # Inference: explicit point tensors beat everything, and moving the pick point
    # must change the predicted actions (the policy actually sees the markers).
    policy.reset()
    obs = {
        "observation.state": torch.randn(1, state_dim),
        "observation.images.front": torch.rand(1, 3, 96, 128),
        "pick_point": torch.tensor([[0.2, 0.5]]),
        "place_point": torch.tensor([[0.8, 0.5]]),
    }
    chunk_a = policy.predict_action_chunk(obs)
    obs2 = dict(obs, pick_point=torch.tensor([[0.9, 0.9]]))
    chunk_b = policy.predict_action_chunk(obs2)
    assert chunk_a.shape == (1, chunk, state_dim)
    assert not torch.allclose(chunk_a, chunk_b), "policy output ignores the pick point"

    # Task-string source (the lerobot-rollout path).
    obs3 = {
        "observation.state": obs["observation.state"],
        "observation.images.front": obs["observation.images.front"],
        "task": ["pick@0.2,0.5 place@0.8,0.5"],
    }
    chunk_c = policy.predict_action_chunk(obs3)
    assert torch.allclose(chunk_a, chunk_c), "task-string points differ from tensor points"

    # Effort scales ONLY the gripper channel, relative to the chunk's first step.
    from point_act.modeling_pointact import parse_task_effort

    assert parse_task_effort("pick@0.2,0.5 effort@0.6") == 0.6
    assert parse_task_effort("effort@9") == 1.3 and parse_task_effort("no effort here") is None
    obs4 = dict(obs3, task=["pick@0.2,0.5 place@0.8,0.5 effort@0.5"])
    policy.reset()
    chunk_e = policy.predict_action_chunk(obs4)
    expected_grip = chunk_c[0, 0, -1] + 0.5 * (chunk_c[0, :, -1] - chunk_c[0, 0, -1])
    assert torch.allclose(chunk_e[0, :, -1], expected_grip, atol=1e-5), "gripper not effort-scaled"
    assert torch.allclose(chunk_e[0, :, :-1], chunk_c[0, :, :-1]), "effort leaked into arm joints"

    print("smoke test OK:", {k: round(v, 4) for k, v in loss_dict.items()})


if __name__ == "__main__":
    main()

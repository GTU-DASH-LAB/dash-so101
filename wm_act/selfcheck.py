"""Did the world model actually learn physics? Run this on the GX10 after training.

Slides through a recorded episode: at each frame t, predict the visual latent at
t+offset from (frame t, the recorded action chunk), then compare with the real
frame t+offset. Two conditions:

  true actions      -> surprise should be LOW  (model knows what its actions cause)
  shuffled actions  -> surprise should be HIGH (prediction must depend on actions,
                       not just visual continuity)

A clear gap between the two is the evidence of action-conditioned dynamics —
"physics understanding" in the only measurable sense. The reported p95 of the
true-action surprise is the threshold to give SurpriseMonitor at runtime.

    PYTHONPATH=. python wm_act/selfcheck.py \
        --policy-path outputs/train/pointact_pickplace/checkpoints/last/pretrained_model \
        --repo-id fouad1233/so101_pencil_pickup --episode 0
"""

import argparse
import random

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = args.device
    policy = (
        get_policy_class(cfg.type)
        .from_pretrained(args.policy_path, config=cfg)
        .to(args.device)
        .eval()
    )
    pre, _ = make_pre_post_processors(cfg, pretrained_path=args.policy_path)

    ds = LeRobotDataset(args.repo_id, episodes=[args.episode])
    offset, chunk = cfg.wm_future_offset, cfg.chunk_size
    n = len(ds)

    def latent(i):
        return policy.visual_latent(pre({k: v.unsqueeze(0) for k, v in ds[i].items() if torch.is_tensor(v)}))

    def chunk_at(i):
        acts = torch.stack([ds[min(i + j, n - 1)]["action"] for j in range(chunk)]).unsqueeze(0)
        return pre({"action": acts})["action"]

    true_s, shuf_s = [], []
    stride = max(1, offset // 3)
    for t in range(0, n - offset, stride):
        lat_t, lat_future = latent(t), latent(t + offset)
        pred = policy.predict_future_latent(lat_t, chunk_at(t))
        true_s.append(float(policy.surprise(pred, lat_future)))
        r = random.randrange(0, n - offset)
        pred_shuf = policy.predict_future_latent(lat_t, chunk_at(r))
        shuf_s.append(float(policy.surprise(pred_shuf, lat_future)))

    def stats(xs):
        xs = sorted(xs)
        return f"mean={sum(xs) / len(xs):.4f}  p95={xs[int(0.95 * (len(xs) - 1))]:.4f}"

    print(f"episode {args.episode}, {len(true_s)} probes, horizon {offset} frames")
    print(f"  true actions:     {stats(true_s)}")
    print(f"  shuffled actions: {stats(shuf_s)}")
    gap = (sum(shuf_s) - sum(true_s)) / len(true_s)
    print(f"  gap: {gap:+.4f}  ({'world model is action-conditioned' if gap > 0 else 'WARNING: no action dependence — check training'})")
    xs = sorted(true_s)
    print(f"  suggested SurpriseMonitor threshold: {xs[int(0.95 * (len(xs) - 1))]:.4f}")


if __name__ == "__main__":
    main()

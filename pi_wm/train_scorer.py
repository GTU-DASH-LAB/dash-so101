"""Train the world-model scorer on LIBERO demonstrations (GX10, ~1-2h).

Images and actions pass through pi0.5's own preprocessing pipeline, so the scorer
trains on exactly the distribution it will judge at eval time. Progress labels come
free from the demos: every LIBERO episode succeeds, so frame t sits at progress
t/(T-1) and good action chunks move progress forward.

    PYTHONPATH=. python pi_wm/train_scorer.py \
        --pi05-path pi_wm_checkpoint \
        --out outputs/scorer_libero.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader, Dataset

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

from pi_wm.scorer import WorldModelScorer, jepa_progress_losses
from so_brain.relabel import episode_starts


class ScorerDataset(Dataset):
    def __init__(self, ds: LeRobotDataset, offset: int, image_key: str):
        self.ds, self.offset, self.image_key = ds, offset, image_key
        starts = episode_starts(ds)
        self.bounds = {}  # episode_index -> (start, end)
        for ep, start in enumerate(starts):
            end = starts[ep + 1] if ep + 1 < len(starts) else len(ds)
            self.bounds[ep] = (start, end)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        start, end = self.bounds[int(item["episode_index"])]
        length = max(end - start, 2)
        frame = int(item["frame_index"])
        item["future_image"] = self.ds[min(i + self.offset, end - 1)][self.image_key]
        item["progress_now"] = torch.tensor(frame / (length - 1), dtype=torch.float32)
        item["progress_future"] = torch.tensor(
            min(frame + self.offset, length - 1) / (length - 1), dtype=torch.float32
        )
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi05-path", help="local pi_wm checkpoint dir; required unless --env-space")
    parser.add_argument(
        "--env-space", action="store_true",
        help="train on raw dataset images/actions (no pi05 pipeline) — the judge for committee.py",
    )
    parser.add_argument("--chunk-size", type=int, default=50, help="only used with --env-space")
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--image-key", default="observation.images.image")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--future-offset", type=int, default=10, help="frames ahead the WM predicts")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="outputs/scorer_libero.pt")
    args = parser.parse_args()

    if args.env_space:
        pre, chunk = None, args.chunk_size
    else:
        assert args.pi05_path, "--pi05-path is required unless --env-space is set"
        cfg = PreTrainedConfig.from_pretrained(args.pi05_path)
        cfg.device = args.device
        pre, _ = make_pre_post_processors(cfg, pretrained_path=args.pi05_path)
        chunk = cfg.chunk_size

    ds = LeRobotDataset(
        args.repo_id,
        delta_timestamps={"action": [i / 30 for i in range(chunk)]},  # HuggingFaceVLA/libero is 30fps
    )
    loader = DataLoader(
        ScorerDataset(ds, args.future_offset, args.image_key),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True,
    )

    scorer = WorldModelScorer(
        action_dim=args.action_dim, chunk_size=chunk, pretrained_backbone=True
    ).to(args.device)
    opt = torch.optim.AdamW([p for p in scorer.parameters() if p.requires_grad], lr=args.lr)

    step = 0
    while step < args.steps:
        for batch in loader:
            future_raw = batch.pop("future_image")
            if pre is None:  # env-space: raw images (0-1 CHW) and raw env actions
                image = batch[args.image_key].to(args.device)
                future = future_raw.to(args.device)
                actions = batch["action"].to(args.device)
            else:
                processed = pre(batch)
                # second pass so the future frame gets the identical image pipeline
                image = processed[args.image_key]
                future = pre({**batch, args.image_key: future_raw})[args.image_key]
                actions = processed["action"]

            loss, parts = jepa_progress_losses(
                scorer,
                image,
                future,
                actions,
                batch["progress_now"].to(args.device),
                batch["progress_future"].to(args.device),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            scorer.ema_update()

            if step % 100 == 0:
                print(f"step {step:6d}  loss={loss.item():.4f}  {parts}")
            step += 1
            if step >= args.steps:
                break

    scorer.save(args.out)
    print(f"scorer saved to {args.out}")


if __name__ == "__main__":
    main()

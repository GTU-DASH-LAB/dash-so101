#!/usr/bin/env python3
"""Open a recorded session from disk and decode EVERY frame of every demo.
usage: verify_session.py <session_dir> <expected_episodes>"""
import sys
from pathlib import Path

from lerobot.datasets import LeRobotDataset

sess, expected = Path(sys.argv[1]), int(sys.argv[2])
ds = LeRobotDataset("test/fake_session", root=sess / "dataset")
assert ds.num_episodes == expected, f"expected {expected} demos, found {ds.num_episodes}"
for ep in range(ds.num_episodes):
    row = ds.meta.episodes[ep]
    n = row["dataset_to_index"] - row["dataset_from_index"]
    for i in range(row["dataset_from_index"], row["dataset_to_index"]):
        item = ds[i]
        assert item["observation.images.wrist"].shape == (3, 480, 640)
    print(f"  demo {ep}: {n} frames, all decode")
leftover = list((sess / "dataset").glob("images/**/*.png"))
assert not leftover, f"{len(leftover)} leftover temp PNGs"
print(f"VERIFIED {ds.num_episodes} demos, {ds.num_frames} frames, no leftover temp images")

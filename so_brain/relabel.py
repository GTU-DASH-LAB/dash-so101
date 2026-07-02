"""Auto-label a LeRobotDataset with pick/place points for Point-ACT training.

For each episode: take the first frame, locate the pick object and the place
destination with OWLv2, write normalized points to a JSON sidecar keyed by
episode_index. Run this where the dataset will be trained (the GX10):

    PYTHONPATH=. python so_brain/relabel.py --repo-id fouad1233/so101_pencil_pickup \
        --pick "a pen" --place "a black mouse pad" --out point_labels.json

Omit --pick/--place to derive them from each episode's recorded task string.
"""

import argparse
import json

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so_brain.grounding import locate
from so_brain.planner import naive_plan


def episode_starts(ds: LeRobotDataset) -> list[int]:
    """Global dataset index of frame 0 of each episode."""
    episodes = ds.meta.episodes
    row = episodes[0]
    if "dataset_from_index" in row:
        return [int(episodes[e]["dataset_from_index"]) for e in range(len(episodes))]
    # Fallback: episodes are stored contiguously; accumulate lengths.
    starts, acc = [], 0
    for e in range(len(episodes)):
        starts.append(acc)
        acc += int(episodes[e]["length"])
    return starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--pick", help="detector phrase for the object (default: parse episode task)")
    parser.add_argument("--place", help="detector phrase for the destination")
    parser.add_argument("--out", default="point_labels.json")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    ds = LeRobotDataset(args.repo_id)
    img_key = next(k for k in ds.meta.camera_keys)
    labels, failures = {}, []

    for ep, start in enumerate(episode_starts(ds)):
        item = ds[start]
        img = (item[img_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        pick_phrase, place_phrase = args.pick, args.place
        if not (pick_phrase and place_phrase):
            parsed = naive_plan(str(item.get("task", "")))
            pick_phrase = pick_phrase or parsed["object"]
            place_phrase = place_phrase or parsed["destination"]

        entry = {}
        for kind, phrase in (("pick", pick_phrase), ("place", place_phrase)):
            try:
                u, v, score = locate(img, phrase, threshold=args.threshold)
                entry[kind] = [round(u, 4), round(v, 4)]
                print(f"ep {ep:3d} {kind}:{phrase!r} -> ({u:.3f}, {v:.3f}) score={score:.2f}")
            except LookupError:
                failures.append((ep, kind, phrase))
                print(f"ep {ep:3d} {kind}:{phrase!r} -> NOT FOUND (episode will train unmarked)")
        if entry:
            labels[str(ep)] = entry

    with open(args.out, "w") as f:
        json.dump(labels, f, indent=1)
    print(f"\nwrote {len(labels)}/{len(episode_starts(ds))} episodes to {args.out}")
    if failures:
        print(f"{len(failures)} detection failures — consider better phrases or lower --threshold")


if __name__ == "__main__":
    main()

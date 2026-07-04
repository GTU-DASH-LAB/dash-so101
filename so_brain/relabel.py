"""Auto-label a LeRobotDataset with pick/place points for Point-ACT training.

Locates the pick object and the place destination with the grounder and writes
normalized points to a JSON sidecar keyed by episode_index. With --every N the
object is re-located every N frames THROUGH each episode, so training markers
track the object while it moves (grasped, carried, bumped) instead of lying
after frame 0. Run this where the dataset will be trained (GX10 / Kaggle):

    PYTHONPATH=. python so_brain/relabel.py --repo-id fouad1233/so101_pencil_pickup \
        --pick "a pen" --place "a black mouse pad" --every 10 --out point_labels.json

Omit --pick/--place to derive them from each episode's recorded task string.
--every 0 labels frame 0 only (the old static format; both formats load).
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
    parser.add_argument("--every", type=int, default=0,
                        help="re-locate every N frames through each episode (0 = frame 0 only)")
    args = parser.parse_args()

    ds = LeRobotDataset(args.repo_id)
    img_key = next(k for k in ds.meta.camera_keys)
    starts = episode_starts(ds)
    labels, failures = {}, 0

    def detect(img, pick_phrase, place_phrase):
        nonlocal failures
        entry = {}
        for kind, phrase in (("pick", pick_phrase), ("place", place_phrase)):
            try:
                u, v, _ = locate(img, phrase, threshold=args.threshold)
                entry[kind] = [round(u, 4), round(v, 4)]
            except LookupError:
                failures += 1
        return entry

    for ep, start in enumerate(starts):
        end = starts[ep + 1] if ep + 1 < len(starts) else len(ds)
        item = ds[start]
        pick_phrase, place_phrase = args.pick, args.place
        if not (pick_phrase and place_phrase):
            parsed = naive_plan(str(item.get("task", "")))
            pick_phrase = pick_phrase or parsed["object"]
            place_phrase = place_phrase or parsed["destination"]

        def frame_img(i):
            return (ds[i][img_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        if args.every:
            frames = {}
            for off in range(0, end - start, args.every):
                entry = detect(frame_img(start + off), pick_phrase, place_phrase)
                if entry:
                    frames[str(off)] = entry
            if frames:
                labels[str(ep)] = {"every": args.every, "frames": frames}
            print(f"ep {ep:3d}: {len(frames)}/{(end - start + args.every - 1) // args.every} "
                  f"frames labeled ({pick_phrase!r} -> {place_phrase!r})")
        else:
            entry = detect(frame_img(start), pick_phrase, place_phrase)
            if entry:
                labels[str(ep)] = entry
            print(f"ep {ep:3d}: {sorted(entry)} ({pick_phrase!r} -> {place_phrase!r})")

    with open(args.out, "w") as f:
        json.dump(labels, f, indent=1)
    print(f"\nwrote {len(labels)}/{len(starts)} episodes to {args.out}")
    if failures:
        print(f"{failures} detection failures — consider better phrases or lower --threshold")


if __name__ == "__main__":
    main()

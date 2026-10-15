#!/usr/bin/env python3
"""Offline sanity check of a trained policy, the way the robot will run it (no arm needed).

    offline_check.py --policy <pretrained_model dir or Hub id> --dataset-root <dir>
                     [--repo-id ID] [--episodes 90-99] [--frames 60] [--horizon 15] [--device cuda]
                     [--set key=value ...]   policy config overrides, e.g. checkpoint_path=...
                     [--swap-images]         give each frame the camera images of ANOTHER frame

Loads the policy and its saved pre/post-processors with STOCK lerobot (none of the training
launchers' patches), feeds single frames from held-out episodes of the ORIGINAL dataset (arm
frame, full-size images, the state at that instant), takes the next H actions the policy
would execute from that one observation (select_action's queue), and compares them with what
the human commanded over the same H frames. Baseline: "freeze" = keep commanding the current
state for H frames. At t+0 the command is ~the current state (a weak test), but over 0.5 s the
arm moves, so a policy that understands the scene must clearly beat "freeze".

What it catches: wrong normalization or joint-convention transforms (errors of tens to
hundreds of degrees on some joint), camera/feature-key mismatches (load errors), and a policy
that ignores its inputs (error close to "hold the current state" for every joint).
--swap-images is the vision ablation: if the error barely changes, the policy is not using the
camera (it is replaying state->action habits).
"""

import argparse
import json
import time

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", default="fouad1233/so101_pen_in_bowl_wrist")
    ap.add_argument("--episodes", default="90-99")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--swap-images", action="store_true")
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.episodes.split("-"))
    ds = LeRobotDataset(a.repo_id, root=a.dataset_root, episodes=list(range(lo, hi + 1)))
    names = ds.meta.features["action"]["names"]

    cfg = PreTrainedConfig.from_pretrained(a.policy)
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        setattr(cfg, k, v)
    cfg.device = a.device
    t0 = time.time()
    policy = get_policy_class(cfg.type).from_pretrained(a.policy, config=cfg).to(a.device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=a.policy,
                                         preprocessor_overrides={"device_processor": {"device": a.device}})
    print(f"loaded {cfg.type} in {time.time() - t0:.0f} s")

    H = a.horizon
    ep_of = np.asarray(ds.hf_dataset["episode_index"])
    ok = [i for i in range(len(ds) - H) if ep_of[i] == ep_of[i + H]]
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(ok, size=min(a.frames, len(ok)), replace=False).tolist())
    donor = np.roll(idx, len(idx) // 2) if a.swap_images else idx
    sync = torch.cuda.synchronize if a.device.startswith("cuda") else (lambda: None)
    err, freeze, lat = [], [], []
    for i, d in zip(idx, donor):
        item = ds[i]
        obs = {k: v for k, v in item.items() if k.startswith("observation.") and not k.endswith("_is_pad")}
        if d != i:
            obs.update({k: v for k, v in ds[int(d)].items() if k.startswith("observation.images.")})
        obs["task"] = item["task"]
        policy.reset()
        preds = []
        with torch.inference_mode():
            batch = pre(obs)
            for k in range(H):
                sync()
                t = time.time()
                act = post(policy.select_action(batch))
                sync()
                if k == 0:
                    lat.append(time.time() - t)
                preds.append(act.squeeze(0).float().cpu().numpy())
        gt = np.stack([ds[i + k]["action"].numpy() for k in range(H)])
        err.append(np.abs(np.stack(preds) - gt))                       # [H, D]
        freeze.append(np.abs(item["observation.state"].numpy()[None] - gt))
    err, freeze = np.stack(err), np.stack(freeze)                       # [N, H, D]
    print(f"{len(idx)} held-out frames (episodes {a.episodes}){' with SWAPPED images' if a.swap_images else ''}; next {H} actions ({H / ds.fps:.2f} s) vs the human's")
    print(f"{'joint':<18}{'policy MAE':>11}{'freeze MAE':>11}{'policy @t+H':>12}{'freeze @t+H':>12}")
    for j, n in enumerate(names):
        print(f"{n:<18}{err[:, :, j].mean():>11.2f}{freeze[:, :, j].mean():>11.2f}"
              f"{err[:, -1, j].mean():>12.2f}{freeze[:, -1, j].mean():>12.2f}")
    print(f"{'mean':<18}{err.mean():>11.2f}{freeze.mean():>11.2f}{err[:, -1].mean():>12.2f}{freeze[:, -1].mean():>12.2f}   (degrees)")
    lat_ms = np.array(lat[1:] or lat) * 1000
    print(f"inference latency per new chunk: median {np.median(lat_ms):.0f} ms, max {lat_ms.max():.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

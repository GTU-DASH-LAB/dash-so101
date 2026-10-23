#!/usr/bin/env python3
"""Inference speed AND timing stability of a trained policy, the way lerobot-rollout runs it (no arm).

    bench_inference.py --policy <pretrained_model dir> [--set key=value ...]
                       [--accel none|trt] [--ticks 300] [--pace 30] [--episode 95]
                       [--dataset-root DIR] [--repo-id ID] [--seed 0] [--save out.npz]
                       [--compare ref.npz]

Each control tick runs exactly what the sync inference engine runs:
    action = postprocessor(policy.select_action(preprocessor(observation)))
on consecutive frames of one held-out episode, so the observation changes every tick as on the
robot. --pace 30 sleeps to the next 30 Hz tick like the control loop does (the GPU idles between
ticks, as in real use; --pace 0 runs back to back).

Reported:
  * every tick: wall time of the whole tick (pre + policy + post)
  * inference ticks only (the ticks on which the model actually ran, i.e. a new chunk was
    predicted; every tick for ACT with temporal ensembling): mean, p50, p90, p99, max, std
  * deadline misses: ticks longer than one control period (33.3 ms at 30 Hz) - each one is a
    tick on which the arm waited
--save writes the actions; --compare prints the per-joint difference against a saved run
(the parity check for an accelerated engine: same frames, same seed, same actions?).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402


def load_policy(path, sets, device):
    cfg = PreTrainedConfig.from_pretrained(path)
    for kv in sets:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        setattr(cfg, k, v)
    cfg.device = device
    cls = get_policy_class(cfg.type)
    if getattr(cfg, "use_peft", False):  # LoRA checkpoint: base weights + adapter, as lerobot-rollout loads it
        from peft import PeftConfig, PeftModel

        pc = PeftConfig.from_pretrained(path)
        policy = PeftModel.from_pretrained(cls.from_pretrained(pc.base_model_name_or_path, config=cfg), path, config=pc)
    else:
        policy = cls.from_pretrained(path, config=cfg)
    policy = policy.to(device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return cfg, policy, pre, post


def stats(ms):
    ms = np.asarray(ms)
    if ms.size == 0:
        return "n=0"
    return (f"n={ms.size:4d}  mean {ms.mean():7.2f}  p50 {np.percentile(ms, 50):7.2f}  "
            f"p90 {np.percentile(ms, 90):7.2f}  p99 {np.percentile(ms, 99):7.2f}  "
            f"max {ms.max():7.2f}  std {ms.std():6.2f}  (ms)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--accel", default="none", choices=["none", "trt"])
    ap.add_argument("--act-precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--dataset-root", default=str(Path.home() / ".cache/huggingface/lerobot/fouad1233/so101_pen_in_bowl_wrist"))
    ap.add_argument("--repo-id", default="fouad1233/so101_pen_in_bowl_wrist")
    ap.add_argument("--episode", type=int, default=95)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20, help="ticks run first and not counted")
    ap.add_argument("--pace", type=float, default=30.0, help="control rate to emulate (0 = back to back)")
    ap.add_argument("--fps", type=float, default=30.0, help="control rate that defines a deadline miss")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save")
    ap.add_argument("--compare")
    a = ap.parse_args()

    t0 = time.time()
    cfg, policy, pre, post = load_policy(a.policy, a.set, a.device)
    print(f"loaded {cfg.type} in {time.time() - t0:.0f} s", flush=True)
    if a.accel == "trt":
        import trt_accel

        t0 = time.time()
        info = trt_accel.accelerate(policy, act_precision=a.act_precision)
        print(f"TensorRT: {info} ({time.time() - t0:.0f} s)", flush=True)

    ds = LeRobotDataset(a.repo_id, root=a.dataset_root, episodes=[a.episode])
    n_frames = len(ds)
    names = ds.meta.features["action"]["names"]

    # Count the ticks on which the model actually ran (a new chunk), by wrapping the method
    # select_action calls when its queue is empty.
    target = policy.base_model.model if hasattr(policy, "base_model") else policy
    ran = {"n": 0}
    orig = target.predict_action_chunk

    def counted(*args, **kw):
        ran["n"] += 1
        return orig(*args, **kw)

    target.predict_action_chunk = counted

    sync = torch.cuda.synchronize
    period = 1.0 / a.pace if a.pace > 0 else 0.0
    torch.manual_seed(a.seed)
    policy.reset()
    pre.reset()
    post.reset()
    tick_ms, inf_ms, pre_ms, pol_ms, post_ms, actions, frames = [], [], [], [], [], [], []
    next_t = time.perf_counter()
    for k in range(a.warmup + a.ticks):
        fi = k % n_frames
        if fi == 0 and k > 0:  # wrapped around the episode: a new "episode" starts fresh
            policy.reset(); pre.reset(); post.reset()
        item = ds[fi]
        obs = {kk: v for kk, v in item.items() if kk.startswith("observation.") and not kk.endswith("_is_pad")}
        obs["task"] = item["task"]
        if period:
            now = time.perf_counter()
            if next_t > now:
                time.sleep(next_t - now)
            next_t += period
        before = ran["n"]
        sync()
        t_a = time.perf_counter()
        with torch.inference_mode():
            batch = pre(dict(obs))
            sync(); t_b = time.perf_counter()
            act = policy.select_action(batch)
            sync(); t_c = time.perf_counter()
            act = post(act)
            sync(); t_d = time.perf_counter()
        if k < a.warmup:
            continue
        tick = (t_d - t_a) * 1000
        tick_ms.append(tick); pre_ms.append((t_b - t_a) * 1000); pol_ms.append((t_c - t_b) * 1000)
        post_ms.append((t_d - t_c) * 1000)
        if ran["n"] > before:
            inf_ms.append(tick)
        actions.append(act.squeeze(0).float().cpu().numpy())
        frames.append(fi)

    actions = np.stack(actions)
    budget = 1000.0 / a.fps
    print(f"\n{cfg.type}  accel={a.accel}  episode {a.episode}  {a.ticks} ticks after {a.warmup} warm-up, "
          f"paced at {a.pace:g} Hz" if a.pace else f"\n{cfg.type}  accel={a.accel}  back to back")
    print(f"  every tick      {stats(tick_ms)}")
    print(f"  inference ticks {stats(inf_ms)}")
    print(f"  breakdown (mean ms): pre {np.mean(pre_ms):.2f}  policy {np.mean(pol_ms):.2f}  post {np.mean(post_ms):.2f}")
    print(f"  deadline misses (> {budget:.1f} ms): {int((np.asarray(tick_ms) > budget).sum())} of {len(tick_ms)} ticks")
    if a.save:
        np.savez(a.save, actions=actions, frames=np.array(frames), tick_ms=np.array(tick_ms),
                 inf_ms=np.array(inf_ms), names=np.array(names))
        print(f"  saved {a.save}")
    if a.compare:
        ref = np.load(a.compare)
        m = min(len(ref["actions"]), len(actions))
        assert (ref["frames"][:m] == np.array(frames[:m])).all(), "different frames - not comparable"
        d = np.abs(ref["actions"][:m] - actions[:m])
        print(f"  vs {a.compare}: max |diff| per joint (deg): " +
              "  ".join(f"{n}={d[:, j].max():.3f}" for j, n in enumerate(names)))
        print(f"  mean |diff| {d.mean():.4f} deg, max {d.max():.3f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

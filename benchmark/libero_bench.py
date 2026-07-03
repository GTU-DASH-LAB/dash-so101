#!/usr/bin/env python
"""LIBERO benchmark for the wm_act family on the GX10. Per-task protocol:
these policies are language-blind, so each run trains on ONE task's episodes and
evaluates on that task (--env.task_ids). Compare variants on the same task:

    export MUJOCO_GL=egl
    PYTHONPATH=. python benchmark/libero_bench.py --suite libero_object --task-id 0
    PYTHONPATH=. python benchmark/libero_bench.py --suite libero_object --task-id 0 \
        --variants wm_act_large --steps 50000

Results accumulate in benchmark/results.json; a markdown table prints after each run.
Use --dry-run to inspect the exact commands without training anything.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ID = "HuggingFaceVLA/libero"
RESULTS = Path("benchmark/results.json")

# Benchmark defaults sized for LIBERO (short-horizon, low-fps sim vs the SO-101 setup).
COMMON = [
    "--policy.chunk_size=20",
    "--policy.n_action_steps=20",
    "--policy.wm_future_offset=10",
]
VARIANTS = {
    # plain ACT ablation: world-model loss off — measures what the WM objective buys
    "act": ["--policy.type=wm_act", "--policy.wm_loss_weight=0"],
    "wm_act": ["--policy.type=wm_act"],
    "wm_act_large": [
        "--policy.type=wm_act",
        "--policy.vision_backbone=resnet34",
        "--policy.pretrained_backbone_weights=ResNet34_Weights.IMAGENET1K_V1",
        "--policy.dim_model=768",
        "--policy.n_encoder_layers=6",
        "--policy.dim_feedforward=4096",
        "--policy.wm_hidden_dim=768",
    ],
}


def task_description(suite: str, task_id: int) -> str:
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[suite]().get_task(task_id).language


def task_episodes(description: str) -> list[int]:
    """Episode indices in the LIBERO dataset whose task string matches."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(REPO_ID)
    episodes = []
    for i in range(meta.total_episodes):
        tasks = meta.episodes[i]["tasks"]
        if any(description.lower() in str(t).lower() for t in tasks):
            episodes.append(i)
    return episodes


def run(cmd: list[str], dry: bool) -> str:
    print("+", " ".join(cmd))
    if dry:
        return ""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-2000:])
    sys.stderr.write(proc.stderr[-2000:])
    proc.check_returncode()
    return proc.stdout


def success_rate(eval_dir: Path, stdout: str) -> float | None:
    for f in sorted(eval_dir.rglob("eval_info.json")):
        info = json.loads(f.read_text())
        agg = info.get("aggregated", info)
        for key in ("pc_success", "avg_success", "success_rate"):
            if key in agg:
                return float(agg[key])
    m = re.search(r"pc_success[^\d]*([\d.]+)", stdout)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-desc", help="override task language (skips libero import)")
    parser.add_argument("--episodes", help="override episode indices, e.g. '0,1,2'")
    parser.add_argument("--variants", default="act,wm_act")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-episodes", type=int, default=10, help="eval episodes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    desc = args.task_desc or task_description(args.suite, args.task_id)
    episodes = (
        [int(e) for e in args.episodes.split(",")] if args.episodes else task_episodes(desc)
    )
    print(f"task {args.suite}[{args.task_id}]: {desc!r} — {len(episodes)} episodes")
    if not episodes and not args.dry_run:
        sys.exit("no matching episodes found — pass --episodes explicitly")

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    for variant in args.variants.split(","):
        name = f"{variant}_{args.suite}_t{args.task_id}"
        out_train = f"benchmark/outputs/{name}"
        out_eval = f"benchmark/eval/{name}"

        run(
            ["lerobot-train", "--policy.discover_packages_path=wm_act"]
            + VARIANTS[variant]
            + COMMON
            + [
                "--policy.device=cuda",
                "--policy.push_to_hub=false",
                f"--dataset.repo_id={REPO_ID}",
                f"--dataset.episodes=[{','.join(map(str, episodes))}]",
                f"--batch_size={args.batch_size}",
                f"--steps={args.steps}",
                f"--save_freq={args.steps}",
                f"--output_dir={out_train}",
                f"--job_name={name}",
                "--wandb.enable=false",
            ],
            args.dry_run,
        )
        stdout = run(
            [
                "lerobot-eval",
                "--policy.discover_packages_path=wm_act",
                f"--policy.path={out_train}/checkpoints/last/pretrained_model",
                "--env.type=libero",
                f"--env.task={args.suite}",
                f"--env.task_ids=[{args.task_id}]",
                # match the dataset's render resolution, not the env default (360)
                "--env.observation_height=256",
                "--env.observation_width=256",
                "--eval.batch_size=1",
                f"--eval.n_episodes={args.n_episodes}",
                f"--output_dir={out_eval}",
            ],
            args.dry_run,
        )
        if args.dry_run:
            continue
        rate = success_rate(Path(out_eval), stdout)
        results.append(
            {"variant": variant, "suite": args.suite, "task_id": args.task_id,
             "task": desc, "steps": args.steps, "success": rate}
        )
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1))

    if results:
        print("\n| variant | suite | task | steps | success |")
        print("|---|---|---|---|---|")
        for r in results:
            print(f"| {r['variant']} | {r['suite']} | {r['task_id']} | {r['steps']} | {r['success']} |")


if __name__ == "__main__":
    main()

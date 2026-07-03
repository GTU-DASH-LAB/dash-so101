"""The committee: Pragmatic Chaos, done right for robots.

The Netflix Prize was won by *blending* many models' rating predictions — averages
of scalars, where errors cancel. Robot actions are multimodal: averaging "go around
the left" with "go around the right" hits the obstacle. So a robotics ensemble must
SELECT, not average: every member policy proposes action chunks, one env-space
world-model judge (train_scorer.py --env-space) predicts each chunk's future and
picks the winner, a sanity filter (the "IK reflex": bounds + finiteness in
end-effector action space) vetoes degenerate candidates first.

    export MUJOCO_GL=egl
    PYTHONPATH=. python pi_wm/committee.py \
        --member pi_wm_checkpoint:8 \
        --member outputs/train/molmoact2_libero:4 \
        --scorer outputs/scorer_env.pt --suite libero_object --n-episodes 10

`--member path:n` = sample n candidate chunks from that checkpoint (any policy type
LeRobot can load; heterogeneous architectures welcome — that is the point). The
per-member pick tally printed at the end is the committee's "blend weights": if one
member never wins, drop it.

Caveats (honest): all members share one env processor (XVLA needs its own — don't mix
it in); a member only helps if fine-tuned on the same benchmark; and like Netflix's
winning ensemble, K models cost K inferences — measure whether the gain pays.
"""

import argparse
import json
from pathlib import Path

import torch

IMAGE_KEY = "observation.images.image"
RESULTS = Path("benchmark/committee_results.json")


def chunk_is_sane(chunk: torch.Tensor, bound: float = 1.001) -> bool:
    """Reflex-layer veto in env action space: finite and inside the Box(-1,1) bounds.
    (On the real SO-101, lerobot's RobotKinematics FK adds workspace checks here.)"""
    return bool(torch.isfinite(chunk).all()) and float(chunk.abs().max()) <= bound


def fit_chunk(chunk: torch.Tensor, length: int) -> torch.Tensor:
    """Trim or last-step-pad a (T, D) chunk to `length` so mixed-horizon members
    can be stacked for the judge."""
    if chunk.shape[0] >= length:
        return chunk[:length]
    pad = chunk[-1:].expand(length - chunk.shape[0], chunk.shape[1])
    return torch.cat([chunk, pad], dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append", required=True,
                        help="checkpoint[:n_samples]; repeat per member")
    parser.add_argument("--scorer", required=True, help="env-space scorer (--env-space)")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    import numpy as np

    import pi_wm  # noqa: F401 — registers plugin types
    from lerobot.configs import PreTrainedConfig
    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    from pi_wm.scorer import WorldModelScorer

    members = []
    for spec in args.member:
        path, _, n = spec.partition(":")
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = args.device
        if hasattr(cfg, "n_samples"):
            cfg.n_samples = 1  # the committee does the sampling, not the member
        policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        policy.to(args.device).eval()
        pre, post = make_pre_post_processors(cfg, pretrained_path=path)
        members.append({"name": Path(path).name, "policy": policy, "pre": pre,
                        "post": post, "n": int(n or 1), "cfg": cfg})
    scorer = WorldModelScorer.load(args.scorer, map_location=args.device).to(args.device)

    task_ids = [int(t) for t in args.task_ids.split(",")] if args.task_ids else None
    env_cfg = LiberoEnv(task=args.suite, task_ids=task_ids,
                        observation_height=256, observation_width=256)
    env_pre, env_post = make_env_pre_post_processors(env_cfg, members[0]["cfg"])
    envs = make_env(env_cfg, n_envs=1)

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    member_names = ",".join(f"{m['name']}:{m['n']}" for m in members)

    for suite, task_envs in envs.items():
        for task_id, env in task_envs.items():
            max_steps = env.call("_max_episode_steps")[0]
            successes, tally = 0, dict.fromkeys([m["name"] for m in members], 0)
            for ep in range(args.n_episodes):
                raw_obs, _ = env.reset(seed=args.seed + ep)
                obs = env_pre(preprocess_observation(raw_obs))
                instruction = list(env.call("task_description"))[0]
                success, step, queue = False, 0, []

                while step < max_steps and not success:
                    if not queue:
                        obs["task"] = [instruction]
                        candidates = []  # (member name, env-space chunk (T, D))
                        for m in members:
                            m["policy"].reset()
                            batch = m["pre"](dict(obs))
                            for _ in range(m["n"]):
                                with torch.inference_mode():
                                    norm = m["policy"].predict_action_chunk(batch)
                                env_chunk = torch.stack(
                                    [m["post"](norm[:, t]) for t in range(norm.shape[1])], dim=1
                                )[0].cpu()
                                if chunk_is_sane(env_chunk):
                                    candidates.append((m["name"], env_chunk))
                        if not candidates:  # everyone vetoed — hold still briefly
                            queue = [np.zeros(members[0]["cfg"].output_features[ACTION].shape)] * 5
                            continue
                        image = obs[IMAGE_KEY].to(args.device)
                        stacked = torch.stack(
                            [fit_chunk(c, scorer.chunk_size) for _, c in candidates]
                        ).unsqueeze(0).to(args.device)
                        best = int(scorer.score(image, stacked)[0].argmax())
                        tally[candidates[best][0]] += 1
                        chosen = candidates[best][1][: args.n_action_steps]
                        queue = [a.numpy() for a in chosen]

                    action = torch.tensor(queue.pop(0)).unsqueeze(0)
                    action = env_post({ACTION: action})[ACTION]
                    raw_obs, _, terminated, truncated, info = env.step(action.cpu().numpy())
                    obs = env_pre(preprocess_observation(raw_obs))
                    step += 1
                    final = info.get("final_info", info)
                    flag = final.get("is_success")
                    if flag is not None and bool(np.asarray(flag).flatten()[0]):
                        success = True
                    elif np.asarray(terminated | truncated).flatten()[0]:
                        break

                successes += success
                print(f"[{suite}/{task_id} ep{ep}] {'success' if success else 'fail'} ({step} steps)")

            print(f"== {suite}/task{task_id} committee[{member_names}]: "
                  f"{successes}/{args.n_episodes}  picks: {tally}")
            results.append({"suite": suite, "task_id": int(task_id), "members": member_names,
                            "successes": successes, "episodes": args.n_episodes, "tally": tally})
            RESULTS.parent.mkdir(exist_ok=True)
            RESULTS.write_text(json.dumps(results, indent=1))

    close_envs(envs)


if __name__ == "__main__":
    main()

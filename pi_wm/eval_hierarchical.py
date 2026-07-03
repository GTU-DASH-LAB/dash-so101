"""Hierarchical CoT evaluation on LIBERO — the unseen / long-horizon experiment.

pi0.5 fails on LIBERO-Long compositions it never saw, even though it has seen every
atomic skill. This harness closes that gap by composing everything in this repo:

  1. DECOMPOSE (CoT VLM): the long instruction becomes atomic sub-commands phrased
     like the training distribution ("pick up the X and place it in the Y").
  2. EXECUTE: each sub-command is fed to the policy as its task string (pi0.5 is
     language-conditioned; pi-WM best-of-N applies per decision if configured).
     The action queue is flushed on every sub-goal switch.
  3. VERIFY: grounder evidence (object moved to destination?) and/or a CoT VLM check
     decide when to advance. A stuck sub-goal gets one retry, then the episode fails.
  4. REMEMBER: failures land in episodic memory with the failed sub-goal; later
     episodes of the same task plan with those lessons — LIBERO's lifelong-learning
     theme, made literal.

Run on the GX10 (needs ollama reachable for CoT; grounder optional):

    export MUJOCO_GL=egl
    PYTHONPATH=. python pi_wm/eval_hierarchical.py \
        --policy-path pi_wm_checkpoint --scorer-path outputs/scorer_libero.pt \
        --suite libero_10 --n-episodes 10

    # baselines under the identical loop:
    PYTHONPATH=. python pi_wm/eval_hierarchical.py --policy-path pi_wm_checkpoint \
        --suite libero_10 --n-episodes 10 --no-plan --n-samples 1   # = plain pi0.5
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import pi_wm  # noqa: F401 — registers the pi_wm policy type
from lerobot.configs import PreTrainedConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION

from so_brain import planner
from so_brain.memory import Memory

FRAME = "benchmark/_hier_frame.png"
RESULTS = Path("benchmark/hier_results.json")
IMAGE_KEY = "observation.images.image"


def save_frame(obs: dict) -> str:
    img = obs[IMAGE_KEY][0]
    array = img.numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.transpose(1, 2, 0)
    if array.dtype != np.uint8:
        array = (array * 255).clip(0, 255).astype(np.uint8)
    Path(FRAME).parent.mkdir(exist_ok=True)
    cv2.imwrite(FRAME, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    return FRAME


def grounder_evidence(frame: str, sub: dict, tol: float = 0.10) -> tuple[bool, str]:
    """(auto_done, evidence text) from open-vocab detection; ("", False) when disabled."""
    from so_brain import grounding

    try:
        ou, ov, _ = grounding.locate(frame, sub["object"])
        du, dv, _ = grounding.locate(frame, sub["destination"])
    except LookupError as e:
        return False, f"Detector evidence: {e}."
    dist = ((ou - du) ** 2 + (ov - dv) ** 2) ** 0.5
    return dist <= tol, (
        f"Detector evidence: {sub['object']!r} at ({ou:.2f},{ov:.2f}), "
        f"{sub['destination']!r} at ({du:.2f},{dv:.2f}), distance {dist:.2f}."
    )


def subgoal_done(sub: dict, frame: str, use_grounder: bool, use_vlm: bool) -> tuple[bool, str]:
    evidence = ""
    if use_grounder and sub.get("object") and sub.get("destination"):
        auto_done, evidence = grounder_evidence(frame, sub)
        if auto_done:
            return True, evidence
    if use_vlm:
        try:
            verdict = planner.verify_subgoal(sub["instruction"], frame, evidence)
            return verdict["done"], verdict["note"]
        except Exception as e:  # noqa: BLE001 — VLM down: fall back to step budget
            return False, f"verifier unavailable ({e})"
    return False, evidence or "no verifier"


def main():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--policy-path", required=True)
    parser_.add_argument("--suite", default="libero_10")
    parser_.add_argument("--task-ids", default="", help="comma list; empty = all in suite")
    parser_.add_argument("--n-episodes", type=int, default=10)
    parser_.add_argument("--n-samples", type=int, default=None, help="override pi_wm best-of-N")
    parser_.add_argument("--scorer-path", default=None)
    parser_.add_argument("--n-action-steps", type=int, default=10)
    parser_.add_argument("--check-every", type=int, default=40, help="steps between verifications")
    parser_.add_argument("--max-subgoal-steps", type=int, default=150)
    parser_.add_argument("--no-plan", action="store_true", help="baseline: raw long instruction")
    parser_.add_argument("--no-memory", action="store_true")
    parser_.add_argument("--no-vlm-verify", action="store_true")
    parser_.add_argument("--grounder", action="store_true", help="use SO_BRAIN_GROUNDER for evidence")
    parser_.add_argument("--device", default="cuda")
    parser_.add_argument("--seed", type=int, default=1000)
    args = parser_.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = args.device
    cfg.n_action_steps = args.n_action_steps
    if args.n_samples is not None and hasattr(cfg, "n_samples"):
        cfg.n_samples = args.n_samples
    if args.scorer_path and hasattr(cfg, "scorer_path"):
        cfg.scorer_path = args.scorer_path
    policy = get_policy_class(cfg.type).from_pretrained(args.policy_path, config=cfg)
    policy.to(args.device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.policy_path)

    task_ids = [int(t) for t in args.task_ids.split(",")] if args.task_ids else None
    env_cfg = LiberoEnv(
        task=args.suite, task_ids=task_ids, observation_height=256, observation_width=256
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg, cfg)
    envs = make_env(env_cfg, n_envs=1)

    memory = Memory("benchmark/libero_memory.jsonl")
    mode = "no_plan" if args.no_plan else "hier" + ("" if args.no_memory else "+mem")
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []

    for suite, task_envs in envs.items():
        for task_id, env in task_envs.items():
            max_steps = env.call("_max_episode_steps")[0]
            successes = 0
            for ep in range(args.n_episodes):
                raw_obs, _ = env.reset(seed=args.seed + ep)
                policy.reset()
                obs = env_pre(preprocess_observation(raw_obs))
                instruction = list(env.call("task_description"))[0]
                frame = save_frame(obs)

                lessons = "" if args.no_memory else memory.lessons(instruction)
                subgoals = [{"instruction": instruction, "object": "", "destination": ""}]
                if not args.no_plan:
                    try:
                        subgoals = planner.decompose(instruction, frame, lessons)
                    except Exception as e:  # noqa: BLE001 — planner down: degrade to flat
                        print(f"decompose unavailable ({e}); running flat")
                print(f"[{suite}/{task_id} ep{ep}] {instruction!r} -> {len(subgoals)} subgoal(s)")

                success, step, k, sub_steps, retried = False, 0, 0, 0, False
                fail_note = ""
                supports_effort = hasattr(cfg, "effort_action_index")
                while step < max_steps and not success:
                    sub = subgoals[min(k, len(subgoals) - 1)]
                    instruction_k = sub["instruction"]
                    if retried and sub.get("grasp"):
                        # the plain phrasing already failed once — steer the grasp via language
                        instruction_k = f"{instruction_k}, gripping it by {sub['grasp']}"
                    obs["task"] = [instruction_k]
                    if supports_effort and sub.get("effort") is not None:
                        obs["effort"] = torch.tensor([float(sub["effort"])])
                    batch = pre(obs)
                    with torch.inference_mode():
                        action = post(policy.select_action(batch))
                    action = env_post({ACTION: action})[ACTION]
                    raw_obs, _, terminated, truncated, info = env.step(action.cpu().numpy())
                    obs = env_pre(preprocess_observation(raw_obs))
                    step, sub_steps = step + 1, sub_steps + 1

                    final = info.get("final_info", info)
                    flag = final.get("is_success")
                    if flag is not None and bool(np.asarray(flag).flatten()[0]):
                        success = True
                        break
                    if np.asarray(terminated | truncated).flatten()[0]:
                        break

                    if k < len(subgoals) - 1 and sub_steps % args.check_every == 0:
                        frame = save_frame(obs)
                        done, note = subgoal_done(sub, frame, args.grounder, not args.no_vlm_verify)
                        if done:
                            print(f"  subgoal {k} done after {sub_steps} steps: {note}")
                            k, sub_steps, retried = k + 1, 0, False
                            policy.reset()  # flush queued actions from the old subgoal
                    if sub_steps >= args.max_subgoal_steps:
                        if not retried:
                            print(f"  subgoal {k} stuck; retrying once")
                            retried, sub_steps = True, 0
                            policy.reset()
                        else:
                            fail_note = f"stuck twice at subgoal {k}: {sub['instruction']!r}"
                            break

                outcome = "success" if success else "fail"
                successes += success
                if not args.no_memory:
                    memory.log(
                        instruction=instruction, object=subgoals[min(k, len(subgoals) - 1)].get("object", ""),
                        destination=subgoals[min(k, len(subgoals) - 1)].get("destination", ""),
                        outcome=outcome, note=fail_note or f"finished at subgoal {k}/{len(subgoals)}",
                        suite=suite, task_id=int(task_id), episode=ep, mode=mode,
                    )
                print(f"  -> {outcome} ({step} steps)")

            rate = successes / args.n_episodes
            print(f"== {suite}/task{task_id} [{mode}]: {successes}/{args.n_episodes} = {rate:.0%}")
            results.append(
                {"suite": suite, "task_id": int(task_id), "mode": mode,
                 "n_samples": getattr(cfg, "n_samples", 1), "successes": successes,
                 "episodes": args.n_episodes}
            )
            RESULTS.write_text(json.dumps(results, indent=1))

    close_envs(envs)
    print("\n| mode | suite | task | success |")
    print("|---|---|---|---|")
    for r in results:
        print(f"| {r['mode']} (n={r['n_samples']}) | {r['suite']} | {r['task_id']} | {r['successes']}/{r['episodes']} |")


if __name__ == "__main__":
    main()

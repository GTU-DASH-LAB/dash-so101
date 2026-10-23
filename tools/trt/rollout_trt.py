#!/usr/bin/env python3
"""lerobot-rollout with TensorRT: same CLI, same strategies, same robot code and safety limits; the
policy's heavy network runs as a TensorRT engine (trt_accel.py).

    rollout_trt.py [--trt-act-precision fp16|fp32] [--trt-n-action-steps N] <lerobot-rollout args>

The engine is built (first run of a checkpoint, 30-80 s) or loaded from the cache right after the
policy is loaded and BEFORE the robot connects, so the arm is never waiting on a build.

Per policy type:
  act        whole network as one engine (fp16 by default: actions within 0.11 deg of PyTorch).
  molmoact2  action expert as one engine per flow step; plus two per-tick fixes that do not change
             any action: the per-tick self.eval() walk (~5 ms) is skipped when already in eval
             mode, and ticks that only pop the next queued action skip the preprocessor (~6 ms:
             tokenizer + image processor; the observation is not used on those ticks).
             With --inference.type=rtc the expert stays in PyTorch (RTC's guidance needs
             gradients through it); the per-tick fixes do not apply to RTC either.
  others     no engine: the run is plain lerobot-rollout (a warning says so).

MolmoAct2 ignores --policy.n_action_steps: its __init__ overwrites it with the base checkpoint's
metadata (30 for SO-100/101). This launcher reports that, keeps 30 (what every run so far
executed) unless --trt-n-action-steps N is given, and then really executes N per chunk.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import trt_accel  # noqa: E402

log = logging.getLogger("rollout_trt")


def _pop_opt(name, default=None):
    """Remove --name VALUE / --name=VALUE from sys.argv (lerobot's parser rejects unknown flags)."""
    for i, arg in enumerate(sys.argv):
        if arg == name and i + 1 < len(sys.argv):
            value = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            return value
        if arg.startswith(name + "="):
            del sys.argv[i]
            return arg.split("=", 1)[1]
    return default


def _fast_sync_get_action(engine, policy):
    """SyncInferenceEngine.get_action, minus prepare/preprocess on ticks that only pop the queue.

    Same result: select_action pops _action_queue without looking at the observation when the
    queue is not empty, and the postprocessor and the action reordering below are the engine's own.
    """
    from lerobot.policies.utils import make_robot_action

    orig = engine.get_action

    def get_action(obs_frame):
        if obs_frame is None or len(policy._action_queue) == 0:
            return orig(obs_frame)
        with torch.inference_mode():
            action = engine._postprocessor(policy._action_queue.popleft())
        action_dict = make_robot_action(action.squeeze(0).cpu(), engine._dataset_features)
        return torch.tensor([action_dict[k] for k in engine._ordered_action_keys])

    engine.get_action = get_action


def install(precision="fp16", n_steps=None):
    """Patch lerobot so build_rollout_context loads the policy with TensorRT. Returns the patched
    build_rollout_context (also installed as lerobot_rollout.build_rollout_context)."""
    import lerobot.rollout.context as C
    from lerobot.rollout.inference.factory import RTCInferenceConfig
    from lerobot.rollout.inference.sync import SyncInferenceEngine
    from lerobot.scripts import lerobot_rollout as LR

    state = {}
    orig_build = LR.build_rollout_context
    orig_get_class = C.get_policy_class

    def get_policy_class(name):
        cls = orig_get_class(name)
        if getattr(cls, "_trt_wrapped", False):
            return cls
        orig_fp = cls.from_pretrained.__func__

        def from_pretrained(klass, *args, **kw):
            policy = orig_fp(klass, *args, **kw)
            cfg = state["cfg"]
            if cfg.policy.use_peft:
                log.warning("TensorRT: LoRA-adapter (use_peft) checkpoints are not covered - plain PyTorch")
                return policy
            policy = policy.to(cfg.policy.device).eval()
            try:
                info = trt_accel.accelerate(policy, act_precision=precision)
                log.info("TensorRT: %s", info)
                state["accelerated"] = True
            except NotImplementedError as e:
                log.warning("TensorRT: %s - running plain PyTorch", e)
            return policy

        cls.from_pretrained = classmethod(from_pretrained)
        cls._trt_wrapped = True
        return cls

    def build(cfg, *args, **kw):
        state["cfg"] = cfg
        cli_n = getattr(cfg.policy, "n_action_steps", None)
        ctx = orig_build(cfg, *args, **kw)
        policy = ctx.policy.policy
        if policy.config.type == "molmoact2":
            effective = policy.config.n_action_steps
            if n_steps is not None:
                trt_accel.set_n_action_steps(policy, n_steps)
                log.warning("MolmoAct2: executing %s of each %s-action chunk (--trt-n-action-steps)",
                            n_steps, policy.config.chunk_size)
            elif cli_n is not None and cli_n != effective:
                log.warning("MolmoAct2 ignores --policy.n_action_steps=%s and executes %s of each chunk "
                            "(its checkpoint metadata). Kept at %s; --trt-n-action-steps N overrides.",
                            cli_n, effective, effective)
            if isinstance(cfg.inference, RTCInferenceConfig):
                log.warning("TensorRT: RTC guidance needs gradients through the action expert - "
                            "with --inference.type=rtc the expert runs in PyTorch")
            elif isinstance(ctx.policy.inference, SyncInferenceEngine):
                _fast_sync_get_action(ctx.policy.inference, policy)
                log.info("Per-tick: queue-pop ticks skip the preprocessor")
        return ctx

    C.get_policy_class = get_policy_class
    LR.build_rollout_context = build
    return build


def main():
    install(_pop_opt("--trt-act-precision", "fp16"), _pop_opt("--trt-n-action-steps"))
    from lerobot.scripts import lerobot_rollout as LR

    LR.main()


if __name__ == "__main__":
    main()

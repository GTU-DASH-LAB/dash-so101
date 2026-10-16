#!/usr/bin/env python3
"""Make a trained checkpoint deployable on THIS arm: switch the joint-convention transform ON.

    deploy_config.py <checkpoint pretrained_model dir> <out dir> [--hub-base]

Training ran in MolmoAct2's "model frame" (make_model_frame_dataset.py) with the transform
OFF. On the robot, observations arrive in the arm frame and actions must go back in it, so
the deployable copy sets joint_signs/joint_offsets in config.json AND in the saved
pre/post-processor steps (the processors are loaded from their own JSON, not from the config):
  preprocessor:  state_model  = sign * state_arm + offset     (before normalizing)
  postprocessor: action_arm   = (action_model - offset) / sign (after unnormalizing)

Weights are hard-linked, not copied. --hub-base also points checkpoint_path and the FAST action
tokenizer at their Hub ids instead of this machine's local downloads (for publishing).
"""

import json
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    src, out = Path(args[0]), Path(args[1])
    signs, offsets = json.loads(os.environ["JOINT_SIGNS"]), json.loads(os.environ["JOINT_OFFSETS"])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file():
            if f.suffix == ".json":
                shutil.copy2(f, out / f.name)
            else:
                os.link(f, out / f.name)

    cfg = json.loads((out / "config.json").read_text())
    cfg["joint_signs"], cfg["joint_offsets"] = [float(x) for x in signs], [float(x) for x in offsets]
    if "--hub-base" in sys.argv:
        cfg["checkpoint_path"] = os.environ.get("BASE_CHECKPOINT_HUB", "allenai/MolmoAct2-SO100_101")
        cfg["discrete_action_tokenizer"] = "allenai/MolmoAct2-FAST-Tokenizer"
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    touched = 0
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        p = out / name
        if not p.exists():
            continue
        pipe = json.loads(p.read_text())
        for step in pipe.get("steps", []):
            c = step.get("config") or {}
            if "joint_signs" in c and "joint_offsets" in c:
                c["joint_signs"], c["joint_offsets"] = signs, offsets
                touched += 1
            if "--hub-base" in sys.argv and "checkpoint_path" in c:
                c["checkpoint_path"] = cfg["checkpoint_path"]
            if "--hub-base" in sys.argv and "discrete_action_tokenizer" in c:
                c["discrete_action_tokenizer"] = cfg["discrete_action_tokenizer"]
        p.write_text(json.dumps(pipe, indent=2))
    if touched != 2:
        print(f"Expected 2 frame-transform steps (state in, action out), found {touched}", file=sys.stderr)
        return 1
    print(f"Deployable checkpoint: {out}  (joint transform ON: signs {signs}, offsets {offsets})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

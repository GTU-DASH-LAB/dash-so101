#!/usr/bin/env python3
"""Make a FLUX 3 LoRA checkpoint (trained on AWS) loadable on any machine.

    deploy_config.py <checkpoint dir, e.g. outputs/flux3_pen_bowl/checkpoints/006000> <out dir>

Takes the EMA weights (`pretrained_model_ema`, what upstream evaluates; falls back to
`pretrained_model`). The LoRA adapter records its base as the path it was trained from,
which is /home/ubuntu/work/init_policy on the AWS machine. That init is the Hub checkpoint
black-forest-labs/flux-3-action-so101 with the SAME weights (prepare_init.py changes only
config.json and the processors, and those travel with this checkpoint). So the deployable
copy names the Hub id as its base: lerobot loads the base weights from there (offline from
the cache after local_weights.py) with this checkpoint's own config and processors.

Weights are hard-linked, not copied. The same copy is what 2b_publish_model.sh uploads.
"""

import json
import os
import shutil
import sys
from pathlib import Path

BASE_HUB = "black-forest-labs/flux-3-action-so101"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    ck, out = Path(sys.argv[1]), Path(sys.argv[2])
    src = next((ck / d for d in ("pretrained_model_ema", "pretrained_model") if (ck / d / "adapter_config.json").is_file()), None)
    if src is None:
        print(f"No LoRA adapter under {ck} (pretrained_model_ema/ or pretrained_model/)", file=sys.stderr)
        return 1
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file():
            (shutil.copy2 if f.suffix == ".json" else os.link)(f, out / f.name)

    ad = json.loads((out / "adapter_config.json").read_text())
    old = ad.get("base_model_name_or_path")
    ad["base_model_name_or_path"], ad["revision"] = BASE_HUB, None
    (out / "adapter_config.json").write_text(json.dumps(ad, indent=2))
    cfg = json.loads((out / "config.json").read_text())
    if not cfg.get("use_peft"):
        print("config.json lacks use_peft=true; lerobot would ignore the adapter", file=sys.stderr)
        return 1
    if "pretrained_path" in cfg:
        cfg["pretrained_path"] = BASE_HUB
    cfg["device"] = "cuda"
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Deployable checkpoint: {out}  ({src.name}; base {old} -> {BASE_HUB})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

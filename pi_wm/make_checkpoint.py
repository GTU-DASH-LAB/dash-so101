"""Create a local pi-WM checkpoint from the published pi0.5 LIBERO weights.

Downloads lerobot/pi05_libero_finetuned (~7GB — GX10 only!) and rewrites the config
type to "pi_wm" so from_pretrained instantiates PiWMPolicy. Weights and processor
pipelines are untouched: pi-WM is weights-identical to pi0.5.

    python pi_wm/make_checkpoint.py --out pi_wm_checkpoint
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--out", default="pi_wm_checkpoint")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(args.source, local_dir=args.out))
    config_file = path / "config.json"
    config = json.loads(config_file.read_text())
    config["type"] = "pi_wm"
    config_file.write_text(json.dumps(config, indent=2))
    print(f"pi-WM checkpoint ready at {path} (type: {config.get('type')})")
    print("next: PYTHONPATH=. python pi_wm/train_scorer.py --pi05-path", args.out)


if __name__ == "__main__":
    main()

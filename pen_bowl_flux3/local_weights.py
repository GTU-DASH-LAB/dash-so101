#!/usr/bin/env python3
"""Install the FLUX 3 weights for running on THIS PC into the Hugging Face cache.

    local_weights.py [--dl ~/models/flux3/dl]

The Hub downloader stalls on this network, so the four big files (25.3 GB) are fetched with
aria2c (~/models/flux3/dl/fetch.sh). This script checks each one's sha256 against the Hub,
moves it into the cache as the blob the Hub client expects, and lets the Hub client fetch
the small files and make the snapshot links. After that, the Hub ids in the configs
(black-forest-labs/flux-3-action-so101, flux-3-action-base:text_encoder@<rev>, ...) resolve
offline, for a local checkpoint and for the published model alike. Then it builds the local
wrist-only init_policy (prepare_init.py), the base that the LoRA checkpoints load onto.

Not needed: flux-3-action-base.safetensors (14 GB), the training-from-scratch trunk.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

SO101, SO101_REV = "black-forest-labs/flux-3-action-so101", "main"
BASE, BASE_REV = "black-forest-labs/flux-3-action-base", "62878e2925e59b7a89ec14463ce89932624c490d"
BIG = {  # aria2c output name -> (repo, revision, file in repo)
    "so101_model.safetensors": (SO101, SO101_REV, "model.safetensors"),
    "base_model-00001-of-00002.safetensors": (BASE, BASE_REV, "text_encoder/model-00001-of-00002.safetensors"),
    "base_model-00002-of-00002.safetensors": (BASE, BASE_REV, "text_encoder/model-00002-of-00002.safetensors"),
    "base_video_vae.safetensors": (BASE, BASE_REV, "video_vae.safetensors"),
}
PATTERNS = {SO101: ["*.json", "*.safetensors", "LICENSE.md"], BASE: ["text_encoder/*", "video_vae.safetensors"]}
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", default="~/models/flux3/dl")
    a = ap.parse_args()
    dl = Path(a.dl).expanduser()
    api = HfApi()

    for name, (repo, rev, fname) in BIG.items():
        meta = {s.rfilename: s for s in api.model_info(repo, revision=rev, files_metadata=True).siblings}[fname]
        want = meta.lfs.sha256
        blob = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "blobs" / want
        if blob.is_file() and blob.stat().st_size == meta.size:
            print(f"in cache  {repo}/{fname}")
            continue
        src = dl / name
        if not src.is_file() or (dl / f"{name}.aria2").exists():
            print(f"not downloaded yet: {src} (run {dl}/fetch.sh)", file=sys.stderr)
            return 1
        print(f"checking  {name} ({meta.size / 1e9:.1f} GB) ...", flush=True)
        got = sha256(src)
        if got != want:
            print(f"sha256 mismatch for {name}: {got} != {want}. Delete it and download again.", file=sys.stderr)
            return 1
        blob.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, blob)
        print(f"installed {repo}/{fname}")

    for repo, rev in ((SO101, SO101_REV), (BASE, BASE_REV)):
        path = snapshot_download(repo, revision=rev, allow_patterns=PATTERNS[repo])
        print(f"snapshot  {repo}@{rev[:8]} -> {path}")

    so101_dir = snapshot_download(SO101, revision=SO101_REV, allow_patterns=PATTERNS[SO101])
    env = dict(os.environ)
    rc = subprocess.run(["bash", "-c", f'set -a; source "{HERE}/config.env"; set +a; exec "{sys.executable}" '
                         f'"{HERE}/prepare_init.py" --ckpt "{so101_dir}" --out "{HERE}/init_policy"'], env=env).returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())

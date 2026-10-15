#!/usr/bin/env python3
"""Write a copy of the dataset in MolmoAct2-SO100_101's joint convention ("model frame").

The checkpoint learned SO-100/101 joints in the pre-LeRobot-0.5 calibration. Ours (LeRobot
>= 0.5) differs by  model = sign * arm + offset  with signs [1,-1,1,1,1,1] and offsets
[0,90,90,0,0,0] (docs/source/molmoact2.mdx). lerobot's MolmoAct2 processor can apply that to
the STATE only, and only before normalizing with this dataset's statistics, which are in the
arm frame; the action targets are never converted. Used for fine-tuning, that would leave
shoulder_lift and elbow_flex normalized ~90 deg off and the state and action conventions
mismatched.

So the data itself is converted once (state AND action, and their statistics, exactly), the
policy is trained in the model frame with the processor transform OFF, and the transform is
switched ON only for deployment (deploy_config.py): robot state -> model frame in, model
actions -> arm frame out.

Videos are shared via a symlink (unchanged); only parquet and meta are rewritten (~3 MB).
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KEYS = ("observation.state", "action")
QPAIRS = {"q01": "q99", "q10": "q90", "q50": "q50", "q90": "q10", "q99": "q01"}


def transform_values(arr: np.ndarray, s: np.ndarray, o: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float64).copy()
    n = len(s)
    out[..., :n] = s * out[..., :n] + o
    return out


def transform_stats(st: dict, s: np.ndarray, o: np.ndarray) -> dict:
    """Exact affine transform of per-dimension stats; a negative sign swaps min/max and q_k/q_(100-k)."""
    n = len(s)
    g = {k: np.asarray(v, dtype=np.float64) for k, v in st.items()}
    out = {k: v.copy() for k, v in g.items()}
    neg = s < 0
    for k in ("mean", "min", "max", *QPAIRS):
        if k in g:
            out[k][:n] = s * g[k][:n] + o
    for a, b in (("min", "max"), ("max", "min"), *QPAIRS.items()):
        if a in g and b in g:
            out[a][:n] = np.where(neg, s * g[b][:n] + o, out[a][:n])
    if "std" in g:
        out["std"][:n] = np.abs(s) * g["std"][:n]
    return {k: v.tolist() for k, v in out.items()}


def main() -> int:
    src = Path(os.environ["DATASET_ROOT"]).expanduser()
    dst = Path(os.environ["MODEL_FRAME_DATASET_ROOT"]).expanduser()
    s = np.asarray(json.loads(os.environ["JOINT_SIGNS"]), dtype=np.float64)
    o = np.asarray(json.loads(os.environ["JOINT_OFFSETS"]), dtype=np.float64)
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)
    (dst / "videos").symlink_to(src / "videos")
    shutil.copy2(src / "meta" / "info.json", dst / "meta" / "info.json")
    shutil.copy2(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")

    stats = json.loads((src / "meta" / "stats.json").read_text())
    for k in KEYS:
        stats[k] = transform_stats(stats[k], s, o)
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    # Per-episode stats live as columns "stats/<key>/<stat>" in meta/episodes.
    for p in sorted((src / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(p)
        cols = {}
        for k in KEYS:
            names = [c for c in t.column_names if c.startswith(f"stats/{k}/")]
            per = {c.split("/")[-1]: t.column(c).to_pylist() for c in names}
            if not per:
                continue
            rows = [transform_stats({st: per[st][i] for st in per}, s, o) for i in range(t.num_rows)]
            for st in per:
                cols[f"stats/{k}/{st}"] = [r[st] for r in rows]
        for c, vals in cols.items():
            t = t.set_column(t.column_names.index(c), c, pa.array(vals, type=t.schema.field(c).type))
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)

    n_frames = 0
    for p in sorted((src / "data").rglob("*.parquet")):
        t = pq.read_table(p)
        for k in KEYS:
            arr = np.stack(t.column(k).to_numpy(zero_copy_only=False))
            new = transform_values(arr, s, o).astype(np.float32)
            t = t.set_column(t.column_names.index(k), k,
                             pa.array(list(new), type=t.schema.field(k).type))
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)
        n_frames += t.num_rows

    q = stats["observation.state"]
    print(f"Model-frame dataset: {dst} ({n_frames} frames)")
    print(f"  state q01 {[round(x, 1) for x in q['q01']]}")
    print(f"  state q99 {[round(x, 1) for x in q['q99']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

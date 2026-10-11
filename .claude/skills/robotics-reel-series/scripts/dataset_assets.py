"""Turn a LeRobot v3 dataset into reel material: real stats, a demo-montage grid, single-episode clips.

Needs pandas (the lelab tool venv has it):
  PY=~/.local/share/uv/tools/lelab/bin/python

  $PY dataset_assets.py stats --root ROOT
      -> JSON: episodes, frames, fps, hours, cameras, episode-length range (use these numbers on screen;
         never invent stats)
  $PY dataset_assets.py grid  --root ROOT --camera observation.images.wrist --n 21 --secs 9 --out grid.mp4
      -> 1080x1920 montage of N demos (3 columns, 360x270 tiles), silent H.264. Day 1 used n=21, secs=9.
  $PY dataset_assets.py clip  --root ROOT --camera observation.images.wrist --episode 5 --secs 14 --out clip.mp4
      -> one episode cut from its packed video file (pick a clean grasp; check with a contact sheet first)
  $PY dataset_assets.py sheet --root ROOT --camera observation.images.wrist --episode 5 --out sheet.jpg
      -> 3x3 contact sheet of one episode, to choose which episode to clip

ROOT is the local dataset dir, e.g. ~/.cache/huggingface/lerobot/fouad1233/so101_pen_in_bowl_wrist
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys

import pandas as pd


def load(root):
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet")))
    if not files:
        sys.exit(f"no episode metadata under {root}/meta/episodes (is this a LeRobot v3 dataset?)")
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index")
    return info, df


def video_of(root, info, row, cam):
    k = f"videos/{cam}/"
    path = info["video_path"].format(video_key=cam, chunk_index=int(row[k + "chunk_index"]),
                                     file_index=int(row[k + "file_index"]))
    return os.path.join(root, path), float(row[k + "from_timestamp"]), float(row[k + "to_timestamp"])


def run(cmd):
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"ffmpeg failed with exit {r.returncode}")


def cmd_stats(a):
    info, df = load(a.root)
    fps = info["fps"]
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    frames = int(df["length"].sum())
    print(json.dumps({
        "episodes": int(len(df)), "frames": frames, "fps": fps,
        "hours": round(frames / fps / 3600, 2), "minutes": round(frames / fps / 60, 1),
        "cameras": cams,
        "episode_seconds": {"min": round(df["length"].min() / fps, 1), "mean": round(df["length"].mean() / fps, 1),
                            "max": round(df["length"].max() / fps, 1)},
        "tasks": sorted({t for ts in df["tasks"] for t in (ts if hasattr(ts, "__iter__") and not isinstance(ts, str) else [ts])}),
    }, indent=2, default=str))


def cmd_grid(a):
    info, df = load(a.root)
    fps = info["fps"]
    pool = df[df["length"] > a.secs * fps]  # only episodes long enough to fill the tile
    sel = pool.iloc[::max(1, len(pool) // a.n)].head(a.n)
    tw, th, cols = 360, 270, 3
    rows = math.ceil(len(sel) / cols)
    inputs, filt, layout = [], "", []
    for i, (_, r) in enumerate(sel.iterrows()):
        path, t0, _ = video_of(a.root, info, r, a.camera)
        inputs += ["-ss", f"{t0:.3f}", "-t", str(a.secs), "-i", path]
        filt += f"[{i}:v]scale={tw}:{th},setsar=1,fps=30[v{i}];"
        layout.append(f"{(i % cols) * tw}_{(i // cols) * th}")
    stack = "".join(f"[v{i}]" for i in range(len(sel)))
    pad_top = max(0, (1920 - rows * th) // 2)
    filt += (f"{stack}xstack=inputs={len(sel)}:layout={'|'.join(layout)}:fill=black,"
             f"pad=1080:1920:0:{pad_top}:black,format=yuv420p[out]")
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", filt, "-map", "[out]", "-t", str(a.secs),
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-an", a.out])
    print(f"grid: {a.out}  tiles={len(sel)}  episodes={[int(e) for e in sel['episode_index']]}")


def _episode(df, ep):
    m = df[df["episode_index"] == ep]
    if m.empty:
        sys.exit(f"episode {ep} not found")
    return m.iloc[0]


def cmd_clip(a):
    info, df = load(a.root)
    path, t0, t1 = video_of(a.root, info, _episode(df, a.episode), a.camera)
    secs = min(a.secs, t1 - t0 - a.offset)
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0 + a.offset:.3f}", "-t", f"{secs:.3f}", "-i", path,
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-an", a.out])
    print(f"clip: {a.out}  episode={a.episode}  {secs:.1f}s from {os.path.basename(path)}@{t0 + a.offset:.2f}s")


def cmd_sheet(a):
    info, df = load(a.root)
    path, t0, t1 = video_of(a.root, info, _episode(df, a.episode), a.camera)
    dur = t1 - t0
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", path,
         "-vf", f"fps={9 / dur:.4f},scale=320:-1,tile=3x3", "-frames:v", "1", a.out])
    print(f"sheet: {a.out}  episode={a.episode}  {dur:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats"); s.add_argument("--root", required=True); s.set_defaults(fn=cmd_stats)
    g = sub.add_parser("grid"); g.add_argument("--root", required=True); g.add_argument("--camera", required=True)
    g.add_argument("--n", type=int, default=21); g.add_argument("--secs", type=float, default=9)
    g.add_argument("--out", required=True); g.set_defaults(fn=cmd_grid)
    c = sub.add_parser("clip"); c.add_argument("--root", required=True); c.add_argument("--camera", required=True)
    c.add_argument("--episode", type=int, required=True); c.add_argument("--offset", type=float, default=0)
    c.add_argument("--secs", type=float, default=14); c.add_argument("--out", required=True); c.set_defaults(fn=cmd_clip)
    h = sub.add_parser("sheet"); h.add_argument("--root", required=True); h.add_argument("--camera", required=True)
    h.add_argument("--episode", type=int, required=True); h.add_argument("--out", required=True); h.set_defaults(fn=cmd_sheet)
    a = ap.parse_args()
    a.root = os.path.expanduser(a.root)
    a.fn(a)


if __name__ == "__main__":
    main()

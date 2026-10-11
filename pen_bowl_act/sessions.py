#!/usr/bin/env python3
"""Session bookkeeping for pen_bowl_act. Called by 1_record.sh, manage_data.sh and
1b_publish_dataset.sh. Run `./manage_data.sh help` for the human-facing commands.

Layout (SESSIONS_DIR from config.env):
    session_01/dataset/        a complete LeRobotDataset (one recording session)
    session_01/session.json    status, target, timestamps
    session_01/episodes.jsonl  one line per saved demo
    session_01/record.log      full log of every recording run of that session
    _trash/<time>__session_01  deleted sessions / pre-edit copies (restorable)

Nothing here ever deletes data except `empty-trash`. Everything else MOVES data
into _trash first, so every delete can be undone with `restore`.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TTY = sys.stdout.isatty()
RED, GRN, YEL, DIM, OFF = ("\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[2m", "\033[0m") if TTY else ("",) * 5


def env(name: str) -> str:
    v = os.environ.get(name)
    if v is None:
        sys.exit(f"{name} is not set; run this through the .sh wrappers (they source config.env)")
    return v


SESSIONS_DIR = Path(env("SESSIONS_DIR"))
TRASH = SESSIONS_DIR / "_trash"
FPS = int(os.environ.get("FPS", "30"))
MIN_EPISODE_S = float(os.environ.get("MIN_EPISODE_S", "3"))


def session_dirs() -> list[Path]:
    if not SESSIONS_DIR.is_dir():
        return []
    return sorted(p for p in SESSIONS_DIR.iterdir() if p.is_dir() and p.name.startswith("session_"))


def session_number(p: Path) -> int:
    return int(p.name.split("_")[1])


def resolve_session(name: str) -> Path:
    """Accept '3', '03', 'session_03'."""
    if name.isdigit():
        name = f"session_{int(name):02d}"
    p = SESSIONS_DIR / name
    if not (p.is_dir() and p.name.startswith("session_")):
        known = ", ".join(s.name for s in session_dirs()) or "none"
        sys.exit(f"{RED}No session '{name}'.{OFF} Sessions: {known}")
    return p


def parquet_ok(path: Path) -> bool:
    """A parquet file whose writer was never closed (hard crash) has no footer."""
    import pyarrow.parquet as pq

    try:
        pq.ParquetFile(path).metadata
        return True
    except Exception:  # noqa: BLE001
        return False


def inspect(sess: Path) -> dict:
    """Cheap health check from files only (no video decoding)."""
    root = sess / "dataset"
    meta = json.loads((sess / "session.json").read_text()) if (sess / "session.json").exists() else {}
    info = {"name": sess.name, "path": str(sess), "episodes": 0, "frames": 0, "status": meta.get("status", "?"),
            "healthy": True, "problem": "", "lengths": []}
    info_json = root / "meta" / "info.json"
    if not info_json.is_file():
        info.update(healthy=False, problem="no dataset (recording never saved anything)")
        return info
    bad = [p for p in list(root.glob("data/**/*.parquet")) + list(root.glob("meta/episodes/**/*.parquet"))
           if not parquet_ok(p)]
    if bad:
        info.update(healthy=False, problem=f"unreadable file(s) from a hard crash: {[str(b.relative_to(root)) for b in bad]}")
        return info
    i = json.loads(info_json.read_text())
    info["episodes"], info["frames"] = i["total_episodes"], i["total_frames"]
    if info["episodes"] == 0:
        info.update(healthy=False, problem="dataset exists but has 0 demos")
        return info
    try:
        import pyarrow.parquet as pq

        tables = [pq.read_table(p, columns=["episode_index", "length"])
                  for p in sorted(root.glob("meta/episodes/**/*.parquet"))]
        lengths = sorted((r["episode_index"], r["length"]) for t in tables for r in t.to_pylist())
        info["lengths"] = [n for _, n in lengths]
        if len(lengths) != info["episodes"]:
            info.update(healthy=False, problem=f"info.json says {info['episodes']} demos, episode index has {len(lengths)}")
    except Exception as e:  # noqa: BLE001
        info.update(healthy=False, problem=f"episode index unreadable: {e}")
    return info


def move_to_trash(path: Path, label: str) -> Path:
    TRASH.mkdir(parents=True, exist_ok=True)
    dest = TRASH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__{label}"
    shutil.move(str(path), str(dest))
    return dest


def load_dataset(sess: Path):
    from lerobot.datasets import LeRobotDataset

    meta = json.loads((sess / "session.json").read_text())
    return LeRobotDataset(meta["repo_id"], root=sess / "dataset")


# --------------------------------------------------------------------------- commands
def cmd_status(a) -> int:
    rows = [inspect(s) for s in session_dirs()]
    healthy = [r for r in rows if r["healthy"]]
    total = sum(r["episodes"] for r in healthy)
    target = int(os.environ.get("TARGET_EPISODES", "100"))
    if a.json:
        print(json.dumps({"sessions": rows, "total_episodes": total, "target": target}))
        return 0
    if not rows:
        print("No sessions recorded yet. Start with ./1_record.sh")
        return 0
    print(f"{'session':<12}{'demos':>6}{'minutes':>9}  {'short':>5}  status")
    for r in rows:
        mins = r["frames"] / FPS / 60
        short = sum(1 for n in r["lengths"] if n < MIN_EPISODE_S * FPS)
        if r["healthy"]:
            state = f"{GRN}ok{OFF} ({r['status']})"
        else:
            state = f"{RED}BROKEN{OFF}: {r['problem']}"
        print(f"{r['name']:<12}{r['episodes']:>6}{mins:>9.1f}  {short:>5}  {state}")
    bar_n = int(30 * min(total, target) / target)
    print(f"\nTotal usable demos: {total}/{target}  [{'#' * bar_n}{'.' * (30 - bar_n)}]")
    if any(not r["healthy"] for r in rows):
        print(f"{YEL}A BROKEN session cannot be used. Move it to the trash with: "
              f"./manage_data.sh delete-session <name>{OFF}")
    if TRASH.is_dir() and any(TRASH.iterdir()):
        print(f"{DIM}Trash has {len(list(TRASH.iterdir()))} item(s): ./manage_data.sh trash{OFF}")
    return 0


def cmd_next(a) -> int:
    """Decide which session 1_record.sh records into. Prints: NAME TARGET TOTAL_BEFORE."""
    rows = [inspect(s) for s in session_dirs()]
    total = sum(r["episodes"] for r in rows if r["healthy"])
    if total >= a.target:
        print(f"DONE - - {total}")
        return 0
    last = rows[-1] if rows else None
    if last is not None and not last["healthy"] and last["problem"].startswith("unreadable"):
        print(f"BROKEN {last['name']} - {total}")
        return 0
    if last is not None and last["episodes"] < a.per_session and (
        last["healthy"] or last["problem"].startswith("no dataset") or "0 demos" in last["problem"]
    ):
        before = total - (last["episodes"] if last["healthy"] else 0)
        sess_target = min(a.per_session, last["episodes"] + (a.target - total))
        print(f"{last['name']} {sess_target} {before}")
        return 0
    nxt = (session_number(Path(rows[-1]["path"])) + 1) if rows else 1
    print(f"session_{nxt:02d} {min(a.per_session, a.target - total)} {total}")
    return 0


def cmd_episodes(a) -> int:
    sess = resolve_session(a.session)
    r = inspect(sess)
    if not r["healthy"]:
        sys.exit(f"{RED}{sess.name} is broken: {r['problem']}{OFF}")
    log = {}
    if (sess / "episodes.jsonl").exists():
        for line in (sess / "episodes.jsonl").read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                log[d["episode_index"]] = d
    print(f"{sess.name}: {r['episodes']} demos")
    for i, n in enumerate(r["lengths"]):
        flag = f"  {YEL}<- very short{OFF}" if n < MIN_EPISODE_S * FPS else ""
        when = log.get(i, {}).get("saved_at", "")
        print(f"  demo {i:>2}: {n / FPS:5.1f} s  {DIM}{when}{OFF}{flag}")
    return 0


def cmd_review(a) -> int:
    """Cut each demo out of the session video into its own mp4 and open it."""
    sess = resolve_session(a.session)
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = json.loads((sess / "session.json").read_text())
    m = LeRobotDatasetMetadata(meta["repo_id"], root=sess / "dataset")
    key = m.video_keys[0]
    eps = range(m.total_episodes) if a.episode == "all" else [int(a.episode)]
    out_dir = Path(os.environ.get("HERE", ".")) / "review" / sess.name
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for ep in eps:
        if not 0 <= ep < m.total_episodes:
            sys.exit(f"{sess.name} has demos 0..{m.total_episodes - 1}")
        row = m.episodes[ep]
        src = sess / "dataset" / m.get_video_file_path(ep, key)
        t0, t1 = row[f"videos/{key}/from_timestamp"], row[f"videos/{key}/to_timestamp"]
        dst = out_dir / f"demo_{ep:02d}.mp4"
        # Re-encode (not -c copy) so the cut is frame-exact and plays everywhere.
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(src),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(dst)]
        subprocess.run(cmd, check=True)
        made.append(dst)
        print(f"  demo {ep:>2}: {t1 - t0:5.1f} s -> {dst}")
    if not a.no_open and os.environ.get("DISPLAY"):
        target = made[0] if len(made) == 1 else out_dir
        subprocess.Popen(["xdg-open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


def cmd_delete_episodes(a) -> int:
    sess = resolve_session(a.session)
    r = inspect(sess)
    if not r["healthy"]:
        sys.exit(f"{RED}{sess.name} is broken ({r['problem']}). Delete the whole session instead.{OFF}")
    idx = sorted(set(a.episodes))
    if any(not 0 <= i < r["episodes"] for i in idx):
        sys.exit(f"{sess.name} has demos 0..{r['episodes'] - 1}; got {idx}")
    if len(idx) == r["episodes"]:
        sys.exit("That is every demo in the session. Use delete-session instead.")
    if not a.yes:
        ans = input(f"Delete demo(s) {idx} from {sess.name} ({r['episodes']} -> {r['episodes'] - len(idx)})? "
                    f"The original goes to the trash. [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            print("Nothing changed.")
            return 1
    from lerobot.datasets.dataset_tools import delete_episodes

    meta = json.loads((sess / "session.json").read_text())
    tmp = sess / "dataset.new"
    if tmp.exists():
        shutil.rmtree(tmp)  # leftover from an interrupted earlier edit; never user data
    new = delete_episodes(load_dataset(sess), episode_indices=idx, output_dir=tmp, repo_id=meta["repo_id"])
    n_new = new.meta.total_episodes
    if n_new != r["episodes"] - len(idx):
        shutil.rmtree(tmp)
        sys.exit(f"{RED}Edit produced {n_new} demos, expected {r['episodes'] - len(idx)}. Nothing changed.{OFF}")
    # Swap: original -> trash (restorable), edited copy -> in place.
    backup = TRASH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__{sess.name}__before_deleting_{'-'.join(map(str, idx))}"
    TRASH.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sess, backup, ignore=shutil.ignore_patterns("dataset.new"))
    shutil.rmtree(sess / "dataset")
    tmp.rename(sess / "dataset")
    # Keep episodes.jsonl aligned with the new (re-numbered) episode indices.
    if (sess / "episodes.jsonl").exists():
        kept = [json.loads(line) for line in (sess / "episodes.jsonl").read_text().splitlines() if line.strip()]
        kept = [d for d in kept if d["episode_index"] not in idx]
        for new_i, d in enumerate(sorted(kept, key=lambda d: d["episode_index"])):
            d["episode_index"] = new_i
        (sess / "episodes.jsonl").write_text("".join(json.dumps(d) + "\n" for d in kept))
    meta["episodes"] = n_new
    meta.setdefault("edits", []).append({"deleted": idx, "at": datetime.now().isoformat(timespec="seconds")})
    (sess / "session.json").write_text(json.dumps(meta, indent=2))
    print(f"{GRN}Done.{OFF} {sess.name} now has {n_new} demos (renumbered 0..{n_new - 1}).")
    print(f"Undo with: ./manage_data.sh restore {backup.name}")
    return 0


def cmd_delete_session(a) -> int:
    sess = resolve_session(a.session)
    r = inspect(sess)
    if not a.yes:
        ans = input(f"Move {sess.name} ({r['episodes']} demos) to the trash? [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            print("Nothing changed.")
            return 1
    dest = move_to_trash(sess, sess.name)
    print(f"{GRN}Moved to trash.{OFF} Undo with: ./manage_data.sh restore {dest.name}")
    print("The next ./1_record.sh will record a NEW session to replace these demos.")
    return 0


def cmd_trash(a) -> int:
    items = sorted(TRASH.iterdir()) if TRASH.is_dir() else []
    if not items:
        print("Trash is empty.")
        return 0
    for p in items:
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
        print(f"  {p.name}  ({size:.0f} MB)")
    print("Restore one with: ./manage_data.sh restore <name>   |   delete all forever: ./manage_data.sh empty-trash")
    return 0


def cmd_restore(a) -> int:
    src = TRASH / a.name
    if not src.is_dir():
        sys.exit(f"No trash item '{a.name}'. See ./manage_data.sh trash")
    sess_name = a.name.split("__")[1]
    dest = SESSIONS_DIR / sess_name
    if dest.exists():
        aside = move_to_trash(dest, f"{sess_name}__replaced_by_restore")
        print(f"Current {sess_name} moved to trash first: {aside.name}")
    shutil.move(str(src), str(dest))
    print(f"{GRN}Restored{OFF} {dest}")
    return 0


def cmd_empty_trash(a) -> int:
    if not TRASH.is_dir() or not any(TRASH.iterdir()):
        print("Trash is already empty.")
        return 0
    ans = input(f"PERMANENTLY delete everything in {TRASH}? Type 'yes': ")
    if ans.strip() != "yes":
        print("Nothing deleted.")
        return 1
    shutil.rmtree(TRASH)
    print("Trash emptied.")
    return 0


def cmd_check(a) -> int:
    """Deep check: load every session and decode the first and last frame of every demo."""
    import torch  # noqa: F401  (LeRobotDataset needs it; import early for a clear error)

    bad = 0
    for sess in session_dirs():
        r = inspect(sess)
        if not r["healthy"]:
            print(f"  {RED}BROKEN{OFF} {sess.name}: {r['problem']}")
            bad += 1
            continue
        try:
            ds = load_dataset(sess)
            for ep in range(ds.num_episodes):
                row = ds.meta.episodes[ep]
                for idx in (row["dataset_from_index"], row["dataset_to_index"] - 1):
                    ds[idx]
            print(f"  {GRN}ok{OFF}     {sess.name}: {ds.num_episodes} demos, every first/last frame decodes")
        except Exception as e:  # noqa: BLE001
            print(f"  {RED}BROKEN{OFF} {sess.name}: {type(e).__name__}: {e}")
            bad += 1
    print("All sessions healthy." if not bad else f"{RED}{bad} broken session(s).{OFF} Delete them with delete-session.")
    return 1 if bad else 0


def cmd_merge(a) -> int:
    """Merge all healthy sessions into one dataset at --out-root (must not exist)."""
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import merge_datasets

    out = Path(a.out_root)
    if out.exists():
        sys.exit(f"{out} already exists")
    sessions = [s for s in session_dirs() if inspect(s)["healthy"]]
    if not sessions:
        sys.exit("No healthy sessions to merge.")
    datasets = [load_dataset(s) for s in sessions]
    print(f"Merging {len(sessions)} sessions: {', '.join(s.name for s in sessions)}")
    merged = merge_datasets(datasets, output_repo_id=a.out_repo, output_dir=out)
    expected = sum(d.num_episodes for d in datasets)
    if merged.meta.total_episodes != expected:
        sys.exit(f"{RED}Merged dataset has {merged.meta.total_episodes} demos, expected {expected}.{OFF}")
    # Re-open from disk and decode the first and last frame of every demo.
    ds = LeRobotDataset(a.out_repo, root=out)
    for ep in range(ds.num_episodes):
        row = ds.meta.episodes[ep]
        ds[row["dataset_from_index"]]
        ds[row["dataset_to_index"] - 1]
    print(f"{GRN}Merged and verified:{OFF} {ds.num_episodes} demos, {ds.num_frames} frames "
          f"({ds.num_frames / FPS / 60:.1f} min) -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status"); p.add_argument("--json", action="store_true"); p.set_defaults(f=cmd_status)
    p = sub.add_parser("next"); p.add_argument("--per-session", type=int, required=True)
    p.add_argument("--target", type=int, required=True); p.set_defaults(f=cmd_next)
    p = sub.add_parser("episodes"); p.add_argument("session"); p.set_defaults(f=cmd_episodes)
    p = sub.add_parser("review"); p.add_argument("session"); p.add_argument("episode", nargs="?", default="all")
    p.add_argument("--no-open", action="store_true"); p.set_defaults(f=cmd_review)
    p = sub.add_parser("delete-episodes"); p.add_argument("session"); p.add_argument("episodes", type=int, nargs="+")
    p.add_argument("--yes", action="store_true"); p.set_defaults(f=cmd_delete_episodes)
    p = sub.add_parser("delete-session"); p.add_argument("session"); p.add_argument("--yes", action="store_true")
    p.set_defaults(f=cmd_delete_session)
    p = sub.add_parser("trash"); p.set_defaults(f=cmd_trash)
    p = sub.add_parser("restore"); p.add_argument("name"); p.set_defaults(f=cmd_restore)
    p = sub.add_parser("empty-trash"); p.set_defaults(f=cmd_empty_trash)
    p = sub.add_parser("check"); p.set_defaults(f=cmd_check)
    p = sub.add_parser("merge"); p.add_argument("--out-root", required=True); p.add_argument("--out-repo", required=True)
    p.set_defaults(f=cmd_merge)
    a = ap.parse_args()
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())

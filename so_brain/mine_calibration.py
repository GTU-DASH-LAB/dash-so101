"""Distill recorded demos into a classical controller calibration.

For each episode of the teleop dataset: locate the pen and the pad in the first
frame (OWLv2), find the grasp moment (gripper-close onset) and the release moment
(gripper re-open) in the joint series, and record the arm pose at each. Then fit
a quadratic regression pixel(u, v) -> joints for the grasp / approach / place
poses. The fit doubles as the camera-to-robot calibration for pid_pick.py: its
analytic Jacobian d(joints)/d(u, v) converts pixel error to joint corrections.

    PYTHONPATH=. .venv/bin/python so_brain/mine_calibration.py \
        --repo-id fouad1233/so101_pencil_pickup --pick "a pen" --place "a black mouse pad"

Writes so_brain/pid_calib.json.
"""

import argparse
import json

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
APPROACH_LEAD_S = 1.0  # "hover" pose = this long before the gripper starts closing


def episode_first_frames(repo_id: str) -> dict[int, np.ndarray]:
    """Decode frame 0 of every episode (RGB uint8) straight from the hub videos."""
    import av
    import pandas as pd
    from huggingface_hub import hf_hub_download

    meta = pd.read_parquet(
        hf_hub_download(repo_id, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset")
    )
    frames: dict[int, np.ndarray] = {}
    key = "videos/observation.images.front"
    for (chunk, file), group in meta.groupby([f"{key}/chunk_index", f"{key}/file_index"]):
        path = hf_hub_download(
            repo_id, f"{key}/chunk-{chunk:03d}/file-{file:03d}.mp4", repo_type="dataset"
        )
        container = av.open(path)
        stream = container.streams.video[0]
        wanted = sorted(
            (row[f"{key}/from_timestamp"], int(row["episode_index"])) for _, row in group.iterrows()
        )
        i = 0
        for frame in container.decode(stream):
            while i < len(wanted) and frame.time >= wanted[i][0] - 1e-3:
                frames[wanted[i][1]] = frame.to_ndarray(format="rgb24")
                i += 1
            if i >= len(wanted):
                break
        container.close()
    return frames


def grasp_release_indices(gripper: np.ndarray, fps: float) -> tuple[int, int]:
    """Frame indices where the gripper starts closing (grasp) and re-opens (release)."""
    # Demos rest with the gripper closed, open on approach, close onto the pen
    # (holding plateau), and reopen at release: closed -> OPEN -> hold -> OPEN -> closed.
    # The grasp is therefore the first downward mid-crossing AFTER the first open.
    open_level = np.percentile(gripper, 95)
    closed_level = np.percentile(gripper, 5)
    if open_level - closed_level < 5:
        raise ValueError("gripper barely moves in this episode")
    mid = (open_level + closed_level) / 2
    idx = np.arange(len(gripper))
    above = np.flatnonzero(gripper > mid)
    if len(above) == 0:
        raise ValueError("gripper never opens in this episode")
    t_open = above[0]
    below = np.flatnonzero((idx > t_open) & (gripper < mid))
    if len(below) == 0:
        raise ValueError("gripper never closes after opening")
    t_grasp = below[0]
    after = np.flatnonzero((idx > t_grasp) & (gripper > mid))
    t_release = after[0] if len(after) else len(gripper) - 1
    return int(t_grasp), int(t_release)


def features(uv: np.ndarray) -> np.ndarray:
    """Quadratic pixel features: [u, v, u^2, v^2, uv, 1]."""
    u, v = uv[..., 0], uv[..., 1]
    return np.stack([u, v, u * u, v * v, u * v, np.ones_like(u)], axis=-1)


def fit(uv: np.ndarray, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares W: features(u,v) @ W -> pose. Returns (W, per-joint RMS residual)."""
    X = features(uv)
    W, *_ = np.linalg.lstsq(X, poses, rcond=None)
    resid = poses - X @ W
    return W, np.sqrt((resid**2).mean(axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--pick", required=True, help="detector phrase for the object (as in relabel)")
    ap.add_argument("--place", required=True, help="detector phrase for the destination")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default="so_brain/pid_calib.json")
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    from so_brain import grounding

    df = pd.read_parquet(
        hf_hub_download(args.repo_id, "data/chunk-000/file-000.parquet", repo_type="dataset")
    )
    frames = episode_first_frames(args.repo_id)
    print(f"decoded first frames for {len(frames)} episodes")

    lead = int(APPROACH_LEAD_S * args.fps)
    pen_uv, pad_uv, grasp_pose, approach_pose, place_pose = [], [], [], [], []
    open_vals, closed_vals, home_pose = [], [], []
    skipped = []
    for ep, group in df.groupby("episode_index"):
        group = group.sort_values("frame_index")
        state = np.stack(group["observation.state"].to_numpy())
        action = np.stack(group["action"].to_numpy())
        try:
            u1, v1, s1 = grounding.locate(frames[int(ep)], args.pick)
            u2, v2, s2 = grounding.locate(frames[int(ep)], args.place)
            t_grasp, t_release = grasp_release_indices(state[:, 5], args.fps)
        except (LookupError, ValueError, KeyError) as e:
            skipped.append((int(ep), str(e)))
            continue
        pen_uv.append([u1, v1])
        pad_uv.append([u2, v2])
        grasp_pose.append(state[t_grasp, :5])
        approach_pose.append(state[max(t_grasp - lead, 0), :5])
        place_pose.append(state[t_release, :5])
        # Gripper set-points come from the ACTION channel (leader commands): the
        # follower never reaches them around a gripped object, and we command too.
        open_vals.append(np.percentile(action[:, 5], 95))
        closed_vals.append(action[t_grasp : t_release, 5].min())
        home_pose.append(state[0, :5])

    n = len(pen_uv)
    print(f"mined {n} episodes ({len(skipped)} skipped: {skipped[:5]})")
    if n < 15:
        raise SystemExit("too few episodes mined for a trustworthy fit — inspect skips")

    pen_uv, pad_uv = np.array(pen_uv), np.array(pad_uv)
    calib = {"joints": JOINTS[:5], "pick_phrase": args.pick, "place_phrase": args.place}
    for name, uv, poses in (
        ("grasp", pen_uv, np.array(grasp_pose)),
        ("approach", pen_uv, np.array(approach_pose)),
        ("place", pad_uv, np.array(place_pose)),
    ):
        W, rms = fit(uv, poses)
        calib[name] = {"W": W.tolist(), "rms": rms.tolist()}
        print(f"{name:9s} fit RMS per joint: " + "  ".join(f"{j}={r:.2f}" for j, r in zip(JOINTS, rms)))
    # The pad never moved across demos, so the place regression is degenerate (no
    # pixel signal). The median demonstrated release pose is the robust choice.
    calib["place_median"] = np.median(np.array(place_pose), axis=0).tolist()
    calib["gripper_open"] = float(np.median(open_vals))
    calib["gripper_closed"] = float(np.median(closed_vals))
    calib["home"] = np.median(np.array(home_pose), axis=0).tolist()
    # Demo-covered workspace, for runtime safety clamps (5th-95th pixel percentiles).
    calib["pen_uv_range"] = [
        np.percentile(pen_uv, 5, axis=0).tolist(),
        np.percentile(pen_uv, 95, axis=0).tolist(),
    ]
    joint_min = np.stack(df["observation.state"].to_numpy()).min(axis=0)[:5]
    joint_max = np.stack(df["observation.state"].to_numpy()).max(axis=0)[:5]
    calib["joint_min"], calib["joint_max"] = joint_min.tolist(), joint_max.tolist()
    # Raw per-episode samples, for downstream fits (e.g. URDF-based table homography).
    calib["samples"] = {
        "pen_uv": pen_uv.tolist(),
        "pad_uv": pad_uv.tolist(),
        "grasp_pose": np.array(grasp_pose).tolist(),
        "approach_pose": np.array(approach_pose).tolist(),
    }
    print(f"gripper open={calib['gripper_open']:.1f} closed={calib['gripper_closed']:.1f}")

    with open(args.out, "w") as f:
        json.dump(calib, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

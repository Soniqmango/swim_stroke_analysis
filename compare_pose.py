"""Compare pose models on the same clips: how often each one finds the swimmer.

    uv run python compare_pose.py VIDEO [VIDEO ...] [--force]

Runs MediaPipe (heavy) and RTMPose (the model behind Sports2D) on every
clip, caches each pose table as outputs/<clip>/pose_<model>.parquet, and
writes outputs/pose_compare/summary.csv with one row per clip x model.

Each model is scored at its own recommended confidence threshold
(MediaPipe 0.5, RTMPose 0.3 as in Sports2D), because their scores are not
calibrated the same way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from swimstroke.diagnostics import plot_diagnostics, summarize
from swimstroke.pose import load_pose, save_pose, track_video

THRESHOLDS = {"mediapipe": 0.5, "rtmpose": 0.3}
JOINT_GROUPS = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]


def make_model(kind: str):
    if kind == "mediapipe":
        from swimstroke.pose.mediapipe_pose import MediaPipePose

        return MediaPipePose("heavy")
    from swimstroke.pose.rtmpose_pose import RTMPose

    return RTMPose("balanced")


def pose_for(video: Path, kind: str, force: bool) -> pd.DataFrame:
    model = make_model(kind)
    path = Path("outputs") / video.stem / f"pose_{model.name}.parquet"
    if path.exists() and not force:
        model.close()
        return load_pose(path)
    print(f"  {model.name}: {video.name}")
    df = track_video(video, model, progress=True)
    model.close()
    save_pose(df, path)
    return df


def joint_rates(pose: pd.DataFrame, thr: float) -> dict[str, float]:
    """Share of frames where at least one side of each joint is confidently seen."""
    out = {}
    for j in JOINT_GROUPS:
        cols = [f"{s}_{j}_vis" for s in ("left", "right") if f"{s}_{j}_vis" in pose]
        if cols:
            out[f"{j}_seen"] = round(float((pose[cols] >= thr).any(axis=1).mean()), 3)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", type=Path, nargs="+")
    p.add_argument("--force", action="store_true", help="recompute cached pose tables")
    args = p.parse_args()

    rows = []
    for video in args.videos:
        for kind, thr in THRESHOLDS.items():
            pose = pose_for(video, kind, args.force)
            s = summarize(pose, vis_threshold=thr)
            rows.append({
                "clip": video.stem,
                "model": kind,
                "threshold": thr,
                "frames": s["frames"],
                "detection_rate": s["detection_rate"],
                **joint_rates(pose, thr),
            })
            plot_diagnostics(pose, Path("outputs") / video.stem / f"diagnostics_{kind}.png", vis_threshold=thr)

    out = Path("outputs/pose_compare")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

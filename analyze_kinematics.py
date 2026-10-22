"""Breaststroke kinematics from underwater side-view clips.

    uv run python analyze_kinematics.py VIDEO [VIDEO ...] [--force]

Per clip: RTMPose pose (cached as outputs/<clip>/pose_rtmpose-balanced.parquet),
then the plausibility gate, cycle detection and hip velocity
(swimstroke.kinematics). Writes to outputs/<clip>/:
    kinematics_cycles.csv   one row per cycle
    kinematics_series.csv   arm/leg signals and hip velocity on a uniform grid
    kinematics.png          signals with detected events, for checking by eye
and outputs/kinematics/summary.csv with one row per clip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from swimstroke.kinematics import analyze, gate_pose
from swimstroke.pose import load_pose, save_pose, track_video


def rtmpose_for(video: Path, force: bool) -> pd.DataFrame:
    path = Path("outputs") / video.stem / "pose_rtmpose-balanced.parquet"
    if path.exists() and not force:
        return load_pose(path)
    from swimstroke.pose.rtmpose_pose import RTMPose

    print(f"  pose: {video.name}")
    df = track_video(video, RTMPose("balanced"), progress=True)
    save_pose(df, path)
    return df


def plot(res: dict, title: str, path: Path) -> None:
    s = res["series"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax0.plot(s["t"], s["arm_reach"], label="arm reach (TL)")
    ax0.plot(s["t"], s["leg_reach"], label="heel position (TL)")
    for t in res["leg_events"]["t"]:
        ax0.axvline(t, color="C1", ls="--", lw=0.8)
    ax0.plot(res["arm_events"]["t"], res["arm_events"]["value"], "v", color="C0", label="arm pull-in")
    ax0.set_title(f"{title}: dashed = leg recovery (cycle start)", fontsize=10)
    ax0.legend(loc="upper right", fontsize=8)
    ax1.plot(s["t"], s["hip_vel_rel"], color="C2")
    ax1.set_ylabel("hip velocity\n(TL/s, rel. camera)")
    ax1.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", type=Path, nargs="+")
    p.add_argument("--force", action="store_true", help="recompute cached pose tables")
    args = p.parse_args()

    rows = []
    for video in args.videos:
        out = Path("outputs") / video.stem
        gated, gate = gate_pose(rtmpose_for(video, args.force))
        res = analyze(gated)
        summary = {k: v for k, v in res["summary"].items() if k != "params"}
        rows.append({"clip": video.stem, **gate, **summary})
        if res["summary"]["cycles"] == 0:
            print(f"{video.stem}: no reliable cycles (kept {gate['kept_rate']:.0%} of frames)")
            continue
        res["cycles"].to_csv(out / "kinematics_cycles.csv", index=False)
        res["series"].round(4).to_csv(out / "kinematics_series.csv", index=False)
        plot(res, video.stem, out / "kinematics.png")

    Path("outputs/kinematics").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/kinematics/summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

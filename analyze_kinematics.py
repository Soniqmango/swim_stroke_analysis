"""Breaststroke kinematics from underwater side-view clips.

    uv run python analyze_kinematics.py VIDEO [VIDEO ...] [--overlay] [--force]

Per clip: RTMPose pose (cached as outputs/<clip>/pose_rtmpose-balanced.parquet),
then the plausibility gate and swimstroke.kinematics.analyze. Writes to
outputs/<clip>/:
    kinematics_cycles.csv    one row per cycle: timing, phases, time gaps,
                             joint angles, asymmetry, hip velocity
    kinematics_events.csv    every timed event (cycle, event, t)
    kinematics_series.csv    signals, joint angles and hip velocity over time
    kinematics.png           signals with detected events, for checking by eye
    kinematics_overlay.mp4   video with skeleton + phases (with --overlay)
and to outputs/kinematics/:
    summary.csv              one row per clip (medians over cycles)
    validation.csv           detected vs hand-labelled events, if
                             data/labels/kinematics_events.csv has the clip
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
from swimstroke.validation import compare_kinematic_events, load_kinematics_labels


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
    s, ev = res["series"], res["events"]
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    ax0.plot(s["t"], s["arm_reach"], label="arm reach (TL)")
    ax0.plot(s["t"], s["leg_reach"], label="heel position (TL)")
    for t in res["leg_events"]["t"]:
        ax0.axvline(t, color="C1", ls="--", lw=0.8)
    marks = {"kick_end": ("C1", "^"), "leg_recovery_start": ("C1", "v"),
             "arm_extension": ("C0", "^"), "pull_start": ("C0", "v"), "pull_in": ("C0", "o")}
    for e, (c, m) in marks.items():
        x = ev[ev["event"] == e]["t"]
        sig = "leg_reach" if c == "C1" else "arm_reach"
        ax0.plot(x, [s[sig].iloc[(s["t"] - xi).abs().idxmin()] for xi in x], m, color=c, ms=5, label=e)
    ax0.set_title(f"{title}: dashed = leg recovery end (cycle start)", fontsize=10)
    ax0.legend(loc="upper right", fontsize=7, ncol=4)
    for side, ls in (("left", "-"), ("right", ":")):
        ax1.plot(s["t"], s[f"{side}_knee_angle"], ls, color="C3", label=f"{side} knee")
        ax1.plot(s["t"], s[f"{side}_elbow_angle"], ls, color="C4", label=f"{side} elbow")
    ax1.set_ylabel("angle (deg)")
    ax1.legend(loc="lower right", fontsize=7, ncol=4)
    ax2.plot(s["t"], s["hip_vel_rel"], color="C2")
    ax2.set_ylabel("hip velocity\n(TL/s, rel. camera)")
    ax2.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", type=Path, nargs="+")
    p.add_argument("--overlay", action="store_true", help="render kinematics_overlay.mp4 per clip")
    p.add_argument("--force", action="store_true", help="recompute cached pose tables")
    args = p.parse_args()

    rows, checks = [], []
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
        res["events"].to_csv(out / "kinematics_events.csv", index=False)
        res["series"].round(4).to_csv(out / "kinematics_series.csv", index=False)
        plot(res, video.stem, out / "kinematics.png")
        if args.overlay:
            from swimstroke.overlay import write_kinematics_overlay
            from swimstroke.pose.rtmpose_pose import RTMPose

            write_kinematics_overlay(video, gated, RTMPose.connections, res["events"], out / "kinematics_overlay.mp4")
        labels = load_kinematics_labels(video.stem)
        if labels is not None:
            checks.append(compare_kinematic_events(res["events"], labels).assign(clip=video.stem))

    agg = Path("outputs/kinematics")
    agg.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(agg / "summary.csv", index=False)
    print(df.set_index("clip").T.to_string())
    if checks:
        val = pd.concat(checks, ignore_index=True)
        val.to_csv(agg / "validation.csv", index=False)
        print("\nValidation against hand labels:\n" + val.to_string(index=False))


if __name__ == "__main__":
    main()

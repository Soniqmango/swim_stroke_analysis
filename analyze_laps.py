"""Length times from motion tracking, for clips filmed from one tripod spot.

    python analyze_laps.py data/raw/IMG_0115.MOV data/raw/IMG_0116.MOV data/raw/IMG_0117.MOV

All clips are stabilised onto the first frame of the first clip (--setup-ref
to override), so the swimmer's lane and the virtual wall positions are found
once from all turns together. Results go to outputs/laps/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from swimstroke.laps import length_times, segments, turn_vertices, wall_events, wall_positions
from swimstroke.stabilize import CameraMotion, estimate_motion
from swimstroke.track import find_candidates, track_swimmer
from swimstroke.validation import compare_lengths, error_summary
from swimstroke.video import iter_frames


def track_clip(video: Path, ref_image, force: bool) -> pd.DataFrame:
    """Stabilise -> candidates -> track, each step cached in outputs/<clip>/."""
    out = Path("outputs") / video.stem
    motion_f, blobs_f, track_f = out / "camera_motion_setup.npz", out / "blobs_setup.parquet", out / "track.parquet"
    if track_f.exists() and not force:
        return pd.read_parquet(track_f)
    if motion_f.exists() and not force:
        motion = CameraMotion.load(motion_f)
    else:
        print(f"  {video.stem}: stabilising")
        motion = estimate_motion(video, ref_image=ref_image)
        motion.save(motion_f)
    if blobs_f.exists() and not force:
        cands = pd.read_parquet(blobs_f)
    else:
        print(f"  {video.stem}: finding moving blobs")
        cands = find_candidates(video, motion)
        cands.to_parquet(blobs_f)
    path, _ = track_swimmer(cands, len(motion.t))
    path["t"] = motion.t
    path.to_parquet(track_f)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", type=Path, nargs="+")
    p.add_argument("--setup-ref", type=Path, help="clip whose first frame is the shared reference")
    p.add_argument("--force", action="store_true", help="recompute cached stabilisation/tracking")
    a = p.parse_args()

    ref_video = a.setup_ref or a.videos[0]
    ref_image = next(iter_frames(ref_video, max_frames=1)).image
    out = Path("outputs/laps")
    out.mkdir(parents=True, exist_ok=True)

    tracks = {v.stem: track_clip(v, ref_image, a.force) for v in a.videos}
    segs = {c: segments(t["t"].to_numpy(), t["x"].to_numpy()) for c, t in tracks.items()}

    vertices = pd.concat([turn_vertices(s).assign(clip=c) for c, s in segs.items()], ignore_index=True)
    walls = wall_positions(vertices)
    print("\nTurns found:")
    print(vertices.round(2).to_string(index=False))
    print(f"Virtual wall x (median over turns): { {k: round(v) for k, v in walls.items()} }")

    events = {c: wall_events(s, walls) for c, s in segs.items()}
    lengths = pd.concat([length_times(e).assign(clip=c) for c, e in events.items()], ignore_index=True)
    cmp = compare_lengths(lengths)
    cmp.to_csv(out / "length_validation.csv", index=False)
    summary = error_summary(cmp)
    summary.to_csv(out / "length_error_summary.csv")

    print("\nLength times vs. coach:")
    cols = ["clip", "length", "from_event", "to_event", "time_s_tool", "time_s_coach", "err_s"]
    print(cmp[cols].round(2).to_string(index=False))
    print("\nError summary (s):")
    print(summary.to_string())

    fig, axes = plt.subplots(len(tracks), 1, figsize=(12, 2.6 * len(tracks)), squeeze=False)
    for ax, (c, t) in zip(axes[:, 0], tracks.items()):
        ax.plot(t["t"], t["x"], ".", ms=1.5, color="#264653", label="tracked swimmer")
        for _, e in events[c].iterrows():
            ax.axvline(e["t"], color="#e76f51", lw=1)
            ax.text(e["t"], ax.get_ylim()[1], f" {e['event']}", fontsize=8, va="top", color="#e76f51")
        for w, x in walls.items():
            ax.axhline(x, color="grey", ls=":", lw=0.8)
        ax.set_ylabel(f"{c}\nx (px)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "tracks.png", dpi=110)
    print(f"\nSaved {out}/length_validation.csv, length_error_summary.csv, tracks.png")


if __name__ == "__main__":
    main()

"""Command-line entry point.

    python analyze.py VIDEO [VIDEO ...] --setup NAME

All clips must be filmed from the same camera position (the "setup"). The
first run creates data/calib/setups/NAME.json, stabilised onto the first
clip; later runs reuse its lane, wall positions and (if added) pool geometry.

Per clip -> outputs/<clip>/: events.csv, lengths.csv (+ overlay.mp4 with --overlay)
Session  -> outputs/sessions/<NAME>/: lengths.csv, turns.csv, tracks.png,
            and, where data/labels/lengths.csv has the clip, validation.csv
            and error_summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", type=Path, nargs="+", help="clips from one camera setup")
    p.add_argument("--setup", required=True, help="camera setup name, e.g. hotelpool_2026-10-04")
    p.add_argument("--no-strokes", action="store_true", help="skip pose estimation and stroke counting")
    p.add_argument("--overlay", action="store_true", help="render an annotated video per clip")
    p.add_argument("--force", action="store_true", help="recompute cached results")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for v in args.videos:
        if not v.exists():
            raise SystemExit(f"Video not found: {v}")

    from swimstroke.pipeline import Setup, analyze_session
    from swimstroke.validation import compare_lengths, error_summary, stroke_error_summary

    setup = Setup.load_or_create(args.setup, args.videos[0])
    print(f"Setup '{setup.name}' (reference: {Path(setup.ref_video).name}, "
          f"pool geometry: {'yes' if setup.pool_file else 'no - metres/DPS/turn times disabled'})")
    res = analyze_session(args.videos, setup, strokes=not args.no_strokes, force=args.force)

    out = Path("outputs/sessions") / setup.name
    out.mkdir(parents=True, exist_ok=True)
    lengths = res["lengths"]
    lengths.to_csv(out / "lengths.csv", index=False)
    print(f"\nLane: y = {setup.lane[0]:.4f} x + {setup.lane[1]:.0f}   "
          f"virtual walls: { {k: round(v) for k, v in setup.walls_px.items()} }")

    for name, pc in res["per_clip"].items():
        print(f"\n=== {name} ===")
        cols = [c for c in ["length", "from_event", "to_event", "time_s", "strokes",
                            "cycle_rate_per_min", "dps_m", "pose_coverage"] if c in pc["lengths"]]
        print(pc["lengths"][cols].round(2).to_string(index=False))
        if pc["strokes"]:
            print(f"pose on swimmer in {pc['strokes'].get('pose_on_swimmer', float('nan')):.0%} of frames")
    if len(res["turns"]):
        res["turns"].to_csv(out / "turns.csv", index=False)
        print("\nTurn times (5 m in -> 5 m out):")
        print(res["turns"].round(2).to_string(index=False))

    labelled = compare_lengths(lengths)
    labelled = labelled[labelled["clip"].isin(lengths["clip"])]
    if len(labelled):
        labelled.to_csv(out / "validation.csv", index=False)
        summ = error_summary(labelled)
        summ.to_csv(out / "error_summary.csv")
        print("\nLength times vs. coach:")
        print(labelled[["clip", "length", "from_event", "to_event", "time_s_tool", "time_s_coach", "err_s"]]
              .round(2).to_string(index=False))
        print("\nLength-time error summary (s):")
        print(summ.to_string())
        ss = stroke_error_summary(labelled)
        if ss is not None:
            print("\nStroke-count error:")
            print(ss.to_string())
        else:
            print("\nStroke counts: no length had enough pose coverage to count strokes.")

    plot_tracks(res, out / "tracks.png")
    print(f"\nSaved {out}/")

    if args.overlay:
        render_overlays(res)


def plot_tracks(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clips = res["clips"]
    fig, axes = plt.subplots(len(clips), 1, figsize=(12, 2.6 * len(clips)), squeeze=False)
    walls = res["setup"].walls_px
    for ax, c in zip(axes[:, 0], clips):
        ax.plot(c.track["t"], c.track["x"], ".", ms=1.5, color="#264653")
        cyc = c.cycles
        if cyc is not None and len(cyc):
            ax.plot(cyc["t"], np.interp(cyc["t"], c.track["t"][c.track["x"].notna()], c.track["x"].dropna()),
                    "v", color="#2a9d8f", ms=5, label="hand entry")
        for _, e in res["per_clip"][c.name]["events"].iterrows():
            ax.axvline(e["t"], color="#e76f51", lw=1)
            ax.text(e["t"], ax.get_ylim()[1], f" {e['event']}", fontsize=8, va="top", color="#e76f51")
        for x in walls.values():
            ax.axhline(x, color="grey", ls=":", lw=0.8)
        ax.set_ylabel(f"{c.name}\nx (px)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_overlays(res: dict) -> None:
    from swimstroke.overlay import write_overlay
    from swimstroke.pose.mediapipe_pose import MediaPipePose
    from swimstroke.stabilize import to_reference

    for c in res["clips"]:
        if c.pose is None:
            print(f"  {c.name}: no pose (ran with --no-strokes), skipping overlay")
            continue
        # track lives in the reference image; map it back into this clip's frames
        xy = c.track[["x", "y"]].to_numpy()
        markers = np.array([to_reference(p[None], np.linalg.inv(c.motion.H[i]))[0] for i, p in enumerate(xy)])
        path = write_overlay(
            c.video, c.pose, MediaPipePose.connections, Path("outputs") / c.name / "overlay.mp4",
            entries=None if c.cycles is None else c.cycles["t"].to_numpy(),
            markers=markers, events=res["per_clip"][c.name]["events"],
        )
        print(f"  overlay: {path}")


if __name__ == "__main__":
    main()

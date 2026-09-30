"""Command-line entry point: `python analyze.py VIDEO`.

Outputs go to <out>/<video name>/:
    pose_<model>.parquet   cached pose table (reused on the next run)
    overlay.mp4            skeleton + HUD drawn on the video
    diagnostics.png        detection and wrist/elbow visibility over time
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from swimstroke.diagnostics import plot_diagnostics, summarize
from swimstroke.overlay import write_overlay
from swimstroke.pose import load_pose, save_pose, track_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Swim stroke analysis from pool-deck video.")
    p.add_argument("video", type=Path, help="input video (.mp4/.mov)")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="output root (default: outputs/)")
    p.add_argument("--model", choices=["lite", "full", "heavy"], default="heavy",
                   help="MediaPipe pose model variant (default: heavy)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="only process the first N frames (quick tests; not cached)")
    p.add_argument("--force", action="store_true", help="re-run pose even if a cached result exists")
    p.add_argument("--no-video", action="store_true", help="skip rendering the overlay video")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    out_dir = args.out / args.video.stem
    model_name = f"mediapipe-{args.model}"
    cache = out_dir / f"pose_{model_name}.parquet"

    # Imported here so --help works instantly (MediaPipe is slow to import).
    from swimstroke.pose.mediapipe_pose import MediaPipePose

    # Pose estimation is by far the slowest step, so cache full runs. Partial
    # runs (--max-frames) are never cached, so the cache is always a full clip.
    if cache.exists() and not args.force and args.max_frames is None:
        print(f"Using cached pose: {cache}  (--force to recompute)")
        pose = load_pose(cache)
    else:
        print(f"Running pose estimation ({model_name}) on {args.video} ...")
        t0 = time.time()
        with MediaPipePose(args.model) as est:
            pose = track_video(args.video, est, max_frames=args.max_frames, progress=True)
        print(f"  {len(pose)} frames in {time.time() - t0:.0f}s")
        if args.max_frames is None:
            save_pose(pose, cache)
            print(f"  saved {cache}")
    pose.attrs["model"] = model_name

    stats = summarize(pose)
    print("\nPose quality:")
    print(json.dumps(stats, indent=2))
    (out_dir / "pose_quality.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "pose_quality.json").write_text(json.dumps(stats, indent=2))

    print(f"\nDiagnostics plot: {plot_diagnostics(pose, out_dir / 'diagnostics.png')}")
    if not args.no_video:
        print("Rendering overlay video ...")
        path = write_overlay(args.video, pose, MediaPipePose.connections, out_dir / "overlay.mp4")
        print(f"Overlay video:    {path}")


if __name__ == "__main__":
    main()

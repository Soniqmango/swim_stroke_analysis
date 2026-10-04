r"""Calibrate the phone lens from a checkerboard video.

    python calibrate_lens.py data/raw/checkerboard.MOV --cols 9 --rows 6 --square-mm 25 \
        --name iphone17promax_0.5x_1080p60

--cols/--rows count INNER corners (where four squares meet), not squares:
a board of 10 x 7 squares has 9 x 6 inner corners.
Writes data/calib/<name>.json (tracked in git) and an undistorted preview.
"""

import argparse
from pathlib import Path

import cv2

from swimstroke.lens import calibrate_from_video, undistort_image
from swimstroke.video import iter_frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--cols", type=int, required=True, help="inner corners per row")
    p.add_argument("--rows", type=int, required=True, help="inner corners per column")
    p.add_argument("--square-mm", type=float, required=True, help="measured size of one square")
    p.add_argument("--name", required=True, help="calibration name, e.g. iphone17promax_0.5x_1080p60")
    p.add_argument("--rational", action="store_true", help="extra distortion terms for strong wide-angle lenses")
    p.add_argument("--preview", type=Path, help="frame (video) to undistort as a visual check")
    a = p.parse_args()

    calib = calibrate_from_video(a.video, (a.cols, a.rows), a.square_mm, rational=a.rational, progress=True)
    out = calib.save(Path("data/calib") / f"{a.name}.json")

    K = calib.K_arr
    print(f"\nSaved {out}")
    print(f"  views used:        {calib.n_views}")
    print(f"  RMS reprojection:  {calib.rms_px:.3f} px   (good < 0.5, acceptable < 1.0)")
    print(f"  image coverage:    {calib.coverage:.0%}      (aim > 60%; edges matter most)")
    print(f"  focal length:      fx={K[0, 0]:.0f}  fy={K[1, 1]:.0f} px")
    print(f"  optical centre:    cx={K[0, 2]:.0f}  cy={K[1, 2]:.0f} px")
    print(f"  distortion:        {[round(d, 4) for d in calib.dist]}")
    if calib.rms_px > 1.0 or calib.coverage < 0.6:
        print("  WARNING: weak calibration - record more views closer to the image edges and corners.")

    src = a.preview or a.video
    frame = next(iter_frames(src, max_frames=1)).image
    prev = Path("outputs/calib") / f"{a.name}_preview.jpg"
    prev.parent.mkdir(parents=True, exist_ok=True)
    side = cv2.hconcat([frame, undistort_image(frame, calib)])
    cv2.imwrite(str(prev), cv2.resize(side, None, fx=0.5, fy=0.5))
    print(f"  preview (original | undistorted): {prev}")


if __name__ == "__main__":
    main()

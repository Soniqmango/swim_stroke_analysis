"""Mark pool-surface points on a setup's reference image -> metres.

    python mark_pool.py --setup hotelpool_2026-10-04

A window shows the reference image (the frame every clip is stabilised to).
Click a point ON THE WATER SURFACE whose pool position you know, then type
its coordinates in the terminal as "X Y" (metres):
    X = along the pool, 0 at the wall on the image's LEFT, 25 at the right wall
    Y = across the pool, 0 at the far side (the wall away from the camera)
Good points: where each end wall meets the far side and the near side
(needs the pool width), lane-rope colour changes, flags' positions projected
onto the water line. Use 4+ points spread over the surface; 6+ is better
(the extra points let the tool report its own error). Press q to finish.

Without a screen (or to script it): --points "x,y,X,Y;x,y,X,Y;..."
Add --lens data/calib/<lens>.json once the lens is calibrated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from swimstroke.pipeline import Setup
from swimstroke.pool import PoolGeometry


def click_points(image) -> list[tuple[float, float, float, float]]:
    pts: list[tuple[float, float, float, float]] = []
    view = image.copy()
    scale = 1600 / image.shape[1] if image.shape[1] > 1600 else 1.0
    win = "reference image - click a surface point, then type X Y in the terminal (q = done)"

    def redraw():
        v = cv2.resize(view, None, fx=scale, fy=scale)
        cv2.imshow(win, v)

    def on_click(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        px, py = x / scale, y / scale
        cv2.circle(view, (int(px), int(py)), 6, (0, 0, 255), -1)
        redraw()
        cv2.waitKey(1)
        raw = input(f"  point at ({px:.0f}, {py:.0f}) px -> pool coordinates 'X Y' in m (empty = discard): ").strip()
        if not raw:
            cv2.circle(view, (int(px), int(py)), 6, (128, 128, 128), -1)
        else:
            X, Y = (float(v) for v in raw.replace(",", " ").split())
            pts.append((px, py, X, Y))
            cv2.putText(view, f"({X:g},{Y:g})", (int(px) + 8, int(py) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        redraw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    redraw()
    while cv2.waitKey(50) & 0xFF != ord("q"):
        pass
    cv2.destroyAllWindows()
    return pts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setup", required=True)
    p.add_argument("--points", help='non-interactive: "x,y,X,Y;x,y,X,Y;..."')
    p.add_argument("--lens", help="lens calibration JSON (data/calib/<name>.json)")
    p.add_argument("--length", type=float, default=25.0, help="pool length in metres")
    a = p.parse_args()

    sp = Path("data/calib/setups") / f"{a.setup}.json"
    if not sp.exists():
        raise SystemExit(f"No setup '{a.setup}' yet - run analyze.py with --setup {a.setup} first.")
    setup = Setup.load_or_create(a.setup, "")

    if a.points:
        pts = [tuple(float(v) for v in item.split(",")) for item in a.points.split(";") if item.strip()]
    else:
        pts = click_points(setup.ref_image())
    if len(pts) < 4:
        raise SystemExit(f"Need at least 4 points, got {len(pts)}.")

    geom = PoolGeometry(
        image_points=[[x, y] for x, y, _, _ in pts],
        pool_points=[[X, Y] for _, _, X, Y in pts],
        pool_length_m=a.length,
        lens_file=a.lens,
    )
    out = geom.save(Path("data/calib") / f"{a.setup}_pool.json")
    setup.pool_file = str(out)
    setup.save()
    print(f"Saved {out} ({len(pts)} points, fit error {geom.rms_m:.3f} m RMS"
          f"{'' if len(pts) > 4 else ' - exactly 4 points always fit perfectly; add more to check'}).")
    print(f"Setup '{a.setup}' now uses it: analyze.py will report metres, DPS and 5 m turn times.")


if __name__ == "__main__":
    main()

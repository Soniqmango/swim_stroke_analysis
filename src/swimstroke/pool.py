"""Pixels -> metres on the pool surface, and the metrics that need metres.

The water surface is a plane, and a plane seen by a (pinhole) camera maps to
the image by a homography. Given >= 4 points on the surface whose pool
coordinates are known (e.g. the pool corners, marks measured from a wall),
we solve that homography once per camera setup.

Pool coordinates: X along the pool in metres (0 at the wall on the image's
left, pool_length at the right wall), Y across the pool in metres.

Order of operations for a tracked point (reference-image pixels):
    lens undistortion (lens.py, if calibrated) -> homography -> (X, Y) metres
Stabilisation (stabilize.py) maps every frame onto the reference image in
*distorted* pixels first; doing that before undistortion is a second-order
approximation, fine for drifts of ~100 px.

Image points must be marked on the setup's reference image (the frame all
clips are stabilised to), not on a raw frame.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from swimstroke.lens import LensCalibration, undistort_points


@dataclass
class PoolGeometry:
    image_points: list[list[float]]  # [[x_px, y_px], ...] on the reference image
    pool_points: list[list[float]]  # [[X_m, Y_m], ...] same order
    pool_length_m: float = 25.0
    lens_file: str | None = None  # data/calib/<lens>.json, if the lens is calibrated
    notes: str = ""
    H: list[list[float]] = field(default_factory=list)  # filled by fit()
    rms_m: float = float("nan")  # fit error at the given points

    def __post_init__(self) -> None:
        self._lens = LensCalibration.load(self.lens_file) if self.lens_file else None
        if len(self.image_points) != len(self.pool_points):
            raise ValueError("image_points and pool_points must have the same length")
        if not self.H:
            self.fit()

    def _undistort(self, xy: np.ndarray) -> np.ndarray:
        return undistort_points(xy, self._lens) if self._lens is not None else np.asarray(xy, float)

    def fit(self) -> float:
        """Solve the homography; returns the RMS error (m) at the given points.

        With exactly 4 points the fit is exact (error 0) and gives no check;
        5+ points make the error meaningful, so more points are better.
        """
        img = self._undistort(np.asarray(self.image_points, float))
        pool = np.asarray(self.pool_points, float)
        if len(img) < 4:
            raise ValueError("Need at least 4 surface points with known pool coordinates.")
        if np.linalg.matrix_rank(np.c_[pool, np.ones(len(pool))]) < 3:
            raise ValueError("Pool points are collinear; they must span the pool surface (not just one line).")
        H, _ = cv2.findHomography(img, pool, 0)
        self.H = H.tolist()
        err = np.linalg.norm(self.to_pool(np.asarray(self.image_points, float)) - pool, axis=1)
        self.rms_m = float(np.sqrt(np.mean(err**2)))
        return self.rms_m

    def to_pool(self, xy_px: np.ndarray) -> np.ndarray:
        """(N, 2) reference-image px -> (N, 2) pool metres. NaN rows stay NaN."""
        pts = np.asarray(xy_px, float).reshape(-1, 2)
        out = np.full_like(pts, np.nan)
        ok = ~np.isnan(pts).any(axis=1)
        if ok.any():
            und = self._undistort(pts[ok])
            out[ok] = cv2.perspectiveTransform(und.reshape(-1, 1, 2), np.asarray(self.H)).reshape(-1, 2)
        return out

    def wall_X(self, side: str) -> float:
        return 0.0 if side == "left" else self.pool_length_m

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {k: v for k, v in asdict(self).items()}
        path.write_text(json.dumps(d, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PoolGeometry":
        return cls(**json.loads(Path(path).read_text()))


# ---------- metrics that need metres ----------


def crossing_time(t: np.ndarray, X: np.ndarray, target: float, t_from: float, t_to: float) -> float | None:
    """First time in [t_from, t_to] where X crosses `target`, linearly
    interpolated between the two tracked samples on either side."""
    sel = (t >= t_from) & (t <= t_to) & ~np.isnan(X)
    tt, xx = t[sel], X[sel] - target
    # a crossing: the sign changes, or a sample sits exactly on the target
    # (sign 0) and the next one has moved off it
    a, b = xx[:-1], xx[1:]
    idx = np.flatnonzero(((a <= 0) & (b > 0)) | ((a >= 0) & (b < 0)))
    if len(idx) == 0:
        return None
    i = idx[0]
    return float(tt[i] + (tt[i + 1] - tt[i]) * (0 - xx[i]) / (xx[i + 1] - xx[i]))


def line_crossing(t: np.ndarray, X: np.ndarray, target: float, sel: np.ndarray) -> float | None:
    """Fit X = a*t + b on the selected samples and solve for X = target.
    Used when the swimmer isn't visible at the target (e.g. underwater glide)."""
    ok = sel & ~np.isnan(X)
    if ok.sum() < 5:
        return None
    a, b = np.polyfit(t[ok], X[ok], 1)
    return None if np.isclose(a, 0) else float((target - b) / a)


def turn_times(
    t: np.ndarray,
    X: np.ndarray,
    events: pd.DataFrame,
    pool_length_m: float = 25.0,
    dist_m: float = 5.0,
    fit_span_m: float = 6.0,
) -> pd.DataFrame:
    """Turn time = time from `dist_m` before the wall to `dist_m` after it.

    Both crossings come from a line fitted to the track just beyond the
    5 m mark (from 5 to 5 + fit_span_m metres from the wall), not from the
    single nearest sample: the blob position jitters with splash, and after
    the turn the swimmer may still be gliding underwater (invisible) at 5 m.
    Blob centre vs. head offset is about the same in and out, so it largely
    cancels in the difference.
    """
    rows = []
    turns = events[events["event"] == "turn"]
    all_t = events["t"].to_numpy()
    for _, ev in turns.iterrows():
        Xw = 0.0 if ev["wall"] == "left" else pool_length_m
        d = np.abs(X - Xw)
        prev_ev = all_t[all_t < ev["t"]].max() if (all_t < ev["t"]).any() else -np.inf
        next_ev = all_t[all_t > ev["t"]].min() if (all_t > ev["t"]).any() else np.inf
        band = (d >= dist_m) & (d <= dist_m + fit_span_m)
        mark = Xw + np.sign(pool_length_m / 2 - Xw) * dist_m  # 5 m into the pool from this wall
        t_in = line_crossing(t, X, mark, band & (t > prev_ev) & (t < ev["t"]))
        t_out = line_crossing(t, X, mark, band & (t > ev["t"]) & (t < next_ev))
        rows.append({
            "turn_t": ev["t"], "wall": ev["wall"],
            "t_in_5m": t_in, "t_out_5m": t_out,
            "turn_time_s": (t_out - t_in) if t_in is not None and t_out is not None else None,
        })
    return pd.DataFrame(rows, columns=["turn_t", "wall", "t_in_5m", "t_out_5m", "turn_time_s"])


def distance_per_stroke(t: np.ndarray, X: np.ndarray, entries: np.ndarray) -> float | None:
    """Metres travelled between the first and last near-arm hand entry of a
    length, divided by the strokes taken in between (2 per near-arm cycle).

    Measured between entries, not wall to wall, so the push-off glide and
    the final touch (no strokes) don't inflate it.
    """
    entries = np.sort(np.asarray(entries, float))
    if len(entries) < 2:
        return None
    ok = ~np.isnan(X)
    x0, x1 = np.interp(entries[[0, -1]], t[ok], X[ok])
    n_between = 2 * (len(entries) - 1)  # near-arm cycles between first and last entry
    return float(abs(x1 - x0) / n_between)

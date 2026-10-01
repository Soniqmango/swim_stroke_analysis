"""Stroke detection from the pose table.

Signal: "reach" = how far the wrist is ahead of the shoulder, measured along
the swimmer's own body axis (hips -> shoulders) and divided by torso length:

    reach = dot(wrist - shoulder, unit(shoulder_mid - hip_mid)) / torso_length

  - Projecting on the body axis makes it independent of swim direction and of
    how the swimmer is oriented in the image (diagonal, left-to-right, ...).
  - Dividing by torso length makes it independent of distance to the camera.
  - In freestyle each arm traces one cycle: reach is maximal as the hand
    enters the water in front of the head, minimal as it exits at the hip.

Near arm, not left/right: from the side, pose models often can't tell the
arms apart. When one arm is hidden they put both wrists on the visible one,
or swap labels mid-stroke. So instead of trusting labels we follow the *near*
arm (the one facing the camera) by weighting each wrist by its visibility.
One near-arm reach peak = one arm cycle. Freestyle is symmetric, so
strokes = 2 x cycles, the same way many coaches count one arm and double it.

Pipeline: reach -> resample to a uniform time grid -> smooth -> find_peaks.
Resampling comes first because phone video has a variable frame rate and
smoothing/peak spacing assume evenly spaced samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass(frozen=True)
class StrokeParams:
    fs: float = 60.0  # Hz, uniform resampling rate
    max_gap_s: float = 0.25  # interpolate over pose dropouts up to this long
    min_segment_s: float = 0.5  # ignore tracked stretches shorter than this
    smooth_sigma_s: float = 0.06  # Gaussian smoothing width (~ +-2 frames at 30fps)
    min_cycle_s: float = 0.7  # fastest plausible arm cycle (~85 cycles/min)
    min_prominence: float = 0.5  # peak must stand out by half a torso length
    min_reach: float = 0.0  # at entry the hand is ahead of the shoulder
    max_cycle_s: float = 2.5  # longer gaps between peaks = missed cycles


def _xy(pose: pd.DataFrame, name: str) -> np.ndarray:
    return pose[[f"{name}_x", f"{name}_y"]].to_numpy(dtype=float)


def reach_signal(pose: pd.DataFrame, arm: str) -> np.ndarray:
    """Per-frame reach of one labelled arm, in torso lengths (NaN where no pose)."""
    shoulder_mid = (_xy(pose, "left_shoulder") + _xy(pose, "right_shoulder")) / 2
    hip_mid = (_xy(pose, "left_hip") + _xy(pose, "right_hip")) / 2
    axis = shoulder_mid - hip_mid
    torso = np.linalg.norm(axis, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = axis / torso[:, None]
        rel = _xy(pose, f"{arm}_wrist") - _xy(pose, f"{arm}_shoulder")
        return (rel * unit).sum(axis=1) / torso


def near_arm_reach(pose: pd.DataFrame) -> np.ndarray:
    """Visibility-weighted reach of both wrists: follows whichever arm the
    camera sees best, and degrades gracefully when labels collapse/swap."""
    r_l, r_r = reach_signal(pose, "left"), reach_signal(pose, "right")
    v_l = pose["left_wrist_vis"].to_numpy(dtype=float)
    v_r = pose["right_wrist_vis"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (v_l * r_l + v_r * r_r) / (v_l + v_r)


def resample(t: np.ndarray, y: np.ndarray, fs: float, max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear resample onto a uniform grid; grid points inside a gap longer
    than max_gap_s (or outside the data) become NaN instead of made-up values."""
    grid = np.arange(t[0], t[-1] + 0.5 / fs, 1.0 / fs)
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return grid, np.full(grid.shape, np.nan)
    tv, yv = t[ok], y[ok]
    out = np.interp(grid, tv, yv, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(tv, grid), 1, len(tv) - 1)
    out[(tv[j] - tv[j - 1]) > max_gap_s] = np.nan  # gap around this grid point too long
    return grid, out


def segments(valid: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index ranges of consecutive True values."""
    edges = np.diff(np.concatenate([[0], valid.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def smooth(y: np.ndarray, fs: float, sigma_s: float, min_segment_s: float) -> np.ndarray:
    """Gaussian-smooth each continuous stretch separately (never across gaps).
    Gaussian smoothing is symmetric, so it doesn't shift peaks in time."""
    out = np.full_like(y, np.nan)
    for a, b in segments(~np.isnan(y)):
        if (b - a) / fs >= min_segment_s:
            out[a:b] = gaussian_filter1d(y[a:b], sigma_s * fs, mode="nearest")
    return out


def stroke_signal(pose: pd.DataFrame, params: StrokeParams = StrokeParams()) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sampled, smoothed near-arm reach: (t, reach)."""
    t = pose["t"].to_numpy(dtype=float)
    grid, y = resample(t, near_arm_reach(pose), params.fs, params.max_gap_s)
    return grid, smooth(y, params.fs, params.smooth_sigma_s, params.min_segment_s)


def detect_cycles(pose: pd.DataFrame, params: StrokeParams = StrokeParams()) -> pd.DataFrame:
    """One row per detected near-arm hand entry (= one arm cycle)."""
    t, y = stroke_signal(pose, params)
    rows = []
    for seg, (a, b) in enumerate(segments(~np.isnan(y))):
        # find_peaks can't see across NaN, so search each tracked stretch alone.
        # Consequence: an entry right at the edge of a stretch is missed.
        idx, props = find_peaks(
            y[a:b],
            height=params.min_reach,
            prominence=params.min_prominence,
            distance=max(1, int(params.min_cycle_s * params.fs)),
        )
        for i, prom in zip(idx, props["prominences"]):
            rows.append({
                "t": round(float(t[a + i]), 3),
                "segment": seg,  # which continuous tracked stretch it came from
                "reach": float(y[a + i]),
                "prominence": float(prom),
            })
    return pd.DataFrame(rows, columns=["t", "segment", "reach", "prominence"])


def cycle_intervals(times: np.ndarray, segment: np.ndarray | None = None, max_cycle_s: float = 2.5) -> np.ndarray:
    """Intervals between consecutive entries that are trustworthy cycle lengths.

    An interval only counts if both entries come from the same continuous
    tracked stretch: across a tracking gap an entry may have been missed, and
    the interval would be 2 cycles long. max_cycle_s is a second safety net.
    """
    times = np.asarray(times, float)
    d = np.diff(times)
    keep = d <= max_cycle_s
    if segment is not None:
        segment = np.asarray(segment)
        keep &= segment[1:] == segment[:-1]
    return d[keep]


def summarize_strokes(cycles: pd.DataFrame, pose: pd.DataFrame, params: StrokeParams = StrokeParams()) -> dict:
    """Cycle/stroke rate from consecutive entries plus tracking coverage.

    Rate = 60 / median cycle interval (see cycle_intervals); the median is
    robust to the odd missed or extra peak.
    Convention: freestyle rate is usually quoted in cycles/min; counting every
    hand entry (strokes/min) is 2x that. Both are reported to avoid confusion.
    """
    _, y = stroke_signal(pose, params)
    d = cycle_intervals(cycles["t"], cycles["segment"], params.max_cycle_s)
    period = float(np.median(d)) if len(d) else float("nan")
    return {
        "cycles_detected": int(len(cycles)),
        "strokes_detected": int(2 * len(cycles)),
        "cycle_period_s": round(period, 3),
        "cycle_rate_per_min": round(60 / period, 1) if len(d) else None,
        "stroke_rate_per_min": round(120 / period, 1) if len(d) else None,
        "intervals_used": int(len(d)),
        "tracked_s": round(float((~np.isnan(y)).sum() / params.fs), 2),
        "params": asdict(params),
    }

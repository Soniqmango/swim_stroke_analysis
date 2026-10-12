"""Wall events (start, turns, finish) and length times from the swimmer track.

Input: tracked swimmer x-position over time, in the shared reference image.

1. Segments: the track splits into stretches of steady swimming in one
   direction (left or right), separated by turns.
2. Local lines: near each wall the swimmer moves at roughly constant speed,
   so we fit a straight line x(t) to the last few seconds before a wall and
   the first few seconds after it.
3. Virtual wall: at a turn, the incoming and outgoing lines cross at
   (t_turn, x_wall). x_wall is where the *blob centre* is when the swimmer
   "is at the wall", about a body length from where the hand touches. It is
   the same for every turn at that wall, so we take the median over turns
   (optionally pooled over clips filmed from the same tripod position).
4. Events: a turn happens at the crossing of its two lines; a start (push-off)
   when the first segment's line leaves x_wall; a finish when the last
   segment's line reaches x_wall.

The hand touch happens a roughly constant time before the blob centre
reaches x_wall, so that offset largely cancels in a touch-to-touch length
time. The first length (from a push-off) and the last (to a finish) are
the least symmetric, so they are expected to carry the largest bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class LapParams:
    smooth_s: float = 0.5  # median filter on x to remove splash jitter
    min_segment_s: float = 3.0  # shorter stretches aren't a length of swimming
    min_speed_px_s: float = 20.0  # slower = not swimming along the pool
    fit_window_s: float = 3.0  # line fit over this much track next to a wall
    max_gap_s: float = 1.0  # a gap longer than this inside a segment splits it


@dataclass
class Segment:
    t: np.ndarray
    x: np.ndarray
    direction: int  # +1 moving right (increasing x), -1 moving left

    def line(self, at_end: bool, window_s: float) -> tuple[float, float]:
        """(slope, intercept) of x = slope * t + intercept near one end."""
        if at_end:
            sel = self.t >= self.t[-1] - window_s
        else:
            sel = self.t <= self.t[0] + window_s
        slope, intercept = np.polyfit(self.t[sel], self.x[sel], 1)
        return float(slope), float(intercept)


def segments(t: np.ndarray, x: np.ndarray, params: LapParams = LapParams()) -> list[Segment]:
    """Split the track into one-directional swimming segments."""
    ok = ~np.isnan(x)
    t, x = np.asarray(t)[ok], np.asarray(x)[ok]
    if len(t) < 3:
        return []
    dt = np.median(np.diff(t))
    xs = median_filter(x, size=max(3, int(params.smooth_s / dt) | 1), mode="nearest")
    # direction from a centred difference over ~1 s
    k = max(1, int(0.5 / dt))
    v = np.zeros_like(xs)
    v[k:-k] = (xs[2 * k:] - xs[:-2 * k]) / (t[2 * k:] - t[:-2 * k])
    v[:k], v[-k:] = v[k], v[-k - 1]
    d = np.where(v > params.min_speed_px_s, 1, np.where(v < -params.min_speed_px_s, -1, 0))

    out: list[Segment] = []
    start = 0
    for i in range(1, len(t) + 1):
        boundary = i == len(t) or d[i] != d[start] or t[i] - t[i - 1] > params.max_gap_s
        if boundary:
            if d[start] != 0 and t[i - 1] - t[start] >= params.min_segment_s:
                out.append(Segment(t[start:i], xs[start:i], int(d[start])))
            start = i
    # merge neighbours with the same direction (split by a short gap or stall)
    merged: list[Segment] = []
    for s in out:
        if merged and merged[-1].direction == s.direction:
            m = merged[-1]
            merged[-1] = Segment(np.r_[m.t, s.t], np.r_[m.x, s.x], s.direction)
        else:
            merged.append(s)
    return merged


def wall_side(direction: int) -> str:
    """Which wall a segment swims towards."""
    return "right" if direction > 0 else "left"


def turn_vertices(segs: list[Segment], params: LapParams = LapParams()) -> pd.DataFrame:
    """Crossing point of incoming/outgoing lines at every turn."""
    rows = []
    for a, b in zip(segs, segs[1:]):
        if a.direction == b.direction:
            continue
        m1, c1 = a.line(at_end=True, window_s=params.fit_window_s)
        m2, c2 = b.line(at_end=False, window_s=params.fit_window_s)
        if np.isclose(m1, m2):
            continue
        t = (c2 - c1) / (m1 - m2)
        rows.append({"t": t, "x": m1 * t + c1, "wall": wall_side(a.direction)})
    return pd.DataFrame(rows, columns=["t", "x", "wall"])


def wall_positions(vertices: pd.DataFrame) -> dict[str, float]:
    """Virtual wall x per side: median over all turns at that wall."""
    return {w: float(g["x"].median()) for w, g in vertices.groupby("wall")}


def wall_events(segs: list[Segment], walls: dict[str, float], params: LapParams = LapParams()) -> pd.DataFrame:
    """Start, turn and finish times. A turn uses the crossing of its two lines;
    start/finish use where the single available line meets the virtual wall."""
    if not segs:
        return pd.DataFrame(columns=["t", "event", "wall"])
    rows = []

    def cross(line: tuple[float, float], x_wall: float) -> float:
        m, c = line
        return (x_wall - c) / m

    first = segs[0]
    start_wall = wall_side(-first.direction)  # it swims away from this wall
    if start_wall in walls:
        rows.append({"t": cross(first.line(False, params.fit_window_s), walls[start_wall]),
                     "event": "start", "wall": start_wall})
    for _, v in turn_vertices(segs, params).iterrows():
        rows.append({"t": v["t"], "event": "turn", "wall": v["wall"]})
    last = segs[-1]
    end_wall = wall_side(last.direction)
    if end_wall in walls:
        rows.append({"t": cross(last.line(True, params.fit_window_s), walls[end_wall]),
                     "event": "finish", "wall": end_wall})
    return pd.DataFrame(rows).sort_values("t", ignore_index=True)


def length_times(events: pd.DataFrame) -> pd.DataFrame:
    """One row per length between consecutive wall events."""
    t = events["t"].to_numpy()
    return pd.DataFrame({
        "length": np.arange(1, len(t)),
        "t_start": t[:-1],
        "t_end": t[1:],
        "time_s": np.diff(t),
        "from_event": events["event"].to_numpy()[:-1],
        "to_event": events["event"].to_numpy()[1:],
    })

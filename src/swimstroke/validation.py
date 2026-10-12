"""Compare detected events with hand-labelled reference events.

Reference labels live in data/labels/ (tracked in git). This module grows
into the milestone-5 error table; for now it scores stroke detection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from swimstroke.strokes import cycle_intervals

STROKE_LABELS = Path("data/labels/stroke_entries.csv")
LENGTH_LABELS = Path("data/labels/lengths.csv")


def compare_lengths(tool: pd.DataFrame, path: str | Path = LENGTH_LABELS) -> pd.DataFrame:
    """Join tool length times (clip, length, time_s, from_event, to_event) with
    the coach's per-length reference; one row per reference length."""
    ref = pd.read_csv(path, comment="#")
    out = ref.merge(tool, on=["clip", "length"], how="left", suffixes=("_coach", "_tool"))
    out["err_s"] = out["time_s_tool"] - out["time_s_coach"]
    out["err_pct"] = 100 * out["err_s"] / out["time_s_coach"]
    return out


def error_summary(cmp: pd.DataFrame) -> pd.DataFrame:
    """MAE / bias / % error, overall and by length type (start/turn/finish
    boundaries behave differently, so they are reported separately)."""
    c = cmp.dropna(subset=["err_s"]).copy()
    c["kind"] = c["from_event"] + "->" + c["to_event"]

    def stats(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(g),
            "mae_s": g["err_s"].abs().mean(),
            "bias_s": g["err_s"].mean(),
            "mape_pct": g["err_pct"].abs().mean(),
            "max_abs_s": g["err_s"].abs().max(),
        })

    rows = [stats(c).rename("all")]
    rows += [stats(g).rename(k) for k, g in c.groupby("kind")]
    return pd.DataFrame(rows).round(3)


def stroke_error_summary(cmp: pd.DataFrame) -> pd.DataFrame | None:
    """Stroke-count error per length (tool - coach), where the tool counted."""
    if "strokes_tool" not in cmp:
        return None
    c = cmp.dropna(subset=["strokes_tool", "strokes_coach"]).copy()
    if c.empty:
        return None
    err = c["strokes_tool"].astype(float) - c["strokes_coach"].astype(float)
    return pd.DataFrame({
        "n": [len(c)],
        "mae_strokes": [err.abs().mean()],
        "bias_strokes": [err.mean()],
        "mape_pct": [(100 * err.abs() / c["strokes_coach"]).mean()],
        "exact_share": [(err == 0).mean()],
    }).round(3)


def load_stroke_labels(clip: str, path: str | Path = STROKE_LABELS) -> pd.DataFrame | None:
    """Reference near-arm entry times for one clip, or None if not labelled."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    df = df[df["clip"] == clip]
    return df if len(df) else None


def match_events(detected: np.ndarray, reference: np.ndarray, tol_s: float = 0.3) -> dict:
    """Greedy one-to-one matching of event times within +-tol_s.

    Each reference event can be matched by at most one detection (and vice
    versa), closest pairs first, so one detection can't count as two hits.
    """
    detected = np.sort(np.asarray(detected, float))
    reference = np.sort(np.asarray(reference, float))
    pairs = sorted(
        (abs(d - r), i, j)
        for i, d in enumerate(detected)
        for j, r in enumerate(reference)
        if abs(d - r) <= tol_s
    )
    used_d, used_r, offsets = set(), set(), []
    for _, i, j in pairs:
        if i not in used_d and j not in used_r:
            used_d.add(i)
            used_r.add(j)
            offsets.append(detected[i] - reference[j])
    tp = len(offsets)
    fp, fn = len(detected) - tp, len(reference) - tp
    return {
        "reference": len(reference),
        "detected": len(detected),
        "true_pos": tp,
        "false_pos": fp,
        "missed": fn,
        "recall": round(tp / len(reference), 3) if len(reference) else None,
        "precision": round(tp / len(detected), 3) if len(detected) else None,
        "mean_offset_s": round(float(np.mean(offsets)), 3) if offsets else None,
    }


def cycle_rate(times: np.ndarray, segment: np.ndarray | None = None) -> float | None:
    """Cycles/min via the same interval rules for the tool and the reference."""
    d = cycle_intervals(times, segment)
    return round(60 / float(np.median(d)), 1) if len(d) else None


def compare_with_labels(cycles: pd.DataFrame, labels: pd.DataFrame, tol_s: float = 0.3) -> pd.DataFrame:
    """Score detected entries (output of detect_cycles) per labelled window.

    Detections outside the labelled windows are ignored: nobody labelled
    those stretches, so we can't say whether they are right or wrong.
    A labelled window is one continuous pass, so all its reference intervals
    count; the tool's intervals still follow its own tracking segments.
    """
    rows = []
    for window, g in labels.groupby("window", sort=False):
        ref = g["t"].to_numpy(dtype=float)
        lo, hi = ref.min() - tol_s, ref.max() + tol_s
        det = cycles[(cycles["t"] >= lo) & (cycles["t"] <= hi)]
        tool_rate = cycle_rate(det["t"], det["segment"])
        ref_rate = cycle_rate(ref)
        det = det["t"].to_numpy()
        rows.append({
            "window": window,
            **match_events(det, ref, tol_s),
            "ref_cycles_per_min": ref_rate,
            "tool_cycles_per_min": tool_rate,
            "rate_error_pct": round(100 * (tool_rate - ref_rate) / ref_rate, 1)
            if tool_rate is not None and ref_rate else None,
        })
    return pd.DataFrame(rows)

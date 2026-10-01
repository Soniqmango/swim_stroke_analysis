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

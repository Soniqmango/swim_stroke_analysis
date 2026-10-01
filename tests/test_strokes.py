"""Stroke detection and validation tests on synthetic swimmers with known answers."""

import numpy as np
import pandas as pd
import pytest

from swimstroke.strokes import (
    cycle_intervals,
    detect_cycles,
    near_arm_reach,
    resample,
    segments,
    summarize_strokes,
)
from swimstroke.validation import compare_with_labels, match_events

PERIOD = 1.1  # s per arm cycle -> 54.5 cycles/min


def synthetic_pose(duration=6.0, fps=30, period=PERIOD, gap=None, direction=1.0, seed=0):
    """Swimmer whose near-arm wrist oscillates along the body axis.

    Hips at x=0, shoulders at x=100*direction (torso = 100 px), so reach
    peaks (hand entries) are at t = k * period. Timestamps get VFR-like
    jitter. `gap=(t0, t1)` simulates the pose being lost.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fps) + rng.uniform(-0.002, 0.002, int(duration * fps))
    t[0] = 0.0
    reach = np.cos(2 * np.pi * t / period)  # in torso lengths
    sx = 100 * direction
    cols = {"frame": np.arange(len(t)), "t": t, "detected": np.ones(len(t), bool)}
    for side in ("left", "right"):
        cols |= {f"{side}_shoulder_x": sx, f"{side}_shoulder_y": 0.0, f"{side}_shoulder_vis": 1.0,
                 f"{side}_hip_x": 0.0, f"{side}_hip_y": 0.0, f"{side}_hip_vis": 1.0}
    # near (right) arm does the stroke and is clearly visible; the far (left)
    # wrist is a low-visibility guess stuck at the shoulder
    cols |= {"right_wrist_x": sx + direction * 100 * reach, "right_wrist_y": 0.0, "right_wrist_vis": 0.9,
             "left_wrist_x": sx, "left_wrist_y": 0.0, "left_wrist_vis": 0.1}
    df = pd.DataFrame(cols)
    if gap:
        lost = (df["t"] >= gap[0]) & (df["t"] < gap[1])
        df.loc[lost, "detected"] = False
        df.loc[lost, [c for c in df.columns if c.endswith(("_x", "_y", "_vis"))]] = np.nan
    return df


def test_near_arm_reach_follows_visible_arm():
    p = synthetic_pose()
    r = near_arm_reach(p)
    expected = 0.9 * np.cos(2 * np.pi * p["t"] / PERIOD) / (0.9 + 0.1)
    np.testing.assert_allclose(r, expected, atol=1e-9)


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_detects_every_cycle_in_both_swim_directions(direction):
    cycles = detect_cycles(synthetic_pose(direction=direction))
    expected = PERIOD * np.arange(1, 6)  # 1.1 ... 5.5 (t=0 is an edge, not a peak)
    np.testing.assert_allclose(cycles["t"], expected, atol=0.03)


def test_rate_matches_known_period():
    p = synthetic_pose()
    s = summarize_strokes(detect_cycles(p), p)
    assert s["cycle_rate_per_min"] == pytest.approx(60 / PERIOD, rel=0.02)
    assert s["stroke_rate_per_min"] == pytest.approx(2 * s["cycle_rate_per_min"], abs=0.2)


def test_gap_splits_segments_and_is_not_interpolated():
    p = synthetic_pose(gap=(2.5, 2.9))
    cycles = detect_cycles(p)
    assert cycles["segment"].nunique() == 2
    # no detection invented inside the gap
    assert not ((cycles["t"] > 2.5) & (cycles["t"] < 2.9)).any()


def test_resample_masks_long_gaps_only():
    t = np.array([0.0, 0.1, 0.2, 1.0, 1.1])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    grid, out = resample(t, y, fs=10, max_gap_s=0.25)
    assert out[1] == pytest.approx(1.0)
    assert np.isnan(out[5])  # t=0.5 lies in a 0.8 s gap
    assert out[-1] == pytest.approx(4.0)


def test_segments():
    assert segments(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert segments(np.zeros(3, bool)) == []


def test_intervals_across_tracking_gap_are_excluded():
    # 1.0 -> 3.2 spans a gap where a cycle was probably missed
    times = np.array([0.0, 1.1, 3.2, 4.3])
    seg = np.array([0, 0, 1, 1])
    np.testing.assert_allclose(cycle_intervals(times, seg), [1.1, 1.1])
    np.testing.assert_allclose(cycle_intervals(times), [1.1, 2.1, 1.1])


def test_match_events_is_one_to_one():
    # one detection between two references may only claim one of them
    m = match_events(np.array([1.0]), np.array([0.9, 1.1]), tol_s=0.3)
    assert (m["true_pos"], m["false_pos"], m["missed"]) == (1, 0, 1)


def test_match_events_counts_false_positives():
    m = match_events(np.array([1.0, 2.0, 5.0]), np.array([1.05, 2.1]), tol_s=0.3)
    assert (m["true_pos"], m["false_pos"], m["missed"]) == (2, 1, 0)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-3)


def test_compare_with_labels_ignores_unlabelled_stretches():
    cycles = pd.DataFrame({"t": [1.1, 2.2, 3.3, 20.0], "segment": [0, 0, 0, 1]})
    labels = pd.DataFrame({"clip": "c", "window": "w", "t": [1.1, 2.2, 3.3]})
    r = compare_with_labels(cycles, labels).iloc[0]
    assert (r["true_pos"], r["false_pos"], r["missed"]) == (3, 0, 0)  # 20.0 not counted
    assert r["rate_error_pct"] == pytest.approx(0.0)

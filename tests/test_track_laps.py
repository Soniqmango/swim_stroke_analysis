"""Tracking and lap-timing tests on synthetic swimmers with known answers."""

import numpy as np
import pandas as pd
import pytest

from swimstroke.laps import length_times, segments, turn_vertices, wall_events, wall_positions
from swimstroke.track import PathParams, best_path, fit_lane, track_swimmer
from swimstroke.validation import compare_lengths, error_summary

# ---------- tracking ----------

N = 400


def scene_candidates(seed=0):
    """A swimmer moving left along y~680 (missing for frames 150-200, like an
    underwater turn) plus bigger 'waves' that travel across the pool in y."""
    rng = np.random.default_rng(seed)
    rows, truth = [], {}
    for f in range(N):
        if not 150 <= f < 200:
            x, y = 1500 - 2.5 * f, 680 + 0.02 * (1500 - 2.5 * f) * 0 + rng.normal(0, 2)
            rows.append((f, f / 60, x + rng.normal(0, 3), y, 400.0))
            truth[f] = x
        for k in range(3):  # three wave crests, each sweeping down the image
            y = 640 + ((f * 2.0 + 60 * k) % 180)
            rows.append((f, f / 60, 900 + 150 * k + rng.normal(0, 5), y, 900.0))
    return pd.DataFrame(rows, columns=["frame", "t", "x", "y", "area"]), truth


def test_best_path_follows_swimmer_not_waves():
    cands, truth = scene_candidates()
    path = best_path(cands, N)
    frames = np.array(list(truth))
    picked = path.loc[frames, "x"].to_numpy()
    on_swimmer = np.abs(picked - np.array([truth[f] for f in frames])) < 15
    wrong = ~np.isnan(picked) & ~on_swimmer
    assert wrong.sum() == 0  # never picks a wave while the swimmer is visible
    # The path may skip an occasional noisy point (cheaper than two jumps through
    # it); harmless for lap timing, which fits lines over seconds of track.
    assert on_swimmer.mean() > 0.9
    # While the swimmer is invisible the path MAY follow a wave crossing the
    # lane (documented limitation). What matters: that stretch must not look
    # like swimming along the pool, so lap timing can't mistake it for a length.
    gap = path.loc[150:199].dropna()
    if len(gap) > 1:
        along_pool_speed = abs(gap["x"].iloc[-1] - gap["x"].iloc[0]) / ((gap.index[-1] - gap.index[0]) / 60)
        assert along_pool_speed < 20  # px/s; laps.LapParams.min_speed_px_s


def test_best_path_rejects_jumps_beyond_speed_limit():
    # two candidates 1000 px apart one frame later can't be the same swimmer
    c = pd.DataFrame({"frame": [0, 1], "t": [0, 1 / 60], "x": [0.0, 1000.0], "y": [0.0, 0.0], "area": [400.0, 400.0]})
    path = best_path(c, 2, PathParams())
    assert path["x"].notna().sum() == 1


def test_fit_lane_ignores_outliers():
    rng = np.random.default_rng(1)
    x = rng.uniform(300, 1700, 500)
    y = 0.02 * x + 650 + rng.normal(0, 3, 500)
    y[:100] += rng.uniform(60, 150, 100)  # 20% wave points far off the lane
    coef = fit_lane(pd.DataFrame({"x": x, "y": y}))
    assert coef[0] == pytest.approx(0.02, abs=0.002)
    assert coef[1] == pytest.approx(650, abs=3)


def test_track_swimmer_returns_lane():
    cands, _ = scene_candidates()
    path, lane = track_swimmer(cands, N)
    assert abs(np.polyval(lane, 1000) - 680) < 10


# ---------- laps ----------

LEFT, RIGHT, SPEED = 250.0, 1700.0, 90.0  # px, px, px/s
T_LEN = (RIGHT - LEFT) / SPEED  # 16.11 s per length
WALL_TIMES = 2.0 + T_LEN * np.arange(4)  # start, turn L, turn R, finish L


def synthetic_track(fps=60, noise=3.0, seed=0):
    """Three lengths R->L->R->L at constant speed; the swimmer is invisible for
    1.5 s around each wall (push-off glide, turn) like the real clips."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, WALL_TIMES[-1] + 2, 1 / fps)
    x = np.full_like(t, np.nan)
    for k in range(3):
        a, b = WALL_TIMES[k], WALL_TIMES[k + 1]
        sel = (t >= a) & (t < b)
        frac = (t[sel] - a) / (b - a)
        x[sel] = RIGHT - frac * (RIGHT - LEFT) if k % 2 == 0 else LEFT + frac * (RIGHT - LEFT)
    x += rng.normal(0, noise, len(x))
    for w in WALL_TIMES:
        x[np.abs(t - w) < 1.5] = np.nan
    return t, x


def test_segments_directions():
    segs = segments(*synthetic_track())
    assert [s.direction for s in segs] == [-1, 1, -1]


def test_turn_vertices_find_walls_and_times():
    v = turn_vertices(segments(*synthetic_track()))
    np.testing.assert_allclose(v["t"], WALL_TIMES[1:3], atol=0.1)
    assert list(v["wall"]) == ["left", "right"]
    walls = wall_positions(v)
    assert walls["left"] == pytest.approx(LEFT, abs=10)
    assert walls["right"] == pytest.approx(RIGHT, abs=10)


def test_length_times_recover_known_splits():
    segs = segments(*synthetic_track())
    ev = wall_events(segs, wall_positions(turn_vertices(segs)))
    assert list(ev["event"]) == ["start", "turn", "turn", "finish"]
    lengths = length_times(ev)
    np.testing.assert_allclose(lengths["time_s"], T_LEN, atol=0.15)


def test_compare_lengths_and_summary(tmp_path):
    ref = tmp_path / "lengths.csv"
    ref.write_text("clip,length,time_s,strokes,annotator\nc,1,16.0,12,x\nc,2,16.5,12,x\n")
    tool = pd.DataFrame({"clip": ["c", "c"], "length": [1, 2], "time_s": [16.4, 16.3],
                         "from_event": ["start", "turn"], "to_event": ["turn", "finish"]})
    cmp = compare_lengths(tool, ref)
    np.testing.assert_allclose(cmp["err_s"], [0.4, -0.2], atol=1e-9)
    s = error_summary(cmp)
    assert s.loc["all", "mae_s"] == pytest.approx(0.3)
    assert s.loc["all", "bias_s"] == pytest.approx(0.1)

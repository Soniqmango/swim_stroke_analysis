"""Tests for push-off detection, per-length strokes, pool geometry (metres),
turn times, DPS and pose gating - synthetic data with known answers."""

import cv2
import numpy as np
import pandas as pd
import pytest

from swimstroke.lens import LensCalibration
from swimstroke.laps import detect_pushoff
from swimstroke.pipeline import Setup, gate_pose
from swimstroke.pool import PoolGeometry, crossing_time, distance_per_stroke, turn_times
from swimstroke.stabilize import CameraMotion
from swimstroke.strokes import strokes_in_length

# ---------- push-off ----------


def test_pushoff_found_at_onset_of_sustained_rise():
    t = np.arange(0, 5, 1 / 60)
    a = np.full_like(t, 300.0) + np.random.default_rng(0).normal(0, 20, len(t))
    a[(t > 0.4) & (t < 0.6)] += 400  # brief bump (head bobbing) - must be ignored
    rise = t >= 1.6
    a[rise] += np.minimum(1, (t[rise] - 1.6) / 0.8) * 1200  # push-off white water
    t0 = detect_pushoff(t, a)
    assert t0 == pytest.approx(1.6, abs=0.15)


def test_pushoff_none_without_activity():
    t = np.arange(0, 3, 1 / 60)
    assert detect_pushoff(t, np.full_like(t, 5.0)) is None


# ---------- strokes per length ----------


@pytest.mark.parametrize(
    "entries, start, end, expected",
    [
        ([2.0, 3.0, 4.0], 1.9, 4.1, 5),  # tight both ends: 2n-1
        ([2.0, 3.0, 4.0], 1.0, 4.1, 6),  # far arm before the first near entry
        ([2.0, 3.0, 4.0], 1.9, 5.0, 6),  # far arm after the last one
        ([2.0, 3.0, 4.0], 1.0, 5.0, 7),  # both -> odd counts are possible
    ],
)
def test_strokes_in_length(entries, start, end, expected):
    assert strokes_in_length(np.array(entries), start, end, period_s=1.0) == expected


def test_strokes_in_length_without_entries():
    assert strokes_in_length(np.array([]), 0, 10, 1.0) is None


# ---------- pool geometry ----------

# A camera looking at a 25 x 10 m pool surface: ground-truth homography metres -> px
H_TRUE = np.array([[60.0, 8.0, 200.0], [0.5, 25.0, 500.0], [0.0, 0.004, 1.0]])
CORNERS_M = np.array([[0, 0], [25, 0], [25, 10], [0, 10], [12.5, 0], [12.5, 10]], float)


def project(pts_m, H=H_TRUE):
    return cv2.perspectiveTransform(np.asarray(pts_m, float).reshape(-1, 1, 2), H).reshape(-1, 2)


def test_pool_geometry_maps_pixels_to_metres():
    g = PoolGeometry(project(CORNERS_M).tolist(), CORNERS_M.tolist())
    assert g.rms_m < 1e-6
    probe_m = np.array([[5.0, 2.0], [20.0, 7.5]])
    np.testing.assert_allclose(g.to_pool(project(probe_m)), probe_m, atol=1e-6)


def test_pool_geometry_with_lens_distortion(tmp_path):
    K = np.array([[700.0, 0, 960], [0, 700.0, 540], [0, 0, 1]])
    D = np.array([-0.2, 0.05, 0, 0, 0])
    calib = LensCalibration((1920, 1080), K.tolist(), D.tolist(), 0.1, 30, 0.8)
    lens_file = calib.save(tmp_path / "lens.json")

    def distort(px):  # ideal pixels -> what the distorted lens records
        norm = cv2.undistortPoints(px.reshape(-1, 1, 2), K, None)  # to normalised coords
        pts3 = np.c_[norm.reshape(-1, 2), np.ones(len(px))]
        out, _ = cv2.projectPoints(pts3, np.zeros(3), np.zeros(3), K, D)
        return out.reshape(-1, 2)

    g = PoolGeometry(distort(project(CORNERS_M)).tolist(), CORNERS_M.tolist(), lens_file=str(lens_file))
    assert g.rms_m < 0.01
    probe_m = np.array([[3.0, 1.0], [22.0, 9.0]])
    np.testing.assert_allclose(g.to_pool(distort(project(probe_m))), probe_m, atol=0.01)


def test_pool_geometry_rejects_collinear_points():
    line = np.array([[0, 5], [5, 5], [10, 5], [25, 5]], float)
    with pytest.raises(ValueError, match="collinear"):
        PoolGeometry(project(line).tolist(), line.tolist())


def test_pool_geometry_save_load(tmp_path):
    g = PoolGeometry(project(CORNERS_M).tolist(), CORNERS_M.tolist(), notes="test")
    back = PoolGeometry.load(g.save(tmp_path / "pool.json"))
    np.testing.assert_allclose(back.H, g.H)


# ---------- metrics in metres ----------


def v_track(speed_in=1.5, speed_out=1.5, t_turn=20.0, L=25.0):
    """Swimmer approaching the right wall (X=25) and leaving it again."""
    t = np.arange(0, 40, 1 / 60)
    X = np.where(t < t_turn, L - speed_in * (t_turn - t), L - speed_out * (t - t_turn))
    return t, X


def test_crossing_time():
    t, X = v_track()
    assert crossing_time(t, X, 20.0, 0, 20) == pytest.approx(20 - 5 / 1.5, abs=0.02)


def test_turn_time_known_answer():
    t, X = v_track(speed_in=1.5, speed_out=1.25)
    ev = pd.DataFrame({"t": [0.0, 20.0, 39.0], "event": ["start", "turn", "finish"], "wall": ["left", "right", "left"]})
    tt = turn_times(t, X, ev)
    expected = 5 / 1.5 + 5 / 1.25  # in to the wall + back out to 5 m
    assert tt.loc[0, "turn_time_s"] == pytest.approx(expected, abs=0.05)


def test_turn_time_with_invisible_glide():
    # swimmer not visible within 7 m of the wall after the turn (underwater):
    # the 5 m crossing is extrapolated from the visible swimming
    t, X = v_track(speed_in=1.5, speed_out=1.5)
    X = X.copy()
    X[(t > 20) & (X > 18)] = np.nan
    ev = pd.DataFrame({"t": [0.0, 20.0, 39.0], "event": ["start", "turn", "finish"], "wall": ["left", "right", "left"]})
    tt = turn_times(t, X, ev)
    assert tt.loc[0, "turn_time_s"] == pytest.approx(10 / 1.5, abs=0.1)


def test_distance_per_stroke():
    t = np.arange(0, 20, 1 / 60)
    X = 1.2 * t  # 1.2 m/s
    entries = np.array([2.0, 3.0, 4.0, 5.0])  # 1 s cycle -> 1.2 m per cycle -> 0.6 m per stroke
    assert distance_per_stroke(t, X, entries) == pytest.approx(0.6, abs=1e-6)


# ---------- pose gating ----------


def test_gate_pose_rejects_detections_away_from_swimmer():
    n = 4
    names = ["left_hip", "right_hip", "left_wrist"]
    cols = {"frame": range(n), "t": np.arange(n) / 60, "detected": [True] * n}
    hips_x = [100.0, 100.0, 900.0, 100.0]  # frame 2: a "person" far from the swimmer (umbrella)
    for nm in names:
        cols[f"{nm}_x"] = hips_x
        cols[f"{nm}_y"] = [50.0] * n
        cols[f"{nm}_vis"] = [0.9] * n
    pose = pd.DataFrame(cols)
    motion = CameraMotion(np.arange(n) / 60, np.stack([np.eye(3)] * n), (1920, 1080), 0, np.array([0]), np.array([1.0]))
    track = pd.DataFrame({"frame": range(n), "t": np.arange(n) / 60, "x": [110.0] * n, "y": [55.0] * n})
    gated, share = gate_pose(pose, motion, track, max_dist_px=200)
    assert gated["detected"].tolist() == [True, True, False, True]
    assert np.isnan(gated.loc[2, "left_wrist_x"])
    assert share == pytest.approx(0.75)


def test_setup_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("swimstroke.pipeline.SETUPS", tmp_path)
    s = Setup.load_or_create("pool1", "clip.mp4")
    s.lane, s.walls_px = [0.02, 650.0], {"left": 246.0}
    s.save()
    back = Setup.load_or_create("pool1", "other.mp4")
    assert back.ref_video == "clip.mp4" and back.walls_px == {"left": 246.0}

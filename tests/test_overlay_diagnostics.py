"""Tests for overlay drawing, diagnostics and the CLI argument parser."""

import numpy as np
import pandas as pd
import pytest

from swimstroke.cli import parse_args
from swimstroke.diagnostics import plot_diagnostics, summarize
from swimstroke.overlay import draw_hud, draw_pose

NAMES = ["nose", "left_shoulder", "left_wrist", "right_wrist"]


def _pose_table() -> pd.DataFrame:
    # 4 frames: detected, detected, missed, detected
    det = [True, True, False, True]
    vis_wrist = [0.9, 0.2, np.nan, 0.6]
    cols = {"frame": range(4), "t": [0.0, 0.1, 0.2, 0.3], "detected": det}
    for n in NAMES:
        cols[f"{n}_x"] = [10.0, 11.0, np.nan, 12.0]
        cols[f"{n}_y"] = [20.0, 21.0, np.nan, 22.0]
        cols[f"{n}_vis"] = vis_wrist if "wrist" in n else [1.0, 1.0, np.nan, 1.0]
    return pd.DataFrame(cols)


def test_summarize_counts_visibility_over_all_frames():
    s = summarize(_pose_table())
    assert s["frames"] == 4
    assert s["detection_rate"] == 0.75
    # wrist confidently visible in 2 of 4 frames (0.9 and 0.6), not 2 of 3 detected
    assert s["left_wrist_visible"] == 0.5
    assert s["left_shoulder_visible"] == 0.75


def test_plot_diagnostics_writes_png(tmp_path):
    out = plot_diagnostics(_pose_table(), tmp_path / "d.png")
    assert out.exists() and out.stat().st_size > 0


def test_draw_pose_handles_nan_and_draws_something():
    img = np.zeros((100, 100, 3), np.uint8)
    xy = np.array([[10, 10], [50, 50], [np.nan, np.nan], [80, 20]], float)
    vis = np.array([1.0, 1.0, np.nan, 0.1])
    draw_pose(img, xy, vis, NAMES, [(1, 2), (1, 3), (0, 1)])
    assert img.any()


def test_draw_hud_draws_text():
    img = np.zeros((540, 960, 3), np.uint8)
    draw_hud(img, 1.23, 37, detected=False, model="m")
    assert img.any()


def test_cli_defaults():
    a = parse_args(["clip.mp4"])
    assert a.model == "heavy" and a.max_frames is None and not a.force


def test_cli_rejects_unknown_model():
    with pytest.raises(SystemExit):
        parse_args(["clip.mp4", "--model", "huge"])

"""Tests for the pose table, using a fake estimator (no model download needed)."""

import numpy as np
import pytest

from swimstroke.pose import PoseFrame, landmark_names_in, load_pose, save_pose, track_video
from swimstroke.video import VideoWriter

W, H, FPS, N = 64, 48, 30, 10


class FakePose:
    """Detects a pose on even frames only, with coordinates derived from t."""

    name = "fake"
    landmark_names = ["nose", "left_wrist"]
    connections = [(0, 1)]

    def __init__(self):
        self.calls = []

    def process(self, image_bgr, t):
        self.calls.append(t)
        if len(self.calls) % 2 == 0:
            return PoseFrame(False, np.full((2, 2), np.nan), np.full(2, np.nan))
        xy = np.array([[t, 1.0], [2.0, 3.0]])
        return PoseFrame(True, xy, np.array([0.9, 0.4]))

    def close(self):
        pass


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    with VideoWriter(path, fps=FPS, size=(W, H)) as vw:
        for _ in range(N):
            vw.write(np.zeros((H, W, 3), dtype=np.uint8))
    return path


def test_track_video_table_layout(clip):
    est = FakePose()
    df = track_video(clip, est)
    assert len(df) == N
    assert list(df.columns[:3]) == ["frame", "t", "detected"]
    assert "left_wrist_vis" in df.columns
    assert df.attrs["model"] == "fake"
    # Timestamps passed to the estimator are the same ones stored in the table.
    np.testing.assert_allclose(df["t"], est.calls)


def test_undetected_frames_keep_row_with_nan(clip):
    df = track_video(clip, FakePose())
    assert df["detected"].tolist() == [True, False] * (N // 2)
    missed = df[~df["detected"]]
    assert missed["nose_x"].isna().all() and missed["left_wrist_vis"].isna().all()
    hit = df[df["detected"]]
    np.testing.assert_allclose(hit["nose_x"], hit["t"])
    assert (hit["left_wrist_vis"] == 0.4).all()


def test_parquet_round_trip(clip, tmp_path):
    df = track_video(clip, FakePose())
    path = tmp_path / "out" / "pose.parquet"
    save_pose(df, path)
    back = load_pose(path)
    assert back.equals(df)
    assert landmark_names_in(back) == ["nose", "left_wrist"]

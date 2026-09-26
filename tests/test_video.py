"""Tests for video I/O using small synthetic clips (no real footage needed)."""

import numpy as np
import pytest

from swimstroke.video import VideoWriter, iter_frames, probe

W, H, FPS, N = 64, 48, 30, 20


def _frame(i: int) -> np.ndarray:
    """Solid frame whose brightness encodes the frame index."""
    return np.full((H, W, 3), i * 10, dtype=np.uint8)


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    with VideoWriter(path, fps=FPS, size=(W, H)) as vw:
        for i in range(N):
            vw.write(_frame(i))
    return path


def test_probe(clip):
    info = probe(clip)
    assert (info.width, info.height) == (W, H)
    assert info.fps == pytest.approx(FPS, rel=0.01)
    assert info.n_frames == N


def test_iter_frames_count_order_and_content(clip):
    frames = list(iter_frames(clip))
    assert [f.idx for f in frames] == list(range(N))
    # Lossy compression changes pixels a little; brightness order must survive.
    means = [f.image.mean() for f in frames]
    assert means == sorted(means)


def test_timestamps_match_constant_frame_rate(clip):
    ts = np.array([f.t for f in iter_frames(clip)])
    assert ts[0] == pytest.approx(0.0, abs=1e-3)
    assert np.all(np.diff(ts) > 0)
    np.testing.assert_allclose(ts, np.arange(N) / FPS, atol=1e-3)


def test_max_frames(clip):
    assert len(list(iter_frames(clip, max_frames=5))) == 5


def test_writer_rejects_wrong_size(tmp_path):
    with VideoWriter(tmp_path / "x.mp4", fps=FPS, size=(W, H)) as vw:
        with pytest.raises(ValueError):
            vw.write(np.zeros((H + 2, W, 3), dtype=np.uint8))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "nope.mp4")

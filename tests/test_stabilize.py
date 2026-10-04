"""Stabilisation tests: recover a known camera drift on a synthetic scene."""

import cv2
import numpy as np
import pytest

from swimstroke.stabilize import (
    CameraMotion,
    _Matcher,
    estimate_motion,
    interpolate_homographies,
    to_reference,
    water_mask,
)
from swimstroke.video import VideoWriter

W, H = 640, 360


def scene(seed=0) -> np.ndarray:
    """Textured 'background' (shapes on noise) with a blue 'pool' band."""
    rng = np.random.default_rng(seed)
    # grey noise: random *colour* noise would be ~1/6 "blue" and get masked as water
    gray = cv2.GaussianBlur(rng.integers(0, 255, (H, W), dtype=np.uint8), (5, 5), 0)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for _ in range(60):
        b, g, r = rng.integers(0, 255, 3)
        c = (int(0.3 * b), int(g), int(r))  # BGR with weak blue: reddish/greenish shapes
        p = rng.integers(0, [W, H])
        cv2.rectangle(img, tuple(int(v) for v in p), tuple(int(v) for v in p + rng.integers(10, 50, 2)), c, -1)
    img[200:280] = (200, 140, 30)  # BGR blue-ish water
    return img


def drift(i: int) -> np.ndarray:
    """Known camera drift at frame i: slow shift + rotation + slight zoom."""
    a = np.deg2rad(0.02 * i)
    s = 1 + 0.0005 * i
    return np.array([[s * np.cos(a), -s * np.sin(a), 0.4 * i], [s * np.sin(a), s * np.cos(a), -0.25 * i], [0, 0, 1]])


def test_matcher_recovers_known_homography():
    ref = scene()
    T = drift(40)
    moved = cv2.warpPerspective(ref, T, (W, H), borderMode=cv2.BORDER_REFLECT)
    m = _Matcher(ref, scale=1.0)
    Hest, inl = m.homography(moved)
    # Hest maps moved -> ref, i.e. it should undo T
    corners = np.float32([[50, 50], [590, 50], [590, 310], [50, 310]])
    back = to_reference(to_reference(corners, T), Hest)
    np.testing.assert_allclose(back, corners, atol=0.5)
    assert inl > 0.8


def test_estimate_motion_on_video(tmp_path):
    ref = scene()
    path = tmp_path / "drift.mp4"
    n = 21
    with VideoWriter(path, fps=30, size=(W, H)) as vw:
        for i in range(n):
            vw.write(cv2.warpPerspective(ref, drift(i), (W, H), borderMode=cv2.BORDER_REFLECT))
    motion = estimate_motion(path, step=5, scale=1.0)
    assert motion.H.shape == (n, 3, 3)
    assert list(motion.key_frames) == [0, 5, 10, 15, 20]
    # interpolated frame 12 must also undo the drift (video compression adds noise)
    p = np.float32([[320, 100]])
    back = to_reference(to_reference(p, drift(12)), motion.H[12])
    np.testing.assert_allclose(back, p, atol=1.0)
    assert motion.drift_px().max() > 5


def test_interpolate_homographies_hits_key_frames_and_midpoints():
    size = (W, H)
    Hs = np.stack([np.eye(3), np.array([[1, 0, 10.0], [0, 1, -4.0], [0, 0, 1]])])
    out = interpolate_homographies(np.array([0, 10]), Hs, 11, size)
    np.testing.assert_allclose(out[0], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(to_reference(np.float32([[0, 0]]), out[5]), [[5.0, -2.0]], atol=1e-6)


def test_water_mask_finds_blue_band():
    m = water_mask(scene(), grow_px=0)
    assert m[240, 320] == 255  # inside the pool band
    assert m[:150].mean() < 255 * 0.2  # background mostly not water


def test_to_reference_keeps_nan():
    out = to_reference(np.array([[1.0, 2.0], [np.nan, np.nan]]), np.eye(3))
    np.testing.assert_allclose(out[0], [1, 2])
    assert np.isnan(out[1]).all()


def test_save_load(tmp_path):
    m = CameraMotion(np.arange(3.0), np.stack([np.eye(3)] * 3), (W, H), 0, np.array([0, 2]), np.array([1.0, 0.9]))
    back = CameraMotion.load(m.save(tmp_path / "m.npz"))
    np.testing.assert_allclose(back.H, m.H)
    assert back.image_size == (W, H)

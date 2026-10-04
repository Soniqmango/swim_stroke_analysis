"""Lens calibration tests: a synthetic ultra-wide camera with known K and
distortion must be recovered from simulated checkerboard views."""

import cv2
import numpy as np
import pytest

from swimstroke.lens import (
    LensCalibration,
    board_points,
    calibrate_from_corners,
    coverage,
    find_corners,
    undistort_points,
)

SIZE = (1920, 1080)
PATTERN = (9, 6)
SQUARE = 25.0  # mm
# Roughly a phone ultra-wide at 1080p: ~108 deg horizontal FOV, barrel distortion
K_TRUE = np.array([[700.0, 0, 960], [0, 700.0, 540], [0, 0, 1]])
D_TRUE = np.array([-0.25, 0.08, 0.0, 0.0, -0.01])


def synthetic_views(n=25, noise_px=0.1, seed=0):
    """Project the board in n random poses that stay fully inside the image."""
    rng = np.random.default_rng(seed)
    obj = board_points(PATTERN, SQUARE)
    centre = obj.mean(axis=0)
    views = []
    while len(views) < n:
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-350, 350), rng.uniform(-200, 200), rng.uniform(300, 650)])
        R, _ = cv2.Rodrigues(rvec)
        tvec = tvec - R @ centre  # rotate the board about its own centre
        pts, _ = cv2.projectPoints(obj, rvec, tvec, K_TRUE, D_TRUE)
        p = pts.reshape(-1, 2)
        if (p[:, 0] > 5).all() and (p[:, 0] < SIZE[0] - 5).all() and (p[:, 1] > 5).all() and (p[:, 1] < SIZE[1] - 5).all():
            views.append((pts + rng.normal(0, noise_px, pts.shape)).astype(np.float32))
    return views


def test_recovers_known_lens():
    calib = calibrate_from_corners(synthetic_views(), SIZE, PATTERN, SQUARE)
    K = calib.K_arr
    assert K[0, 0] == pytest.approx(700, rel=0.01)
    assert K[1, 1] == pytest.approx(700, rel=0.01)
    assert K[0, 2] == pytest.approx(960, abs=5)
    assert K[1, 2] == pytest.approx(540, abs=5)
    assert calib.dist[0] == pytest.approx(D_TRUE[0], abs=0.02)
    assert calib.rms_px < 0.2  # only the 0.1 px noise we added remains
    assert calib.coverage > 0.4  # property of the random test poses, not of the code


def test_undistort_points_straightens_a_line():
    # A straight line in the world, seen through the distorted lens, bends;
    # after undistortion it must be straight again.
    world = np.array([[x, 0.3, 1.0] for x in np.linspace(-1.2, 1.2, 15)])
    distorted, _ = cv2.projectPoints(world, np.zeros(3), np.zeros(3), K_TRUE, D_TRUE)
    distorted = distorted.reshape(-1, 2)
    calib = LensCalibration(SIZE, K_TRUE.tolist(), D_TRUE.tolist(), 0.0, 0, 0.0)

    def straightness(p):  # max distance from the best-fit line, px
        c = p - p.mean(axis=0)
        _, _, vt = np.linalg.svd(c)
        return np.abs(c @ vt[1]).max()

    assert straightness(distorted) > 20  # visibly curved
    assert straightness(undistort_points(distorted, calib)) < 0.05


def test_undistort_points_keeps_nan():
    calib = LensCalibration(SIZE, K_TRUE.tolist(), D_TRUE.tolist(), 0.0, 0, 0.0)
    out = undistort_points(np.array([[960.0, 540.0], [np.nan, np.nan]]), calib)
    np.testing.assert_allclose(out[0], [960, 540], atol=1e-6)  # centre doesn't move
    assert np.isnan(out[1]).all()


def test_find_corners_on_rendered_board():
    cols, rows, sq = PATTERN[0] + 1, PATTERN[1] + 1, 60  # squares = inner corners + 1
    board = np.full(((rows + 2) * sq, (cols + 2) * sq), 255, np.uint8)  # white margin
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 0
    h, w = board.shape
    dst = np.float32([[500, 200], [1400, 260], [1350, 900], [450, 820]])  # tilted view
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
    img = cv2.warpPerspective(board, H, SIZE, borderValue=200)
    corners = find_corners(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), PATTERN)
    assert corners is not None and len(corners) == PATTERN[0] * PATTERN[1]


def test_find_corners_returns_none_without_board():
    assert find_corners(np.full((480, 640, 3), 128, np.uint8), PATTERN) is None


def test_save_load_round_trip(tmp_path):
    calib = calibrate_from_corners(synthetic_views(n=12), SIZE, PATTERN, SQUARE)
    back = LensCalibration.load(calib.save(tmp_path / "c.json"))
    np.testing.assert_allclose(back.K_arr, calib.K_arr)
    assert back.image_size == SIZE and back.pattern == PATTERN


def test_coverage():
    pts = [np.array([[[10.0, 10.0]], [[1910.0, 1070.0]]])]
    assert coverage(pts, SIZE, grid=2) == 0.5  # 2 of 4 cells

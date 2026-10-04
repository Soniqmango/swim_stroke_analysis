"""Lens (intrinsic) calibration from a checkerboard video.

Why: the 0.5x ultra-wide lens bends straight lines, most strongly near the
image edges. The pool-plane homography (milestone 3) assumes a pinhole
camera where straight lines stay straight, so we first measure the lens
once and then correct every measured point before mapping it to metres.

How: film a printed checkerboard with the SAME phone, lens (0.5x) and video
mode (1080p60) as the swim clips. Zoom level, resolution and the iPhone's
"Lens Correction" setting all change the result, so they must match.
OpenCV finds the inner corners in many frames, and calibrateCamera solves
for the camera matrix K (focal length, optical centre) and the distortion
coefficients that best explain all views together.

We undistort *points* (wrist, hip, wall corners), not whole frames: it is
exact, cheap, and pose estimation keeps running on the original pixels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from swimstroke.video import iter_frames, probe


@dataclass
class LensCalibration:
    image_size: tuple[int, int]  # (width, height) the calibration is valid for
    K: list[list[float]]  # 3x3 camera matrix
    dist: list[float]  # distortion coefficients (OpenCV order)
    rms_px: float  # RMS reprojection error over all corners
    n_views: int
    coverage: float  # share of the image (8x8 grid) touched by corners
    per_view_rms_px: list[float] = field(default_factory=list)
    source: str = ""
    pattern: tuple[int, int] = (0, 0)  # inner corners (cols, rows)
    square_mm: float = 0.0

    @property
    def K_arr(self) -> np.ndarray:
        return np.asarray(self.K, dtype=float)

    @property
    def dist_arr(self) -> np.ndarray:
        return np.asarray(self.dist, dtype=float)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LensCalibration":
        d = json.loads(Path(path).read_text())
        d["image_size"] = tuple(d["image_size"])
        d["pattern"] = tuple(d["pattern"])
        return cls(**d)


def board_points(pattern: tuple[int, int], square: float) -> np.ndarray:
    """3D corner positions on the flat board (z=0), in the units of `square`."""
    cols, rows = pattern
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return np.hstack([grid * square, np.zeros((len(grid), 1))]).astype(np.float32)


def find_corners(image_bgr: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Inner checkerboard corners (N, 1, 2) float32, or None if not found.

    findChessboardCornersSB (sector-based) is more robust to blur and uneven
    light than the classic detector and is already sub-pixel accurate.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(
        gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    )
    return corners.astype(np.float32) if ok else None


def coverage(image_points: list[np.ndarray], image_size: tuple[int, int], grid: int = 8) -> float:
    """Share of an grid x grid partition of the image containing any corner.
    Distortion is strongest at the edges, so the board must visit them."""
    w, h = image_size
    hit = np.zeros((grid, grid), bool)
    for pts in image_points:
        p = pts.reshape(-1, 2)
        gx = np.clip((p[:, 0] / w * grid).astype(int), 0, grid - 1)
        gy = np.clip((p[:, 1] / h * grid).astype(int), 0, grid - 1)
        hit[gy, gx] = True
    return float(hit.mean())


def calibrate_from_corners(
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    pattern: tuple[int, int],
    square: float,
    rational: bool = False,
) -> LensCalibration:
    """Solve K and distortion from detected corners of several views.

    rational=True adds k4-k6 (CALIB_RATIONAL_MODEL) for strong wide-angle
    distortion; only use it with many well-spread views, or it overfits.
    """
    obj = board_points(pattern, square)
    object_points = [obj] * len(image_points)
    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=flags
    )
    per_view = []
    for pts, rv, tv in zip(image_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj - pts) ** 2, axis=2)))))
    return LensCalibration(
        image_size=tuple(image_size),
        K=K.tolist(),
        dist=dist.ravel().tolist(),
        rms_px=float(rms),
        n_views=len(image_points),
        coverage=coverage(image_points, image_size),
        per_view_rms_px=per_view,
        pattern=tuple(pattern),
        square_mm=float(square),
    )


def calibrate_from_video(
    video_path: str | Path,
    pattern: tuple[int, int],
    square_mm: float,
    sample_every: int = 10,
    max_views: int = 40,
    rational: bool = False,
    progress: bool = False,
) -> LensCalibration:
    """Detect the board in every `sample_every`-th frame, keep up to
    `max_views` spread evenly over the video, and calibrate.

    Consecutive frames are nearly identical and add no information, hence
    the sampling. One outlier pass then drops views that fit much worse than
    the rest (motion blur, a mis-detected corner) and recalibrates.
    """
    info = probe(video_path)
    size = (info.width, info.height)
    found: list[np.ndarray] = []
    for f in iter_frames(video_path):
        if f.idx % sample_every:
            continue
        c = find_corners(f.image, pattern)
        if c is not None:
            found.append(c)
        if progress:
            print(f"\r  frame {f.idx}: board found in {len(found)} frames", end="", flush=True)
    if progress:
        print()
    if len(found) < 8:
        raise RuntimeError(
            f"Checkerboard {pattern} found in only {len(found)} sampled frames; need >= 8. "
            "Check --cols/--rows (INNER corners), lighting and blur."
        )
    if len(found) > max_views:
        found = [found[i] for i in np.linspace(0, len(found) - 1, max_views).astype(int)]

    calib = calibrate_from_corners(found, size, pattern, square_mm, rational)
    errs = np.asarray(calib.per_view_rms_px)
    keep = errs <= max(3 * np.median(errs), 0.5)
    if (~keep).any() and keep.sum() >= 8:
        calib = calibrate_from_corners([c for c, k in zip(found, keep) if k], size, pattern, square_mm, rational)
    calib.source = str(video_path)
    return calib


def check_image_size(calib: LensCalibration, image_size: tuple[int, int]) -> None:
    """Refuse to apply a calibration to footage of a different resolution."""
    if tuple(image_size) != tuple(calib.image_size):
        raise ValueError(
            f"Lens calibration is for {calib.image_size}, video is {image_size}. "
            "Calibrate with the same resolution and lens."
        )


def undistort_points(points_xy: np.ndarray, calib: LensCalibration) -> np.ndarray:
    """Map distorted pixel coordinates (N, 2) to where an ideal pinhole camera
    with the same K would have seen them. NaN rows stay NaN."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    out = np.full_like(pts, np.nan)
    ok = ~np.isnan(pts).any(axis=1)
    if ok.any():
        # Undistortion has no closed form and is solved iteratively. OpenCV's
        # default (5 iterations) leaves ~0.15 px error at the edges of an
        # ultra-wide image, exactly where the pool walls are, so iterate to
        # convergence instead.
        und = cv2.undistortPoints(
            pts[ok].reshape(-1, 1, 2), calib.K_arr, calib.dist_arr, None, None, calib.K_arr,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9),
        )
        out[ok] = und.reshape(-1, 2)
    return out


def undistort_image(image: np.ndarray, calib: LensCalibration) -> np.ndarray:
    """Undistorted copy of a frame (for visual checks: pool edges should be straight)."""
    return cv2.undistort(image, calib.K_arr, calib.dist_arr)

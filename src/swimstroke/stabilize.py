"""Camera-motion compensation: map every frame onto one reference frame.

Even on a tripod the image drifts: the iPhone's video stabilisation keeps
re-cropping and warping, and tripod heads settle. On IMG_0115 the image
moves up to ~50 px and ~3% in scale over 40 s, which would turn into metres
of fake swimmer motion and break background subtraction.

Model: a homography per frame. A camera that only rotates (tripod) sees the
whole scene move by a homography regardless of depth, and the phone's own
stabilisation warp is also a homography, so this is exact rather than an
approximation.

Method: SIFT features on the static background (pool water masked out by
colour, since moving water produces features that don't follow the camera),
ratio-test matching against the reference frame, RANSAC homography. The
drift is slow and smooth, so we measure every `step`-th frame and interpolate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from swimstroke.video import iter_frames, probe


@dataclass
class CameraMotion:
    t: np.ndarray  # (N,) seconds, one per video frame
    H: np.ndarray  # (N, 3, 3) maps frame pixels -> reference-frame pixels
    image_size: tuple[int, int]  # (width, height)
    ref_frame: int
    key_frames: np.ndarray  # frames where H was measured (others interpolated)
    inlier_frac: np.ndarray  # RANSAC inlier share per key frame (quality)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, t=self.t, H=self.H, image_size=np.array(self.image_size),
            ref_frame=self.ref_frame, key_frames=self.key_frames, inlier_frac=self.inlier_frac,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CameraMotion":
        d = np.load(path)
        return cls(d["t"], d["H"], tuple(int(v) for v in d["image_size"]), int(d["ref_frame"]),
                   d["key_frames"], d["inlier_frac"])

    def drift_px(self) -> np.ndarray:
        """Max displacement of the four image corners per frame (px)."""
        w, h = self.image_size
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        moved = np.stack([to_reference(corners, H) for H in self.H])
        return np.linalg.norm(moved - corners, axis=2).max(axis=1)


def water_mask(image_bgr: np.ndarray, grow_px: int = 12) -> np.ndarray:
    """255 where the pixel looks like pool water (blue/cyan), grown a little
    so features on ripples near the edge are excluded too."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (85, 40, 80), (110, 255, 255))  # OpenCV hue: 0-180
    k = 2 * grow_px + 1
    return cv2.dilate(m, np.ones((k, k), np.uint8))


class _Matcher:
    """SIFT features of the reference frame, matched against later frames."""

    def __init__(self, ref_bgr: np.ndarray, scale: float, n_features: int = 2000):
        self.scale = scale
        self.sift = cv2.SIFT_create(n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.ref_kp, self.ref_des = self._features(ref_bgr)
        if self.ref_des is None or len(self.ref_kp) < 50:
            raise RuntimeError("Too few background features in the reference frame to stabilise.")
        s = scale
        self._S = np.diag([s, s, 1.0])  # full-res -> working-res
        self._S_inv = np.diag([1 / s, 1 / s, 1.0])

    def _features(self, bgr: np.ndarray):
        small = cv2.resize(bgr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray, cv2.bitwise_not(water_mask(small)))

    def homography(self, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        kp, des = self._features(bgr)
        if des is None or len(kp) < 8:
            return np.full((3, 3), np.nan), 0.0
        # Lowe's ratio test: keep a match only if clearly better than the runner-up
        good = [a for a, b in self.bf.knnMatch(des, self.ref_des, k=2) if a.distance < 0.75 * b.distance]
        if len(good) < 8:
            return np.full((3, 3), np.nan), 0.0
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 1.5)
        if H is None:
            return np.full((3, 3), np.nan), 0.0
        # H was estimated on downscaled images; convert to full-resolution pixels
        return self._S_inv @ H @ self._S, float(inl.mean())


def interpolate_homographies(key_idx: np.ndarray, key_H: np.ndarray, n: int, size: tuple[int, int]) -> np.ndarray:
    """Per-frame homographies from key-frame ones. Homography matrices can't be
    averaged directly, so we interpolate where the four image corners land and
    refit an exact homography through them."""
    w, h = size
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    key_dst = np.stack([to_reference(corners, H) for H in key_H])  # (K, 4, 2)
    frames = np.arange(n)
    out = np.empty((n, 3, 3))
    interp = np.empty((n, 4, 2))
    for c in range(4):
        for xy in range(2):
            interp[:, c, xy] = np.interp(frames, key_idx, key_dst[:, c, xy])
    for i in range(n):
        out[i] = cv2.getPerspectiveTransform(corners, interp[i].astype(np.float32))
    return out


def estimate_motion(
    video_path: str | Path,
    ref_frame: int = 0,
    step: int = 5,
    scale: float = 0.5,
    progress: bool = False,
) -> CameraMotion:
    """Homography from every frame to `ref_frame`."""
    info = probe(video_path)
    size = (info.width, info.height)
    ts, key_idx, key_H, key_q = [], [], [], []
    matcher = None
    for f in iter_frames(video_path):
        ts.append(f.t)
        if f.idx == ref_frame:
            matcher = _Matcher(f.image, scale)
        if matcher is None or (f.idx - ref_frame) % step:
            continue
        if f.idx == ref_frame:
            H, q = np.eye(3), 1.0
        else:
            H, q = matcher.homography(f.image)
        if not np.isnan(H).any():
            key_idx.append(f.idx)
            key_H.append(H)
            key_q.append(q)
        if progress and f.idx % 100 == 0:
            print(f"\r  stabilise: frame {f.idx}", end="", flush=True)
    if progress:
        print()
    if ref_frame > 0:
        raise NotImplementedError("Frames before ref_frame are not handled yet; use ref_frame=0.")
    key_idx = np.asarray(key_idx)
    H_all = interpolate_homographies(key_idx, np.asarray(key_H), len(ts), size)
    return CameraMotion(np.asarray(ts), H_all, size, ref_frame, key_idx, np.asarray(key_q))


def to_reference(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply one homography to (N, 2) points. NaN rows stay NaN."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    out = np.full_like(pts, np.nan)
    ok = ~np.isnan(pts).any(axis=1)
    if ok.any():
        out[ok] = cv2.perspectiveTransform(pts[ok].reshape(-1, 1, 2), H).reshape(-1, 2)
    return out


def warp_to_reference(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

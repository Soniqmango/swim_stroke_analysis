"""Swimmer tracking without pose: motion blobs + a globally optimal path.

When the swimmer is too small for pose estimation (or between pose
detections), they still show up as *something moving in the water*.

1. Candidates: every frame is warped onto the reference frame (stabilize.py)
   so the background is still, then a background model (MOG2) marks pixels
   that changed. Only pixels inside the pool water count. Connected groups of
   changed pixels become candidate blobs.
2. Path: waves, reflections and wake also move, so each frame has several
   candidates. Instead of guessing per frame, we pick the single sequence of
   candidates over the whole video that is most plausible as one swimmer:
   large blobs, small jumps between consecutive picks, few missing frames.
   That "lowest total cost path" is solved exactly with dynamic programming
   (like the Viterbi algorithm), so one wave can't drag the track away unless
   the whole path fits it better.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from swimstroke.stabilize import CameraMotion, warp_to_reference, water_mask
from swimstroke.video import iter_frames


@dataclass(frozen=True)
class BlobParams:
    scale: float = 0.5  # work at half resolution (enough for blobs, 4x faster)
    history: int = 300  # MOG2 frames of background memory (~5 s at 60 fps)
    var_threshold: float = 25.0  # MOG2 sensitivity; lower = more pixels flagged
    min_area_px: int = 6  # at working scale; smaller blobs are noise
    max_candidates: int = 8  # keep the largest N blobs per frame
    warmup_frames: int = 30  # MOG2 flags everything until it has a background


def find_candidates(
    video_path: str | Path,
    motion: CameraMotion,
    params: BlobParams = BlobParams(),
    progress: bool = False,
) -> pd.DataFrame:
    """One row per candidate blob: frame, t, x, y (reference-frame px, full res), area."""
    s = params.scale
    bg = cv2.createBackgroundSubtractorMOG2(params.history, params.var_threshold, detectShadows=False)
    kernel = np.ones((3, 3), np.uint8)
    pool = None
    rows = []
    for f in iter_frames(video_path):
        small = cv2.resize(warp_to_reference(f.image, motion.H[f.idx]), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if pool is None:
            # Water region of the (stabilised) first frame, shrunk so the
            # pool edge and deck reflections are excluded.
            pool = cv2.erode(water_mask(small, grow_px=0), np.ones((9, 9), np.uint8))
        fg = bg.apply(small)
        if f.idx < params.warmup_frames:
            continue
        fg = cv2.morphologyEx(cv2.bitwise_and(fg, pool), cv2.MORPH_OPEN, kernel)
        fg = cv2.dilate(fg, kernel, iterations=2)  # join splash fragments of one swimmer
        n, _, stats, cent = cv2.connectedComponentsWithStats(fg)
        blobs = [(stats[i, cv2.CC_STAT_AREA], cent[i]) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= params.min_area_px]
        blobs.sort(key=lambda b: -b[0])
        for area, (cx, cy) in blobs[: params.max_candidates]:
            rows.append((f.idx, f.t, cx / s, cy / s, area / s**2))
        if progress and f.idx % 200 == 0:
            print(f"\r  blobs: frame {f.idx}", end="", flush=True)
    if progress:
        print()
    return pd.DataFrame(rows, columns=["frame", "t", "x", "y", "area"])


@dataclass(frozen=True)
class PathParams:
    max_gap_frames: int = 180  # bridge up to 3 s at 60 fps (underwater glide after a turn)
    base_jump_px: float = 15.0  # allowed jitter between consecutive picks (splash moves the centroid)
    speed_px_per_frame: float = 3.0  # + this per frame of gap (IMG_0115 swimmer: ~2 px/frame)
    y_weight: float = 3.0  # vertical jumps count 3x: a swimmer stays in their lane,
    #                        waves and wake travel *across* the pool (mostly in y)
    hard_limit: float = 2.0  # jumps beyond hard_limit x allowed are impossible
    miss_cost: float = 1.0  # cost per frame without a pick
    area_weight: float = 0.5  # penalty for blobs smaller than ref_area (log-area)
    ref_area_px: float = 400.0  # a typical swimmer+splash blob at full resolution


def best_path(cands: pd.DataFrame, n_frames: int, params: PathParams = PathParams()) -> pd.DataFrame:
    """Pick at most one candidate per frame along the lowest-cost path.

    Costs (lower is better):
      - per pick: area_weight * log(ref_area / area), only if smaller than ref_area.
        Never negative: otherwise any long chain of waves "earns" its way
        into the path and beats honestly leaving a gap.
      - per jump: (distance / allowed)^2, impossible beyond hard_limit x allowed,
        where allowed = base_jump + speed * frames_elapsed and
        distance = sqrt(dx^2 + (y_weight * dy)^2)
      - per skipped frame: miss_cost       (prefer continuous tracks)
    The path may start and end anywhere; frames outside it cost miss_cost too.
    Returns one row per video frame (x, y NaN where the path has no pick).

    Limitation: while the swimmer is invisible (underwater after a turn) the
    path may follow another blob near its last position, e.g. a wave crossing
    the lane. Waves move across the pool, not along it, so lap timing (which
    only uses stretches of steady motion along the lane) ignores such picks.
    """
    c = cands.sort_values(["frame", "area"], ascending=[True, False]).reset_index(drop=True)
    fr = c["frame"].to_numpy()
    xy = c[["x", "y"]].to_numpy()
    emit = params.area_weight * np.maximum(0.0, np.log(params.ref_area_px / c["area"].to_numpy()))
    w = np.array([1.0, params.y_weight])
    n = len(c)
    best = np.empty(n)  # cost of the best path ending at candidate i, incl. misses before it
    prev = np.full(n, -1)
    starts = np.searchsorted(fr, np.arange(n_frames + 1))  # candidate index range per frame

    for i in range(n):
        f = fr[i]
        best[i] = params.miss_cost * f + emit[i]  # start the path here
        lo = starts[max(0, f - params.max_gap_frames - 1)]
        hi = starts[f]
        if hi > lo:
            gap = f - fr[lo:hi]  # frames elapsed (>= 1)
            allowed = params.base_jump_px + params.speed_px_per_frame * gap
            d = np.linalg.norm((xy[lo:hi] - xy[i]) * w, axis=1)
            jump = (d / allowed) ** 2
            jump[d > params.hard_limit * allowed] = np.inf
            total = best[lo:hi] + jump + params.miss_cost * (gap - 1) + emit[i]
            j = int(np.argmin(total))
            if total[j] < best[i]:
                best[i] = total[j]
                prev[i] = lo + j

    out = pd.DataFrame({"frame": np.arange(n_frames), "x": np.nan, "y": np.nan, "area": np.nan})
    if n == 0:
        return out
    end_cost = best + params.miss_cost * (n_frames - 1 - fr)  # frames after the path ends
    i = int(np.argmin(end_cost))
    while i >= 0:
        out.loc[fr[i], ["x", "y", "area"]] = (xy[i, 0], xy[i, 1], c["area"].iat[i])
        i = prev[i]
    return out


def fit_lane(path: pd.DataFrame, deg: int = 1, iters: int = 5) -> np.ndarray:
    """Robust polynomial y = f(x) through the tracked points: the swimmer's lane
    as seen in the image. Each iteration drops points more than 3 robust
    standard deviations away, so stretches where the path followed waves
    don't bend the lane.

    Degree 1 by default: on IMG_0115-0117 the lane's curvature from the wide
    lens is negligible (quadratic term ~5e-7), while a quadratic was flexible
    enough to bend towards a cluster of waves at the end of IMG_0116."""
    p = path.dropna(subset=["x", "y"])
    x, y = p["x"].to_numpy(), p["y"].to_numpy()
    keep = np.ones(len(x), bool)
    coef = np.polyfit(x, y, deg)
    for _ in range(iters):
        coef = np.polyfit(x[keep], y[keep], deg)
        r = y - np.polyval(coef, x)
        med = np.median(r[keep])
        mad = np.median(np.abs(r[keep] - med))
        # Centre on the median residual, not 0: a fit pulled by outliers shifts
        # all inliers off zero, and thresholding |r| would then reject them.
        keep = np.abs(r - med) <= 3 * 1.4826 * max(mad, 1.0)  # 1.4826*MAD ~ std for normal noise
    return coef


def wall_activity(
    video_path: str | Path,
    motion: CameraMotion,
    roi: tuple[int, int, int, int],
    empty_frames: list[int],
    t_max: float,
) -> pd.DataFrame:
    """Per-frame count of pixels in `roi` (x0, x1, y0, y1; reference px) that
    are normally water but are not water-coloured now.

    "Normally water" comes from frames where the swimmer is known to be far
    away (`empty_frames`), so fixed objects (ladder, umbrella, deck) never
    count, and ripples stay blue so they don't count either. What does count
    is the swimmer's body and the white water they churn up: at a push-off
    this count jumps, which marks the start.
    """
    x0, x1, y0, y1 = roi
    need = set(empty_frames)
    last = max(empty_frames)
    masks: dict[int, np.ndarray] = {}
    times: dict[int, float] = {}
    for f in iter_frames(video_path, max_frames=last + 1):
        if f.t > t_max and f.idx not in need:
            continue
        crop = warp_to_reference(f.image, motion.H[f.idx])[y0:y1, x0:x1]
        masks[f.idx] = water_mask(crop, grow_px=0) > 0
        times[f.idx] = f.t
    normally_water = np.mean([masks[i] for i in empty_frames], axis=0) > 0.8
    rows = [(i, times[i], int((normally_water & ~masks[i]).sum())) for i in sorted(masks) if times[i] <= t_max]
    return pd.DataFrame(rows, columns=["frame", "t", "activity"])


def lane_residual(cands: pd.DataFrame, lane: np.ndarray) -> np.ndarray:
    return cands["y"].to_numpy() - np.polyval(lane, cands["x"].to_numpy())


def track_swimmer(
    cands: pd.DataFrame,
    n_frames: int,
    params: PathParams = PathParams(),
    lane_tol_px: float = 25.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Two passes: rough path -> fit the lane -> path through near-lane candidates only."""
    rough = best_path(cands, n_frames, params)
    lane = fit_lane(rough)
    near = cands[np.abs(lane_residual(cands, lane)) <= lane_tol_px]
    return best_path(near, n_frames, params), lane

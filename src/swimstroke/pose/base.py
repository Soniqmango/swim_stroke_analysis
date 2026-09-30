"""Model-independent pose types and the per-video pose table.

Every pose backend (MediaPipe now, maybe YOLO later) implements PoseEstimator
and returns a PoseFrame per image. Everything downstream (stroke detection,
tracking, overlay) only sees the DataFrame built by `track_video`, so
switching models never touches the analysis code, and model comparisons
are apples to apples.

Pose table layout: one row per video frame
    frame, t, detected, <name>_x, <name>_y, <name>_vis   (for each landmark)
x/y are pixels in the original frame; vis is the model's 0-1 confidence
that the point is visible (not occluded, e.g. by water). Undetected frames
keep their row (so the timeline has no gaps) with NaN coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from swimstroke.video import iter_frames


@dataclass
class PoseFrame:
    detected: bool
    xy: np.ndarray  # (K, 2) float, pixel coordinates
    vis: np.ndarray  # (K,) float in [0, 1]


class PoseEstimator(Protocol):
    name: str
    landmark_names: list[str]
    connections: list[tuple[int, int]]  # skeleton edges as landmark index pairs

    def process(self, image_bgr: np.ndarray, t: float) -> PoseFrame:
        """Estimate the pose in one frame. `t` (seconds) must increase between calls."""
        ...

    def close(self) -> None: ...


def track_video(
    video_path: str | Path,
    estimator: PoseEstimator,
    max_frames: int | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Run `estimator` on every frame and return the pose table."""
    names = estimator.landmark_names
    k = len(names)
    rows_t, rows_det, xy, vis = [], [], [], []
    for f in iter_frames(video_path, max_frames=max_frames):
        pf = estimator.process(f.image, f.t)
        rows_t.append(f.t)
        rows_det.append(pf.detected)
        if pf.detected:
            xy.append(pf.xy)
            vis.append(pf.vis)
        else:
            xy.append(np.full((k, 2), np.nan))
            vis.append(np.full(k, np.nan))
        if progress and f.idx % 100 == 0:
            print(f"\r  pose: frame {f.idx}", end="", flush=True)
    if progress:
        print()

    n = len(rows_t)
    xy_arr = np.asarray(xy).reshape(n, k, 2)
    vis_arr = np.asarray(vis).reshape(n, k)
    cols: dict[str, np.ndarray] = {
        "frame": np.arange(n),
        "t": np.asarray(rows_t),
        "detected": np.asarray(rows_det, dtype=bool),
    }
    for i, name in enumerate(names):
        cols[f"{name}_x"] = xy_arr[:, i, 0]
        cols[f"{name}_y"] = xy_arr[:, i, 1]
        cols[f"{name}_vis"] = vis_arr[:, i]
    df = pd.DataFrame(cols)
    df.attrs["model"] = estimator.name
    return df


def landmark_names_in(df: pd.DataFrame) -> list[str]:
    """Recover landmark names (in order) from a pose table's columns."""
    return [c[: -len("_vis")] for c in df.columns if c.endswith("_vis")]


def save_pose(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_pose(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

"""Draw pose results onto video frames.

The overlay is a debugging tool first: it should make it obvious at a glance
where tracking works and where it fails. So:
  - left side is orange, right side is cyan (catches left/right swaps)
  - confident points (vis >= threshold) are solid, low-confidence ones are
    small grey outlines (the model is guessing, typically underwater)
  - a HUD shows time, frame number and whether a pose was detected

It draws from the pose table (not from the model), so the overlay can be
re-rendered from the cached Parquet file without re-running pose estimation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from swimstroke.pose.base import landmark_names_in
from swimstroke.video import VideoWriter, iter_frames, probe

# BGR colours
LEFT = (0, 140, 255)  # orange
RIGHT = (255, 200, 0)  # cyan
CENTRE = (255, 255, 255)
LOW = (140, 140, 140)
GREEN = (80, 220, 80)
RED = (60, 60, 230)

# Eyes, ears and mouth are only a few pixels apart at pool-deck distance and
# just add clutter; the nose alone marks the head.
_FACE = ("eye", "ear", "mouth")


def _colour(name: str) -> tuple[int, int, int]:
    if name.startswith("left_"):
        return LEFT
    if name.startswith("right_"):
        return RIGHT
    return CENTRE


def draw_pose(
    image: np.ndarray,
    xy: np.ndarray,
    vis: np.ndarray,
    names: list[str],
    connections: list[tuple[int, int]],
    vis_threshold: float = 0.5,
) -> None:
    """Draw one skeleton in place. xy: (K, 2) pixels, vis: (K,)."""
    s = max(1, round(image.shape[0] / 540))  # scale strokes with resolution
    shown = [not any(f in n for f in _FACE) for n in names]
    ok = ~np.isnan(xy).any(axis=1)
    conf = ok & (np.nan_to_num(vis) >= vis_threshold)

    def pt(i: int) -> tuple[int, int]:
        return int(round(xy[i, 0])), int(round(xy[i, 1]))

    for a, b in connections:
        if not (shown[a] and shown[b] and ok[a] and ok[b]):
            continue
        if conf[a] and conf[b]:
            # colour the bone by its side if both ends agree, else white
            ca, cb = _colour(names[a]), _colour(names[b])
            cv2.line(image, pt(a), pt(b), ca if ca == cb else CENTRE, 2 * s, cv2.LINE_AA)
        else:
            cv2.line(image, pt(a), pt(b), LOW, s, cv2.LINE_AA)

    for i, name in enumerate(names):
        if not (shown[i] and ok[i]):
            continue
        if conf[i]:
            cv2.circle(image, pt(i), 3 * s, _colour(name), -1, cv2.LINE_AA)
        else:
            cv2.circle(image, pt(i), 2 * s, LOW, s, cv2.LINE_AA)


def draw_hud(image: np.ndarray, t: float, frame: int, detected: bool, model: str = "") -> None:
    s = max(1, round(image.shape[0] / 540))
    label = "POSE" if detected else "NO POSE"
    lines = [(f"t={t:6.2f}s  frame {frame}", CENTRE), (label, GREEN if detected else RED)]
    if model:
        lines.append((model, LOW))
    y = 14 * s
    for text, colour in lines:
        # dark outline first so the text stays readable on bright water
        cv2.putText(image, text, (8 * s, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 0, 0), 3 * s, cv2.LINE_AA)
        cv2.putText(image, text, (8 * s, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, colour, s, cv2.LINE_AA)
        y += 14 * s


def write_overlay(
    video_path: str | Path,
    pose: pd.DataFrame,
    connections: list[tuple[int, int]],
    out_path: str | Path,
    vis_threshold: float = 0.5,
) -> Path:
    """Render the pose table on top of the source video."""
    info = probe(video_path)
    names = landmark_names_in(pose)
    xs = pose[[f"{n}_x" for n in names]].to_numpy()
    ys = pose[[f"{n}_y" for n in names]].to_numpy()
    vis = pose[[f"{n}_vis" for n in names]].to_numpy()
    model = pose.attrs.get("model", "")

    with VideoWriter(out_path, fps=info.fps, size=(info.width, info.height)) as vw:
        for f in iter_frames(video_path, max_frames=len(pose)):
            img = f.image
            row = pose.iloc[f.idx]
            if row["detected"]:
                xy = np.stack([xs[f.idx], ys[f.idx]], axis=1)
                draw_pose(img, xy, vis[f.idx], names, connections, vis_threshold)
            draw_hud(img, f.t, f.idx, bool(row["detected"]), model)
            vw.write(img)
    return Path(out_path)

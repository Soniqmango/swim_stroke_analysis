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


def draw_hud(
    image: np.ndarray,
    t: float,
    frame: int,
    detected: bool,
    model: str = "",
    cycles: int | None = None,
    entry: bool = False,
) -> None:
    s = max(1, round(image.shape[0] / 540))
    label = "POSE" if detected else "NO POSE"
    lines = [(f"t={t:6.2f}s  frame {frame}", CENTRE), (label, GREEN if detected else RED)]
    if model:
        lines.append((model, LOW))
    if cycles is not None:
        lines.append((f"cycles {cycles}  strokes {2 * cycles}", CENTRE))
    if entry:
        lines.append(("HAND ENTRY", LEFT))
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
    entries: np.ndarray | None = None,
    flash_s: float = 0.3,
    markers: np.ndarray | None = None,
    events: pd.DataFrame | None = None,
) -> Path:
    """Render the pose table on top of the source video.

    `entries`: detected hand-entry times (s). The HUD shows a running cycle
    count and flashes HAND ENTRY for `flash_s` after each one.
    `markers`: (N, 2) per-frame swimmer position in this video's pixels
    (NaN = not tracked), drawn as a ring.
    `events`: wall events (t, event); the HUD shows the last one.
    """
    entries = None if entries is None else np.sort(np.asarray(entries, float))
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
            cycles, entry = None, False
            if entries is not None:
                cycles = int(np.searchsorted(entries, f.t, side="right"))
                entry = cycles > 0 and f.t - entries[cycles - 1] < flash_s
            if markers is not None and f.idx < len(markers) and not np.isnan(markers[f.idx]).any():
                s = max(1, round(img.shape[0] / 540))
                cv2.circle(img, (int(markers[f.idx, 0]), int(markers[f.idx, 1])), 14 * s, GREEN, 2 * s, cv2.LINE_AA)
            draw_hud(img, f.t, f.idx, bool(row["detected"]), model, cycles, entry)
            if events is not None and len(events):
                past = events[events["t"] <= f.t]
                if len(past):
                    e = past.iloc[-1]
                    label = f"{e['event'].upper()} @ {e['t']:.2f}s"
                    s = max(1, round(img.shape[0] / 540))
                    cv2.putText(img, label, (8 * s, img.shape[0] - 10 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 0, 0), 3 * s, cv2.LINE_AA)
                    cv2.putText(img, label, (8 * s, img.shape[0] - 10 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, GREEN, s, cv2.LINE_AA)
            vw.write(img)
    return Path(out_path)


# event -> (limb, phase that starts at this event); see swimstroke.kinematics
PHASE_STARTS = {
    "leg_recovery_end": ("LEGS", "KICK"),
    "kick_end": ("LEGS", "GLIDE"),
    "leg_recovery_start": ("LEGS", "RECOVERY"),
    "arm_extension": ("ARMS", "GLIDE"),
    "pull_start": ("ARMS", "PULL"),
    "pull_in": ("ARMS", "RECOVERY"),
}


def phase_at(events: pd.DataFrame, t: float, limb: str) -> str | None:
    """Phase of one limb at time t: the phase started by the latest event."""
    keys = [e for e, (lb, _) in PHASE_STARTS.items() if lb == limb]
    past = events[(events["t"] <= t) & events["event"].isin(keys)]
    return PHASE_STARTS[past.iloc[-1]["event"]][1] if len(past) else None


def write_kinematics_overlay(
    video_path: str | Path,
    pose: pd.DataFrame,
    connections: list[tuple[int, int]],
    events: pd.DataFrame,
    out_path: str | Path,
    vis_threshold: float = 0.3,
    flash_s: float = 0.25,
) -> Path:
    """Render the gated pose with breaststroke phases, for checking by eye.

    `events`: (cycle, event, t) from swimstroke.kinematics.analyze. The HUD
    shows the cycle number, the current leg and arm phase, and flashes each
    event's name for `flash_s` after it. Frames the gate rejected show
    NO POSE, so it's visible where the analysis had nothing to go on.
    """
    events = events.sort_values("t", kind="stable")
    info = probe(video_path)
    names = landmark_names_in(pose)
    xs = pose[[f"{n}_x" for n in names]].to_numpy()
    ys = pose[[f"{n}_y" for n in names]].to_numpy()
    vis = pose[[f"{n}_vis" for n in names]].to_numpy()

    with VideoWriter(out_path, fps=info.fps, size=(info.width, info.height)) as vw:
        for f in iter_frames(video_path, max_frames=len(pose)):
            img = f.image
            detected = bool(pose.iloc[f.idx]["detected"])
            if detected:
                draw_pose(img, np.stack([xs[f.idx], ys[f.idx]], axis=1), vis[f.idx], names, connections, vis_threshold)
            draw_hud(img, f.t, f.idx, detected)
            s = max(1, round(img.shape[0] / 540))
            past = events[events["t"] <= f.t]
            lines = []
            if len(past):
                lines.append((f"cycle {int(past.iloc[-1]['cycle']) + 1}", CENTRE))
            for limb, colour in (("LEGS", LEFT), ("ARMS", RIGHT)):
                ph = phase_at(events, f.t, limb)
                if ph:
                    lines.append((f"{limb}: {ph}", colour))
            recent = past[f.t - past["t"] < flash_s]
            lines += [(e.upper().replace("_", " "), GREEN) for e in recent["event"]]
            y = img.shape[0] - 10 * s - 16 * s * (len(lines) - 1)
            for text, colour in lines:
                cv2.putText(img, text, (8 * s, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 0, 0), 3 * s, cv2.LINE_AA)
                cv2.putText(img, text, (8 * s, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, colour, s, cv2.LINE_AA)
                y += 16 * s
            vw.write(img)
    return Path(out_path)

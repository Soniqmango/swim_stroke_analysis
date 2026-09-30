"""Pose-quality diagnostics: how often and how well the swimmer is tracked.

These numbers decide whether MediaPipe is good enough or whether we need a
different approach (cropping, YOLO, ...), so they are computed the same way
for every clip and every model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Landmarks that matter for this project: arms for strokes, head/hip for position.
KEY_LANDMARKS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
]


def summarize(pose: pd.DataFrame, vis_threshold: float = 0.5) -> dict[str, float]:
    """Detection rate plus, per key landmark, the share of *all* frames where
    it is confidently visible (a wrist the model can't see is no use to us,
    whether the pose was missed entirely or the arm was underwater)."""
    duration = float(pose["t"].iloc[-1] - pose["t"].iloc[0]) if len(pose) > 1 else 0.0
    out: dict[str, float] = {
        "frames": len(pose),
        "duration_s": round(duration, 2),
        "detection_rate": round(float(pose["detected"].mean()), 3),
    }
    for name in KEY_LANDMARKS:
        col = f"{name}_vis"
        if col in pose:
            out[f"{name}_visible"] = round(float((pose[col] >= vis_threshold).mean()), 3)
    return out


def plot_diagnostics(pose: pd.DataFrame, out_path: str | Path, vis_threshold: float = 0.5) -> Path:
    """Detection strip + wrist/elbow visibility over time."""
    t = pose["t"].to_numpy()
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [1, 4]}
    )

    det = pose["detected"].to_numpy().astype(float)
    ax0.fill_between(t, 0, det, step="mid", color="#2a9d8f", linewidth=0)
    ax0.set_yticks([])
    ax0.set_ylabel("pose\ndetected", rotation=0, ha="right", va="center")
    ax0.set_title(f"{pose.attrs.get('model', 'pose')} - detected in {det.mean():.0%} of frames")

    series = [
        ("left_wrist", "#e76f51", "-"),
        ("right_wrist", "#2a78d6", "-"),
        ("left_elbow", "#e76f51", ":"),
        ("right_elbow", "#2a78d6", ":"),
    ]
    for name, colour, style in series:
        if f"{name}_vis" in pose:
            ax1.plot(t, pose[f"{name}_vis"], style, color=colour, lw=1.2, label=name.replace("_", " "))
    ax1.axhline(vis_threshold, color="grey", lw=0.8, ls="--")
    ax1.text(t[-1], vis_threshold, " threshold", va="center", color="grey", fontsize=8)
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("visibility")
    ax1.set_xlabel("time (s)")
    ax1.legend(loc="upper left", ncol=4, frameon=False, fontsize=9)
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

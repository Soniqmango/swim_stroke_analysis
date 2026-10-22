"""RTMPose backend (via rtmlib), the pose model Sports2D uses.

Two-stage: a person detector (YOLOX) finds the swimmer's box, then RTMPose
locates keypoints inside it. Unlike MediaPipe it re-detects every frame
instead of tracking, so one bad frame (splash, bubbles) does not make it
lose the swimmer for the next few.

Default model is HALPE-26 ("body_with_feet" in Sports2D): the 17 COCO
keypoints plus head, neck, mid-hip, big/small toes and heels. Feet matter
for kick frequency and breaststroke arm-leg coordination.

Landmark names match MediaPipe's where both models have the point
(left_wrist, right_hip, ...), so stroke detection, overlays and diagnostics
work on either model's pose table without changes.

`vis` is RTMPose's keypoint score clipped to [0, 1]. It is a confidence,
like MediaPipe's visibility, but not calibrated the same way: compare
models on their own recommended thresholds (Sports2D uses 0.3).
"""

from __future__ import annotations

import numpy as np

from swimstroke.pose.base import PoseFrame

HALPE26 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck", "mid_hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
]

_EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_ankle", "left_big_toe"),
    ("right_ankle", "right_heel"), ("right_ankle", "right_big_toe"),
    ("head", "neck"), ("neck", "mid_hip"), ("nose", "head"),
]

MODES = ("lightweight", "balanced", "performance")  # faster -> more accurate


class RTMPose:
    landmark_names = HALPE26
    connections = [(HALPE26.index(a), HALPE26.index(b)) for a, b in _EDGES]

    def __init__(self, mode: str = "balanced", device: str = "cpu"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        from rtmlib import BodyWithFeet  # imported here: model download on first use

        self.name = f"rtmpose-{mode}"
        self._model = BodyWithFeet(mode=mode, backend="onnxruntime", device=device)

    def process(self, image_bgr: np.ndarray, t: float) -> PoseFrame:
        keypoints, scores = self._model(image_bgr)  # (P, 26, 2), (P, 26); rtmlib takes BGR
        return pick_swimmer(keypoints, scores, len(self.landmark_names))

    def close(self) -> None:
        pass

    def __enter__(self) -> "RTMPose":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def pick_swimmer(keypoints, scores, k: int) -> PoseFrame:
    """One swimmer per clip: keep the most confident person (others are
    usually reflections on the surface or people on deck)."""
    if keypoints is None or len(keypoints) == 0:
        return PoseFrame(False, np.full((k, 2), np.nan), np.full(k, np.nan))
    scores = np.asarray(scores, dtype=float)
    best = int(np.argmax(scores.mean(axis=1)))
    xy = np.asarray(keypoints[best], dtype=float)
    vis = np.clip(scores[best], 0.0, 1.0)
    return PoseFrame(True, xy, vis)

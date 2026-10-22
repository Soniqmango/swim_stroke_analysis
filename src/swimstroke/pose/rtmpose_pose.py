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
        self._last_hip: np.ndarray | None = None

    def process(self, image_bgr: np.ndarray, t: float) -> PoseFrame:
        keypoints, scores = self._model(image_bgr)  # (P, 26, 2), (P, 26); rtmlib takes BGR
        pf = pick_swimmer(keypoints, scores, len(self.landmark_names), self._last_hip)
        if pf.detected:
            self._last_hip = _hip(pf.xy)
        return pf

    def close(self) -> None:
        pass

    def __enter__(self) -> "RTMPose":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_L_HIP, _R_HIP = HALPE26.index("left_hip"), HALPE26.index("right_hip")
_L_SH, _R_SH = HALPE26.index("left_shoulder"), HALPE26.index("right_shoulder")


def _hip(xy: np.ndarray) -> np.ndarray:
    return (xy[_L_HIP] + xy[_R_HIP]) / 2


def _torso(xy: np.ndarray) -> float:
    return float(np.linalg.norm((xy[_L_SH] + xy[_R_SH]) / 2 - _hip(xy)))


def pick_swimmer(keypoints, scores, k: int, last_hip: np.ndarray | None = None) -> PoseFrame:
    """One swimmer per clip. Follow the same person: among people whose hips
    are within one torso length of the last pick, keep the most confident.
    With nobody near (first frame, or the swimmer was lost), fall back to the
    most confident person overall. Without this, two swimmers in view (a
    second lane, someone passing behind) make the pick flip between them."""
    if keypoints is None or len(keypoints) == 0:
        return PoseFrame(False, np.full((k, 2), np.nan), np.full(k, np.nan))
    kps = np.asarray(keypoints, dtype=float)
    conf = np.asarray(scores, dtype=float).mean(axis=1)
    candidates = np.arange(len(kps))
    if last_hip is not None:
        near = [i for i in candidates if np.linalg.norm(_hip(kps[i]) - last_hip) <= max(_torso(kps[i]), 1.0)]
        if near:
            candidates = np.array(near)
    best = int(candidates[np.argmax(conf[candidates])])
    vis = np.clip(np.asarray(scores[best], dtype=float), 0.0, 1.0)
    return PoseFrame(True, kps[best], vis)

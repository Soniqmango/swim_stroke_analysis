"""MediaPipe Pose Landmarker backend (33 landmarks, Tasks API).

Runs in VIDEO mode: after the first detection, MediaPipe tracks the person
from frame to frame instead of re-detecting from scratch each time. That
helps when the swimmer is briefly hidden by splash.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from swimstroke.pose.base import PoseFrame

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)
VARIANTS = ("lite", "full", "heavy")  # faster → more accurate


def ensure_model(variant: str = "heavy", models_dir: str | Path = "models") -> Path:
    """Return the path to the model file, downloading it on first use."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    path = Path(models_dir) / f"pose_landmarker_{variant}.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe pose model ({variant}) → {path}")
        urllib.request.urlretrieve(MODEL_URL.format(v=variant), path)
    return path


class MediaPipePose:
    name = "mediapipe"
    landmark_names = [lm.name.lower() for lm in vision.PoseLandmark]
    connections = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]

    def __init__(
        self,
        variant: str = "heavy",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        models_dir: str | Path = "models",
    ):
        self.name = f"mediapipe-{variant}"
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(variant, models_dir))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,  # one swimmer per video, for now
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, image_bgr: np.ndarray, t: float) -> PoseFrame:
        h, w = image_bgr.shape[:2]
        # VIDEO mode requires strictly increasing integer-millisecond timestamps.
        ts_ms = max(round(t * 1000), self._last_ms + 1)
        self._last_ms = ts_ms
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # MediaPipe expects RGB
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
        )
        if not result.pose_landmarks:
            k = len(self.landmark_names)
            return PoseFrame(False, np.full((k, 2), np.nan), np.full(k, np.nan))
        lms = result.pose_landmarks[0]
        # MediaPipe returns coordinates normalised to [0, 1]; convert to pixels.
        xy = np.array([[lm.x * w, lm.y * h] for lm in lms])
        vis = np.array([lm.visibility for lm in lms])
        return PoseFrame(True, xy, vis)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MediaPipePose":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

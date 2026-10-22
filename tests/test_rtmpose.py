"""Tests for the RTMPose backend's output handling (no model download needed)."""

import numpy as np

from swimstroke.pose.mediapipe_pose import MediaPipePose
from swimstroke.pose.rtmpose_pose import HALPE26, RTMPose, pick_swimmer

K = len(HALPE26)


def test_no_person_gives_undetected_frame():
    pf = pick_swimmer(np.zeros((0, K, 2)), np.zeros((0, K)), K)
    assert not pf.detected
    assert pf.xy.shape == (K, 2) and np.isnan(pf.xy).all()
    assert np.isnan(pf.vis).all()


def test_keeps_most_confident_person_and_clips_scores():
    kps = np.stack([np.full((K, 2), 1.0), np.full((K, 2), 2.0)])
    scores = np.stack([np.full(K, 0.2), np.full(K, 1.7)])  # RTMPose scores can exceed 1
    pf = pick_swimmer(kps, scores, K)
    assert pf.detected
    np.testing.assert_allclose(pf.xy, 2.0)
    assert (pf.vis == 1.0).all()


def _person(x: float, y: float = 100.0, torso: float = 50.0) -> np.ndarray:
    """A pose whose hips are at (x, y) and shoulders one torso length above."""
    kp = np.full((K, 2), [x, y])
    for name in ("left_shoulder", "right_shoulder"):
        kp[HALPE26.index(name)] = [x, y - torso]
    return kp


def test_follows_same_swimmer_over_more_confident_one():
    kps = np.stack([_person(100), _person(400)])
    scores = np.stack([np.full(K, 0.5), np.full(K, 0.9)])
    # No history: most confident person.
    assert pick_swimmer(kps, scores, K).xy[HALPE26.index("left_hip"), 0] == 400
    # Last pick was near x=110: stay on that swimmer despite lower confidence.
    pf = pick_swimmer(kps, scores, K, last_hip=np.array([110.0, 100.0]))
    assert pf.xy[HALPE26.index("left_hip"), 0] == 100


def test_falls_back_to_most_confident_when_swimmer_lost():
    kps = np.stack([_person(100), _person(400)])
    scores = np.stack([np.full(K, 0.5), np.full(K, 0.9)])
    pf = pick_swimmer(kps, scores, K, last_hip=np.array([250.0, 100.0]))
    assert pf.xy[HALPE26.index("left_hip"), 0] == 400


def test_names_shared_with_mediapipe_for_stroke_detection():
    # strokes.py and the overlay look landmarks up by name, so every joint
    # they use must exist under the same name in both models.
    needed = {f"{s}_{j}" for s in ("left", "right") for j in ("shoulder", "elbow", "wrist", "hip")}
    assert needed <= set(HALPE26)
    assert needed <= set(MediaPipePose.landmark_names)


def test_skeleton_edges_are_valid_indices():
    assert all(0 <= a < K and 0 <= b < K for a, b in RTMPose.connections)

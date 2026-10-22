"""Tests for breaststroke kinematics on a synthetic swimmer (no video needed).

The swimmer faces +x with a torso of L px. Every T seconds the heels come
up to the hips (leg recovery, at t = 0, T, 2T, ...) and the hands are pulled
in 30% of a cycle later. The hips surge forward once per cycle.
"""

import numpy as np
import pandas as pd
import pytest

from swimstroke.kinematics import analyze, gate_pose, leg_reach, swim_direction
from swimstroke.pose.rtmpose_pose import HALPE26

FS, T, DUR, L = 25.0, 2.0, 10.0, 80.0


def bump(phase: np.ndarray) -> np.ndarray:
    """1 at integer phase, falling to 0 within about a quarter cycle."""
    return np.maximum(0.0, np.cos(2 * np.pi * phase)) ** 4


def swimmer(direction: int = 1) -> pd.DataFrame:
    t = np.arange(0, DUR, 1 / FS)
    vel = 1.0 + 0.5 * np.sin(2 * np.pi * t / T)  # TL/s
    hip_x = 100 + direction * L * np.cumsum(vel) / FS
    hip = np.stack([hip_x, np.full_like(t, 300.0)], axis=1)
    fwd = np.array([direction, 0.0])
    leg = -1.8 + 1.3 * bump(t / T)  # heels behind the hips, up at recovery
    arm = 1.2 - 1.5 * bump(t / T - 0.3)  # full reach, dip at pull-in
    pts = {
        "hip": hip,
        "shoulder": hip + L * fwd,
        "elbow": hip + 1.4 * L * fwd,
        "wrist": hip + L * fwd + (arm * L)[:, None] * fwd,
        "knee": hip - 0.9 * L * fwd + [0, 10],
        "ankle": hip + (leg * L)[:, None] * fwd + [0, 15],
        "heel": hip + (leg * L)[:, None] * fwd + [0, 20],
    }
    cols = {"frame": np.arange(len(t)), "t": t, "detected": np.ones(len(t), bool)}
    for name in HALPE26:
        part = name.split("_", 1)[-1]
        xy = pts.get(part, hip)
        cols[f"{name}_x"], cols[f"{name}_y"] = xy[:, 0], xy[:, 1]
        cols[f"{name}_vis"] = np.full(len(t), 0.9)
    return pd.DataFrame(cols)


def test_clean_swimmer_passes_gate():
    _, stats = gate_pose(swimmer())
    assert stats["kept_rate"] == 1.0
    assert stats["typical_torso_px"] == pytest.approx(L)


def test_gate_rejects_stretched_and_teleporting_frames():
    pose = swimmer()
    bad = [60, 61, 62]
    for i in bad[:2]:  # shoulders far away: torso 3x too long (lane rope)
        for s in ("left_shoulder", "right_shoulder"):
            pose.loc[i, f"{s}_x"] += 2 * L
    for s in ("left_hip", "right_hip"):  # hips jump half the image away
        pose.loc[62, f"{s}_x"] += 300
    gated, stats = gate_pose(pose)
    assert not gated.loc[bad, "detected"].any()
    assert gated["detected"].sum() == len(pose) - len(bad)
    assert gated.loc[bad, "left_wrist_x"].isna().all()


def test_gate_rejects_low_confidence_clip():
    pose = swimmer()
    for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        pose[f"{n}_vis"] = 0.2
    _, stats = gate_pose(pose)
    assert stats["kept_rate"] == 0.0


@pytest.mark.parametrize("direction", [1, -1])
def test_cycles_rate_and_arm_leg_timing(direction):
    pose = swimmer(direction)
    assert swim_direction(pose) == direction
    res = analyze(pose)
    s = res["summary"]
    # Leg recoveries at 2, 4, 6, 8 s (the one at t=0 sits on the edge).
    np.testing.assert_allclose(res["leg_events"]["t"], [2, 4, 6, 8], atol=0.05)
    assert s["cycles"] == 3
    assert s["rate_per_min"] == pytest.approx(30, abs=0.5)
    assert s["arm_leg_agreement"] == 1.0
    assert s["arm_pull_pct_median"] == pytest.approx(30, abs=3)


def test_hip_velocity_fluctuation():
    res = analyze(swimmer())
    c = res["cycles"]
    # Sinusoid with amplitude 0.5 TL/s: mean ~1, SD ~0.5/sqrt(2).
    assert c["vel_mean_rel"].mean() == pytest.approx(1.0, abs=0.05)
    assert c["vel_sd"].mean() == pytest.approx(0.5 / np.sqrt(2), abs=0.05)


def test_leg_reach_is_body_relative():
    pose = swimmer()
    r = leg_reach(pose)
    assert np.nanmax(r) == pytest.approx(-0.5, abs=0.02)  # heels up near the hips
    assert np.nanmin(r) == pytest.approx(-1.8, abs=0.02)  # legs extended


def test_fully_rejected_clip_gives_no_cycles():
    core = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    gated, _ = gate_pose(swimmer().assign(**{f"{n}_vis": 0.0 for n in core}))
    assert analyze(gated)["summary"]["cycles"] == 0

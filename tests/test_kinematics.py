"""Tests for breaststroke kinematics on a synthetic swimmer (no video needed).

The swimmer faces +x with a torso of L px. Every T seconds the heels come
up to the hips (leg recovery, at phase 0 of each cycle) and the hands are
pulled in at phase 0.8, as in the real clips. The hips surge forward once
per cycle. With bump() below, a signal is within 10% of its extreme (the
"90% extended" threshold) when |phase - centre| >= 0.155, so the expected
event times are known exactly:
    kick end 15.5%, leg recovery start 84.5%,
    arm extension -4.5% (just before the cycle), pull start 64.5%, pull-in 80%.
"""

import numpy as np
import pandas as pd
import pytest

from swimstroke.kinematics import analyze, gate_pose, joint_angle, leg_reach, swim_direction
from swimstroke.pose.rtmpose_pose import HALPE26

FS, T, DUR, L = 25.0, 2.0, 10.0, 80.0
PULL = 0.8  # phase of the arm pull-in
EXT = 0.155  # distance from a bump's centre where it drops below 10%


def bump(phase: np.ndarray) -> np.ndarray:
    """1 at integer phase, falling to 0 within about a quarter cycle."""
    return np.maximum(0.0, np.cos(2 * np.pi * phase)) ** 4


def swimmer(direction: int = 1, right_knee_scale: float = 1.0) -> pd.DataFrame:
    t = np.arange(0, DUR, 1 / FS)
    vel = 1.0 + 0.5 * np.sin(2 * np.pi * t / T)  # TL/s
    hip_x = 100 + direction * L * np.cumsum(vel) / FS
    hip = np.stack([hip_x, np.full_like(t, 300.0)], axis=1)
    fwd = np.array([direction, 0.0])
    leg = -1.8 + 1.3 * bump(t / T)  # heels behind the hips, up at recovery
    arm = 1.2 - 1.5 * bump(t / T - PULL)  # full reach, dip at pull-in
    # knees bend as the heels come up: knee drops below the hip-heel line
    bend = (bump(t / T) * 0.5 * L)[:, None] * np.array([0, 1.0])
    pts = {
        "hip": hip,
        "shoulder": hip + L * fwd,
        "elbow": hip + 1.4 * L * fwd,
        "wrist": hip + L * fwd + (arm * L)[:, None] * fwd,
        "ankle": hip + (leg * L)[:, None] * fwd + [0, 15],
        "heel": hip + (leg * L)[:, None] * fwd + [0, 20],
    }
    knee = {s: hip + (leg * L / 2)[:, None] * fwd + bend * (right_knee_scale if s == "right" else 1.0)
            for s in ("left", "right")}
    cols = {"frame": np.arange(len(t)), "t": t, "detected": np.ones(len(t), bool)}
    for name in HALPE26:
        side, _, part = name.partition("_")
        xy = knee[side] if part == "knee" else pts.get(part, hip)
        cols[f"{name}_x"], cols[f"{name}_y"] = xy[:, 0], xy[:, 1]
        cols[f"{name}_vis"] = np.full(len(t), 0.9)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------- gate


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


def test_fully_rejected_clip_gives_no_cycles():
    core = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    gated, _ = gate_pose(swimmer().assign(**{f"{n}_vis": 0.0 for n in core}))
    assert analyze(gated)["summary"]["cycles"] == 0


# ---------------------------------------------------------------- cycles and phases


@pytest.mark.parametrize("direction", [1, -1])
def test_cycles_and_rate(direction):
    pose = swimmer(direction)
    assert swim_direction(pose) == direction
    res = analyze(pose)
    s = res["summary"]
    # Leg recoveries at 2, 4, 6, 8 s (the one at t=0 sits on the edge).
    np.testing.assert_allclose(res["leg_events"]["t"], [2, 4, 6, 8], atol=0.05)
    assert s["cycles"] == 3
    assert s["rate_per_min"] == pytest.approx(30, abs=0.5)
    assert s["kicks_per_min"] == pytest.approx(30, abs=0.5)
    assert s["arm_measurable"] == 1.0


def test_event_timing_and_time_gaps():
    s = analyze(swimmer())["summary"]
    # Gaussian smoothing widens each movement slightly, so threshold events
    # land ~2.5% of a cycle late (under 2 video frames at 25 fps).
    tol = 3.5
    assert s["pull_in_pct"] == pytest.approx(100 * PULL, abs=tol)
    kick_end, rec_start = 100 * EXT, 100 * (1 - EXT)
    arm_ext, pull_start = 100 * (PULL + EXT - 1), 100 * (PULL - EXT)
    assert s["leg_kick_pct"] == pytest.approx(kick_end, abs=tol)
    assert s["leg_recovery_pct"] == pytest.approx(100 - rec_start, abs=tol)
    assert s["T1_glide"] == pytest.approx(pull_start - kick_end, abs=2 * tol)  # two events: errors add
    assert s["T2"] == pytest.approx(rec_start - 100 * PULL, abs=2 * tol)
    assert s["T3"] == pytest.approx(kick_end - arm_ext, abs=2 * tol)
    assert s["TTG"] == pytest.approx(s["T1_glide"] + s["T2"] + s["T3"], abs=0.2)
    # arm phases tile the cycle
    assert s["arm_glide_pct"] + s["arm_pull_pct"] + s["arm_recovery_pct"] == pytest.approx(100, abs=0.5)


def test_flat_arm_signal_is_not_measurable():
    pose = swimmer()
    for s in ("left", "right"):  # hands never pulled in: reach constant
        pose[f"{s}_wrist_x"] = pose[f"{s}_shoulder_x"] + 1.2 * L
    res = analyze(pose)
    assert res["summary"]["arm_measurable"] == 0.0
    assert res["summary"]["T1_glide"] is None
    assert len(res["arm_events"]) == 0
    assert res["summary"]["cycles"] == 3  # cycles still come from the legs


def test_wrist_label_swap_spike_is_ignored():
    pose = swimmer()
    for i in (35, 36):  # one wrist jumps behind the hips for 2 frames
        pose.loc[i, "left_wrist_x"] = pose.loc[i, "left_hip_x"] - 2 * L
    assert analyze(pose)["summary"]["pull_in_pct"] == pytest.approx(100 * PULL, abs=2)


# ---------------------------------------------------------------- velocity, angles, asymmetry


def test_hip_velocity_fluctuation():
    c = analyze(swimmer())["cycles"]
    # Sinusoid with amplitude 0.5 TL/s: mean ~1, SD ~0.5/sqrt(2).
    assert c["vel_mean_rel"].mean() == pytest.approx(1.0, abs=0.05)
    assert c["vel_sd"].mean() == pytest.approx(0.5 / np.sqrt(2), abs=0.05)


def test_leg_reach_is_body_relative():
    r = leg_reach(swimmer())
    assert np.nanmax(r) == pytest.approx(-0.5, abs=0.02)  # heels up near the hips
    assert np.nanmin(r) == pytest.approx(-1.8, abs=0.02)  # legs extended


def test_knee_angle_straight_when_extended():
    pose = swimmer()
    ang = joint_angle(pose, "left", "knee")
    assert np.nanmax(ang) > 175  # (nearly) straight leg in the glide
    assert np.nanmin(ang) < 120  # clearly bent at leg recovery


def test_symmetric_swimmer_has_no_asymmetry():
    s = analyze(swimmer())["summary"]
    assert s["knee_rom_si"] == pytest.approx(0, abs=1)
    assert s["heel_up_lr_lag_pct"] == pytest.approx(0, abs=1)
    assert s["asym_cycles"] == 3


def test_weaker_right_knee_shows_positive_asymmetry():
    s = analyze(swimmer(right_knee_scale=0.5))["summary"]
    assert s["knee_rom_si"] > 10  # left bends more than right


def test_events_table_lists_every_event_in_time_order():
    ev = analyze(swimmer())["events"]
    assert list(ev.columns) == ["cycle", "event", "t"]
    assert ev["t"].is_monotonic_increasing
    per_cycle = ev.groupby("cycle")["event"].apply(set)
    expected = {"leg_recovery_end", "kick_end", "leg_recovery_start", "arm_extension", "pull_start", "pull_in"}
    assert all(s == expected for s in per_cycle)
    # cycle 0 starts at the leg recovery at t = 2 s; its pull-in is at 2 + 0.8 * T
    c0 = ev[ev["cycle"] == 0].set_index("event")["t"]
    assert c0["leg_recovery_end"] == pytest.approx(2.0, abs=0.05)
    assert c0["pull_in"] == pytest.approx(2.0 + PULL * T, abs=0.07)

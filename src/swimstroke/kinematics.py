"""Underwater side-view kinematics: clean poses, breaststroke cycles, hip velocity.

Works on the pose table (any backend), but is written for RTMPose's
HALPE-26 output, which includes heels for the leg signal.

1. Gate: RTMPose always returns *some* person, even when it only finds the
   lane rope. A frame is kept only if the body is confidently seen and has
   human proportions, and the hips don't teleport between frames.
2. Cycles: one breaststroke cycle runs from one end of leg recovery (heels
   drawn up to the hips, just before the kick) to the next. That moment is a
   sharp peak in the heel position along the body axis, so it is easier to
   time than the glide, where the arms sit at full reach for a while.
   The arms give an independent check: each cycle should hold exactly one
   arm "pull-in" (hands drawn back under the chest, a sharp dip in the same
   body-axis reach as freestyle, strokes.py). Its position within the cycle
   is a first arm-leg timing measure. Both signals are measured along the
   body axis, so they don't depend on swim direction or camera distance.
3. Hip velocity along the swim direction, in torso lengths per second
   (TL/s). Units are body-relative because the clips have no metre scale.
   If the camera moves with the swimmer (trolley), this is velocity relative
   to the camera: the shape within a cycle is real, the mean is not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from swimstroke.strokes import near_arm_reach, resample, segments, smooth

CORE = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


@dataclass(frozen=True)
class GateParams:
    min_core_score: float = 0.35  # mean confidence of shoulders + hips
    torso_range: tuple[float, float] = (0.6, 1.6)  # x clip's typical torso length
    max_thigh: float = 1.6  # hip-knee, in torso lengths (real: ~0.9)
    max_upper_arm: float = 1.2  # shoulder-elbow, in torso lengths (real: ~0.5)
    max_jump: float = 0.4  # hip movement between frames, in torso lengths
    min_segment_s: float = 0.5  # drop kept stretches shorter than this


@dataclass(frozen=True)
class CycleParams:
    fs: float = 50.0  # Hz, uniform resampling rate
    max_gap_s: float = 0.25
    smooth_sigma_s: float = 0.08
    min_segment_s: float = 1.0
    min_cycle_s: float = 0.9  # fastest plausible breaststroke cycle (~65/min)
    max_cycle_s: float = 4.0  # longer = a missed glide, not one cycle
    arm_prominence: float = 0.4  # torso lengths (0.6 missed shallow pulls)
    leg_prominence: float = 0.4


def _xy(pose: pd.DataFrame, name: str) -> np.ndarray:
    return pose[[f"{name}_x", f"{name}_y"]].to_numpy(dtype=float)


def _mid(pose: pd.DataFrame, a: str, b: str) -> np.ndarray:
    return (_xy(pose, a) + _xy(pose, b)) / 2


def torso_length(pose: pd.DataFrame) -> np.ndarray:
    return np.linalg.norm(_mid(pose, "left_shoulder", "right_shoulder") - _mid(pose, "left_hip", "right_hip"), axis=1)


def _seg_len(pose: pd.DataFrame, a: str, b: str) -> np.ndarray:
    """Longer of the left/right segment (a bad pose stretches at least one)."""
    return np.fmax(
        np.linalg.norm(_xy(pose, f"left_{a}") - _xy(pose, f"left_{b}"), axis=1),
        np.linalg.norm(_xy(pose, f"right_{a}") - _xy(pose, f"right_{b}"), axis=1),
    )


def gate_pose(pose: pd.DataFrame, params: GateParams = GateParams()) -> tuple[pd.DataFrame, dict]:
    """Copy of the pose table with implausible frames marked undetected (NaN).

    Returns (gated pose, stats). The typical torso length is the median over
    frames whose core joints are confidently seen, so a clip that is mostly
    false detections can't define "typical" by itself.
    """
    core = pose[[f"{n}_vis" for n in CORE]].mean(axis=1).to_numpy()
    L = torso_length(pose)
    confident = pose["detected"].to_numpy() & (core >= params.min_core_score)
    ref = float(np.nanmedian(L[confident])) if confident.any() else np.nan

    ok = confident.copy()
    with np.errstate(invalid="ignore"):
        ok &= (L >= params.torso_range[0] * ref) & (L <= params.torso_range[1] * ref)
        ok &= _seg_len(pose, "hip", "knee") <= params.max_thigh * L
        ok &= _seg_len(pose, "shoulder", "elbow") <= params.max_upper_arm * L

    # Teleport check against the last kept frame (not the previous frame,
    # so one bad frame doesn't also reject the good frame after it).
    hip = _mid(pose, "left_hip", "right_hip")
    last = None
    for i in np.flatnonzero(ok):
        if last is not None and i - last <= 3:
            if np.linalg.norm(hip[i] - hip[last]) > params.max_jump * ref * (i - last):
                ok[i] = False
                continue
        last = i

    # Drop short isolated stretches: a real swimmer is tracked continuously.
    t = pose["t"].to_numpy(dtype=float)
    for a, b in segments(ok):
        if t[b - 1] - t[a] < params.min_segment_s:
            ok[a:b] = False

    out = pose.copy()
    lm_cols = [c for c in pose.columns if c.endswith(("_x", "_y", "_vis"))]
    out.loc[~ok, lm_cols] = np.nan
    out["detected"] = ok
    out.attrs = dict(pose.attrs)
    stats = {
        "frames": len(pose),
        "kept_rate": round(float(ok.mean()), 3),
        "typical_torso_px": round(ref, 1) if np.isfinite(ref) else None,
    }
    return out, stats


def body_axis(pose: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hip_mid, unit hip->shoulder axis, torso length) per frame."""
    hip = _mid(pose, "left_hip", "right_hip")
    axis = _mid(pose, "left_shoulder", "right_shoulder") - hip
    L = np.linalg.norm(axis, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return hip, axis / L[:, None], L


def leg_reach(pose: pd.DataFrame) -> np.ndarray:
    """Heel position along the body axis, in torso lengths (negative = behind
    the hips). Highest when the heels are drawn up to the hips at the end of
    the leg recovery. Visibility-weighted over both legs; ankles if no heels."""
    hip, unit, L = body_axis(pose)
    part = "heel" if "left_heel_x" in pose else "ankle"
    vals, weights = [], []
    for side in ("left", "right"):
        rel = _xy(pose, f"{side}_{part}") - hip
        vals.append((rel * unit).sum(axis=1) / L)
        weights.append(pose[f"{side}_{part}_vis"].to_numpy(dtype=float))
    v, w = np.array(vals), np.array(weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(v * w, axis=0) / np.nansum(np.where(np.isnan(v), 0, w), axis=0)


def swim_direction(pose: pd.DataFrame) -> int:
    """+1 if the swimmer heads towards +x (image right), -1 if left."""
    _, unit, _ = body_axis(pose)
    return 1 if np.nanmedian(unit[:, 0]) >= 0 else -1


def _signal(pose: pd.DataFrame, y: np.ndarray, p: CycleParams) -> tuple[np.ndarray, np.ndarray]:
    grid, r = resample(pose["t"].to_numpy(dtype=float), y, p.fs, p.max_gap_s)
    return grid, smooth(r, p.fs, p.smooth_sigma_s, p.min_segment_s)


def _peaks(t: np.ndarray, y: np.ndarray, prominence: float, p: CycleParams) -> pd.DataFrame:
    rows = []
    for seg, (a, b) in enumerate(segments(~np.isnan(y))):
        idx, _ = find_peaks(y[a:b], prominence=prominence, distance=max(1, int(p.min_cycle_s * p.fs)))
        rows += [{"t": round(float(t[a + i]), 3), "segment": seg, "value": float(y[a + i])} for i in idx]
    return pd.DataFrame(rows, columns=["t", "segment", "value"])


def hip_velocity(pose: pd.DataFrame, p: CycleParams = CycleParams()) -> tuple[np.ndarray, np.ndarray]:
    """(t, hip velocity along the swim direction in TL/s) on the uniform grid."""
    hip, _, L = body_axis(pose)
    ref = float(np.nanmedian(L))
    t, x = _signal(pose, hip[:, 0] * swim_direction(pose) / ref, p)
    return t, np.gradient(x, 1.0 / p.fs)


def analyze(pose: pd.DataFrame, p: CycleParams = CycleParams()) -> dict:
    """Breaststroke cycles + per-cycle hip velocity from a gated pose table.

    Returns {"cycles": per-cycle table, "series": time series, "summary": dict,
    "leg_events": leg-recovery peaks, "arm_events": arm pull-in dips}.
    """
    if not pose["detected"].any():  # gate rejected the whole clip
        empty = pd.DataFrame(columns=["t", "segment", "value"])
        return {"cycles": pd.DataFrame(), "series": pd.DataFrame(), "leg_events": empty,
                "arm_events": empty, "summary": {"cycles": 0, "params": asdict(p)}}
    t, arm = _signal(pose, near_arm_reach(pose), p)
    _, leg = _signal(pose, leg_reach(pose), p)
    _, vel = hip_velocity(pose, p)
    leg_ev = _peaks(t, leg, p.leg_prominence, p)
    arm_ev = _peaks(t, -arm, p.arm_prominence, p)  # dips in reach = pull-in
    arm_ev["value"] = -arm_ev["value"]

    rows = []
    for (_, a), (_, b) in zip(leg_ev.iloc[:-1].iterrows(), leg_ev.iloc[1:].iterrows()):
        dur = b["t"] - a["t"]
        if a["segment"] != b["segment"] or dur > p.max_cycle_s:
            continue  # across a tracking gap: may hide a missed cycle
        v = vel[(t >= a["t"]) & (t < b["t"])]
        pulls = arm_ev[(arm_ev["t"] >= a["t"]) & (arm_ev["t"] < b["t"])]
        rows.append({
            "start_s": a["t"],
            "end_s": b["t"],
            "duration_s": round(dur, 3),
            "rate_per_min": round(60 / dur, 1),
            "arm_pulls": len(pulls),
            # when the hands are pulled in, as % of the cycle after leg recovery
            "arm_pull_pct": round(100 * (pulls["t"].iloc[0] - a["t"]) / dur, 1) if len(pulls) == 1 else None,
            "vel_mean_rel": round(float(np.nanmean(v)), 3),
            "vel_min_rel": round(float(np.nanmin(v)), 3),
            "vel_max_rel": round(float(np.nanmax(v)), 3),
            "vel_sd": round(float(np.nanstd(v)), 3),
        })
    cycles = pd.DataFrame(rows)
    summary = {
        "leg_events": int(len(leg_ev)),
        "arm_events": int(len(arm_ev)),
        "cycles": int(len(cycles)),
        "rate_per_min": round(60 / float(cycles["duration_s"].median()), 1) if len(cycles) else None,
        # share of cycles with exactly one arm pull-in: arms and legs agree
        "arm_leg_agreement": round(float(cycles["arm_pulls"].eq(1).mean()), 3) if len(cycles) else None,
        "arm_pull_pct_median": float(cycles["arm_pull_pct"].median()) if len(cycles) else None,
        "vel_sd_median_tl_s": round(float(cycles["vel_sd"].median()), 3) if len(cycles) else None,
        "params": asdict(p),
    }
    series = pd.DataFrame({"t": t, "arm_reach": arm, "leg_reach": leg, "hip_vel_rel": vel})
    return {"cycles": cycles, "series": series, "summary": summary, "leg_events": leg_ev, "arm_events": arm_ev}

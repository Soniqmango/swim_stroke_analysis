"""Underwater side-view breaststroke kinematics.

Works on the pose table (any backend), but is written for RTMPose's
HALPE-26 output, which includes heels for the leg signal.

1. Gate: RTMPose always returns *some* person, even when it only finds the
   lane rope. A frame is kept only if the body is confidently seen and has
   human proportions, and the hips don't teleport between frames.
2. Cycles: one cycle runs from one end of leg recovery (heels drawn up to
   the hips, just before the kick) to the next. That moment is a sharp peak
   in the heel position along the body axis.
3. Events inside each cycle, from two body-axis signals (independent of swim
   direction and camera distance):
     leg reach = heels relative to the hips, in torso lengths (TL)
     arm reach = wrists relative to the shoulders, in TL
   Legs:  kick end    = legs 90% extended backwards
          leg recovery start = legs leave full extension (last time >90%)
   Arms:  arm extension = arms 90% extended forwards (end of arm recovery)
          pull start    = arms leave full extension (end of glide)
          pull-in       = hands drawn back the furthest (end of insweep,
                          start of arm recovery): the minimum of arm reach
   A cycle's arm events are only measured if arm reach swings at least
   `min_arm_depth` in that cycle; a flat signal (hands near the surface,
   wrist labels swapping) gives "not measurable" instead of a wrong time.
4. Phases (% of cycle) and arm-leg time gaps, following the time-gap
   approach of Chollet et al. for breaststroke:
     T1 = pull start - kick end        (glide between leg and arm propulsion)
     T2 = leg recovery start - pull-in (arm recovery to leg recovery)
     T3 = kick end - arm extension     (arms vs legs reaching full extension)
     TTG = T1 + T2 + T3, all in % of cycle duration.
   The exact event definitions are written above so they can be checked
   against the definitions the lab uses.
5. Joint angles (knee, hip, elbow, shoulder, per side), kick frequency, and
   left/right asymmetry. From a side view the far limbs are often hidden, so
   asymmetry is only computed where both sides are confidently seen.
6. Hip velocity along the swim direction, in TL/s. If the camera moves with
   the swimmer (trolley), this is relative to the camera: the shape within a
   cycle is real, the mean swimming speed is not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from swimstroke.strokes import resample, segments, smooth

CORE = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
SIDES = ("left", "right")
# angle name -> (point a, vertex, point c); 180 deg = straight
ANGLES = {
    "knee": ("hip", "knee", "ankle"),
    "hip": ("shoulder", "hip", "knee"),
    "elbow": ("shoulder", "elbow", "wrist"),
    "shoulder": ("hip", "shoulder", "elbow"),
}


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
    despike_s: float = 0.2  # median filter: removes one-frame left/right label swaps
    smooth_sigma_s: float = 0.08
    min_segment_s: float = 1.0
    min_cycle_s: float = 0.9  # fastest plausible breaststroke cycle (~65/min)
    max_cycle_s: float = 4.0  # longer = a missed recovery, not one cycle
    leg_prominence: float = 0.4  # torso lengths
    min_arm_depth: float = 1.0  # arm reach swing needed to time arm events (TL)
    extension_frac: float = 0.8  # "extended" = 80% of the cycle's range (90% sits inside video noise)
    min_vis: float = 0.3  # both-sides visibility needed for asymmetry


# ---------------------------------------------------------------- geometry


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


def body_axis(pose: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hip_mid, unit hip->shoulder axis, torso length) per frame."""
    hip = _mid(pose, "left_hip", "right_hip")
    axis = _mid(pose, "left_shoulder", "right_shoulder") - hip
    L = np.linalg.norm(axis, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return hip, axis / L[:, None], L


def joint_angle(pose: pd.DataFrame, side: str, joint: str) -> np.ndarray:
    """Inner angle at a joint in degrees (180 = straight), in the image plane."""
    a, b, c = (f"{side}_{n}" for n in ANGLES[joint])
    u, v = _xy(pose, a) - _xy(pose, b), _xy(pose, c) - _xy(pose, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (u * v).sum(axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def swim_direction(pose: pd.DataFrame) -> int:
    """+1 if the swimmer heads towards +x (image right), -1 if left."""
    _, unit, _ = body_axis(pose)
    return 1 if np.nanmedian(unit[:, 0]) >= 0 else -1


# ---------------------------------------------------------------- gate


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


# ---------------------------------------------------------------- signals


def _despike(y: np.ndarray, size: int) -> np.ndarray:
    """Median filter each continuous stretch: one-frame left/right swaps of a
    wrist or heel become spikes, which a median removes without shifting
    the real movement in time."""
    out = y.copy()
    if size < 3:
        return out
    for a, b in segments(~np.isnan(y)):
        out[a:b] = median_filter(y[a:b], size=size, mode="nearest")
    return out


def _reach(pose: pd.DataFrame, part: str, origin: str, despike: int = 0) -> dict[str, np.ndarray]:
    """Per side: position of `part` relative to `origin` along the body axis, in TL."""
    _, unit, L = body_axis(pose)
    o = _mid(pose, f"left_{origin}", f"right_{origin}")
    return {s: _despike(((_xy(pose, f"{s}_{part}") - o) * unit).sum(axis=1) / L, despike) for s in SIDES}


def _weighted(pose: pd.DataFrame, per_side: dict[str, np.ndarray], part: str) -> np.ndarray:
    """Visibility-weighted mean of the two sides (follows the side the camera sees)."""
    v = np.array([per_side[s] for s in SIDES])
    w = np.array([pose[f"{s}_{part}_vis"].to_numpy(dtype=float) for s in SIDES])
    w = np.where(np.isnan(v), 0.0, np.nan_to_num(w))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(v * w, axis=0) / w.sum(axis=0)


def leg_reach(pose: pd.DataFrame, despike: int = 0) -> np.ndarray:
    """Heels relative to the hips along the body axis (negative = behind).
    Highest when the heels are drawn up at the end of the leg recovery."""
    part = "heel" if "left_heel_x" in pose else "ankle"
    return _weighted(pose, _reach(pose, part, "hip", despike), part)


def arm_reach(pose: pd.DataFrame, despike: int = 0) -> np.ndarray:
    """Wrists relative to the shoulders along the body axis (positive = ahead).
    Highest in the glide, lowest when the hands are pulled in."""
    return _weighted(pose, _reach(pose, "wrist", "shoulder", despike), "wrist")


def _signal(pose: pd.DataFrame, y: np.ndarray, p: CycleParams) -> tuple[np.ndarray, np.ndarray]:
    grid, r = resample(pose["t"].to_numpy(dtype=float), y, p.fs, p.max_gap_s)
    return grid, smooth(r, p.fs, p.smooth_sigma_s, p.min_segment_s)


def _despike_frames(pose: pd.DataFrame, p: CycleParams) -> int:
    dt = float(np.median(np.diff(pose["t"]))) if len(pose) > 1 else 0.04
    return max(1, int(round(p.despike_s / dt))) | 1  # odd window


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


# ---------------------------------------------------------------- events


def _first(t: np.ndarray, mask: np.ndarray) -> float | None:
    i = np.flatnonzero(mask)
    return float(t[i[0]]) if len(i) else None


def _last(t: np.ndarray, mask: np.ndarray) -> float | None:
    i = np.flatnonzero(mask)
    return float(t[i[-1]]) if len(i) else None


def _robust_range(y: np.ndarray, pct: float = 5.0) -> tuple[float, float]:
    """(low, high) of a signal ignoring the most extreme few % of samples,
    so one outlier frame can't set the "fully extended" level."""
    return float(np.nanpercentile(y, pct)), float(np.nanpercentile(y, 100 - pct))


def leg_events(t: np.ndarray, leg: np.ndarray, a: float, b: float, frac: float) -> dict:
    """Kick end and leg recovery start inside one cycle [a, b)."""
    m = (t >= a) & (t < b)
    y, tt = leg[m], t[m]
    if len(y) == 0 or np.isnan(y).all():
        return {"kick_end": None, "leg_recovery_start": None}
    lo, hi = _robust_range(y)
    extended = y <= hi - frac * (hi - lo)
    return {"kick_end": _first(tt, extended), "leg_recovery_start": _last(tt, extended)}


def arm_events(t: np.ndarray, arm: np.ndarray, a: float, b: float, frac: float, min_depth: float) -> dict:
    """Arm extension, pull start and pull-in for the cycle [a, b), or Nones
    if the arm signal is too flat to time reliably.

    Pull-in and pull start lie inside the cycle. The arms often finish
    extending just before the cycle starts, so the extension is searched from
    the previous pull-in (the lowest arm reach in the half cycle before `a`):
    the first time after it that the arms reach the extended level. It can
    be a negative % of the cycle."""
    none = {"arm_extension": None, "pull_start": None, "pull_in": None, "arm_depth": None}
    m = (t >= a) & (t < b)
    y, tt = arm[m], t[m]
    if len(y) == 0 or np.isnan(y).mean() > 0.3:
        return none
    i_in = int(np.nanargmin(y))
    lo, hi = _robust_range(y)
    lo = min(lo, float(y[i_in]))
    depth = round(hi - lo, 3)
    if depth < min_depth:
        return {**none, "arm_depth": depth}
    thr = lo + frac * (hi - lo)
    pull_start = _last(tt, (y >= thr) & (np.arange(len(y)) < i_in))
    if pull_start is None:
        return {**none, "pull_in": float(tt[i_in]), "arm_depth": depth}
    w = (t >= a - 0.5 * (b - a)) & (t <= pull_start)
    yw, tw = arm[w], t[w]
    before = tw < a
    if not before.any() or np.isnan(yw[before]).mean() > 0.5:
        # previous pull-in not tracked (e.g. first cycle of a stretch): don't guess
        return {"arm_extension": None, "pull_start": pull_start, "pull_in": float(tt[i_in]), "arm_depth": depth}
    start = int(np.nanargmin(np.where(before, yw, np.inf)))
    ext = _first(tw[start:], yw[start:] >= thr)
    return {"arm_extension": ext, "pull_start": pull_start, "pull_in": float(tt[i_in]), "arm_depth": depth}


def _pct(x: float | None, a: float, dur: float) -> float | None:
    return None if x is None else round(100 * (x - a) / dur, 1)


def _gap(x: float | None, y: float | None, dur: float) -> float | None:
    return None if x is None or y is None else round(100 * (x - y) / dur, 1)


def _range_in(t: np.ndarray, y: np.ndarray, a: float, b: float) -> tuple[float, float]:
    m = (t >= a) & (t < b)
    if not m.any() or np.isnan(y[m]).all():
        return np.nan, np.nan
    return float(np.nanmin(y[m])), float(np.nanmax(y[m]))


def _symmetry_index(left: float | None, right: float | None) -> float | None:
    """100 * (L - R) / mean(L, R); 0 = symmetric."""
    if left is None or right is None or not (np.isfinite(left) and np.isfinite(right)) or left + right == 0:
        return None
    return round(100 * (left - right) / ((left + right) / 2), 1)


def _side_peak(t: np.ndarray, y: np.ndarray, a: float, b: float, margin: float = 0.25) -> float | None:
    """Time of one side's heel-up near the cycle start (the peak within a
    window around `a`), or None if that side isn't seen there."""
    dur = b - a
    m = (t >= a - margin * dur) & (t <= a + margin * dur)
    if not m.any() or np.isnan(y[m]).mean() > 0.3:
        return None
    return float(t[m][np.nanargmax(y[m])])


# ---------------------------------------------------------------- analysis


def _side_series(pose: pd.DataFrame, p: CycleParams, k: int) -> dict[str, np.ndarray]:
    """Per-side joint angles and reaches on the uniform grid, NaN where that
    side isn't confidently seen (the far side is often hidden in side view)."""
    out: dict[str, np.ndarray] = {}
    foot = "heel" if "left_heel_x" in pose else "ankle"
    side_leg = _reach(pose, foot, "hip", k)
    side_arm = _reach(pose, "wrist", "shoulder", k)
    for s in SIDES:
        for joint, (_, vertex, _) in ANGLES.items():
            seen = pose[f"{s}_{vertex}_vis"].to_numpy(dtype=float) >= p.min_vis
            ang = np.where(seen, joint_angle(pose, s, joint), np.nan)
            out[f"{s}_{joint}_angle"] = _signal(pose, _despike(ang, k), p)[1]
        heel_seen = pose[f"{s}_{foot}_vis"].to_numpy(dtype=float) >= p.min_vis
        out[f"{s}_leg_reach"] = _signal(pose, np.where(heel_seen, side_leg[s], np.nan), p)[1]
        wrist_seen = pose[f"{s}_wrist_vis"].to_numpy(dtype=float) >= p.min_vis
        out[f"{s}_arm_reach"] = _signal(pose, np.where(wrist_seen, side_arm[s], np.nan), p)[1]
    return out


def _cycle_row(t: np.ndarray, series: dict, a: float, b: float, p: CycleParams) -> tuple[dict, dict, dict]:
    dur = b - a
    le = leg_events(t, series["leg_reach"], a, b, p.extension_frac)
    ae = arm_events(t, series["arm_reach"], a, b, p.extension_frac, p.min_arm_depth)
    v = series["hip_vel_rel"][(t >= a) & (t < b)]
    row = {
        "start_s": a, "end_s": b, "duration_s": round(dur, 3), "rate_per_min": round(60 / dur, 1),
        "kick_end_pct": _pct(le["kick_end"], a, dur),
        "leg_recovery_start_pct": _pct(le["leg_recovery_start"], a, dur),
        "arm_extension_pct": _pct(ae["arm_extension"], a, dur),
        "pull_start_pct": _pct(ae["pull_start"], a, dur),
        "pull_in_pct": _pct(ae["pull_in"], a, dur),
        "arm_depth_tl": ae["arm_depth"],
        # arm-leg time gaps, % of cycle
        "T1_glide": _gap(ae["pull_start"], le["kick_end"], dur),
        "T2": _gap(le["leg_recovery_start"], ae["pull_in"], dur),
        "T3": _gap(le["kick_end"], ae["arm_extension"], dur),
        "TTG": None,
        "vel_mean_rel": round(float(np.nanmean(v)), 3),
        "vel_min_rel": round(float(np.nanmin(v)), 3),
        "vel_max_rel": round(float(np.nanmax(v)), 3),
        "vel_sd": round(float(np.nanstd(v)), 3),
    }
    if None not in (row["T1_glide"], row["T2"], row["T3"]):
        row["TTG"] = round(row["T1_glide"] + row["T2"] + row["T3"], 1)
    # leg phases: kick (start -> kick end), glide (extended), recovery
    if None not in (row["kick_end_pct"], row["leg_recovery_start_pct"]):
        row["leg_kick_pct"] = row["kick_end_pct"]
        row["leg_glide_pct"] = round(row["leg_recovery_start_pct"] - row["kick_end_pct"], 1)
        row["leg_recovery_pct"] = round(100 - row["leg_recovery_start_pct"], 1)
    # arm phases: glide, pull (propulsion), recovery (wraps over the cycle boundary)
    if None not in (row["arm_extension_pct"], row["pull_start_pct"], row["pull_in_pct"]):
        row["arm_glide_pct"] = round(row["pull_start_pct"] - row["arm_extension_pct"], 1)
        row["arm_pull_pct"] = round(row["pull_in_pct"] - row["pull_start_pct"], 1)
        row["arm_recovery_pct"] = round(100 - row["arm_glide_pct"] - row["arm_pull_pct"], 1)
    # joint ranges per side and left/right asymmetry
    for s in SIDES:
        for joint in ("knee", "hip", "elbow"):
            lo, hi = _range_in(t, series[f"{s}_{joint}_angle"], a, b)
            row[f"{s}_{joint}_min_deg"] = round(lo, 1) if np.isfinite(lo) else None
            row[f"{s}_{joint}_rom_deg"] = round(hi - lo, 1) if np.isfinite(lo) else None
    for joint in ("knee", "elbow"):
        row[f"{joint}_rom_si"] = _symmetry_index(row[f"left_{joint}_rom_deg"], row[f"right_{joint}_rom_deg"])
    lt, rt = (_side_peak(t, series[f"{s}_leg_reach"], a, b) for s in SIDES)
    row["heel_up_lr_lag_pct"] = _gap(lt, rt, dur)  # left heel-up minus right
    return row, le, ae


def analyze(pose: pd.DataFrame, p: CycleParams = CycleParams()) -> dict:
    """Breaststroke cycles, phases, arm-leg time gaps, joint angles, kick
    frequency, asymmetry and hip velocity from a gated pose table.

    Returns {"cycles": per-cycle table, "series": time series, "summary": dict,
    "leg_events": leg-recovery peaks, "arm_events": pull-ins,
    "events": every timed event (cycle, event, t), for overlays and validation}.
    """
    empty_ev = pd.DataFrame(columns=["t", "segment", "value"])
    if not pose["detected"].any():  # gate rejected the whole clip
        return {"cycles": pd.DataFrame(), "series": pd.DataFrame(), "leg_events": empty_ev,
                "arm_events": empty_ev, "events": pd.DataFrame(columns=["cycle", "event", "t"]),
                "summary": {"cycles": 0, "params": asdict(p)}}

    k = _despike_frames(pose, p)
    t, arm = _signal(pose, arm_reach(pose, k), p)
    _, leg = _signal(pose, leg_reach(pose, k), p)
    _, vel = hip_velocity(pose, p)
    series = {"t": t, "arm_reach": arm, "leg_reach": leg, "hip_vel_rel": vel, **_side_series(pose, p, k)}

    leg_ev = _peaks(t, leg, p.leg_prominence, p)
    rows, pull_ins, events = [], [], []
    for (_, A), (_, B) in zip(leg_ev.iloc[:-1].iterrows(), leg_ev.iloc[1:].iterrows()):
        if A["segment"] != B["segment"] or B["t"] - A["t"] > p.max_cycle_s:
            continue  # across a tracking gap: may hide a missed cycle
        row, le, ae = _cycle_row(t, series, A["t"], B["t"], p)
        n = len(rows)
        rows.append({"cycle": n, **row})
        timed = {"leg_recovery_end": A["t"], **le, **{k: ae[k] for k in ("arm_extension", "pull_start", "pull_in")}}
        events += [{"cycle": n, "event": e, "t": round(x, 3)} for e, x in timed.items() if x is not None]
        if ae["pull_in"] is not None:
            pull_ins.append({"t": round(ae["pull_in"], 3), "segment": A["segment"],
                             "value": float(np.interp(ae["pull_in"], t, arm))})

    cycles = pd.DataFrame(rows)
    ev = pd.DataFrame(events, columns=["cycle", "event", "t"]).sort_values("t", kind="stable").reset_index(drop=True)
    return {"cycles": cycles, "series": pd.DataFrame(series), "summary": _summarize(cycles, leg_ev, leg, p),
            "leg_events": leg_ev, "arm_events": pd.DataFrame(pull_ins, columns=["t", "segment", "value"]),
            "events": ev}


def kick_intervals(leg_ev: pd.DataFrame, max_s: float) -> np.ndarray:
    """Time between consecutive kicks in the same tracked stretch (a gap may
    hide a missed kick, so intervals across one don't count)."""
    t, seg = leg_ev["t"].to_numpy(dtype=float), leg_ev["segment"].to_numpy()
    d = np.diff(t)
    return d[(seg[1:] == seg[:-1]) & (d <= max_s)]


def _summarize(cycles: pd.DataFrame, leg_ev: pd.DataFrame, leg: np.ndarray, p: CycleParams) -> dict:
    def med(*cols: str) -> float | None:
        parts = [cycles[c] for c in cols if c in cycles]
        vals = pd.concat(parts, ignore_index=True).dropna() if parts else pd.Series(dtype=float)
        return round(float(vals.median()), 1) if len(vals) else None

    n = len(cycles)
    kicks = kick_intervals(leg_ev, p.max_cycle_s)
    return {
        "leg_events": int(len(leg_ev)),
        "cycles": n,
        "rate_per_min": round(60 / float(cycles["duration_s"].median()), 1) if n else None,
        # from leg events alone, so it doesn't depend on the arms being measurable
        "kicks_per_min": round(60 / float(np.median(kicks)), 1) if len(kicks) else None,
        "arm_measurable": round(float(cycles["pull_in_pct"].notna().mean()), 3) if n else None,
        "pull_in_pct": med("pull_in_pct"),
        "T1_glide": med("T1_glide"), "T2": med("T2"), "T3": med("T3"), "TTG": med("TTG"),
        "leg_kick_pct": med("leg_kick_pct"), "leg_glide_pct": med("leg_glide_pct"),
        "leg_recovery_pct": med("leg_recovery_pct"),
        "arm_glide_pct": med("arm_glide_pct"), "arm_pull_pct": med("arm_pull_pct"),
        "arm_recovery_pct": med("arm_recovery_pct"),
        "knee_min_deg": med("left_knee_min_deg", "right_knee_min_deg"),
        "knee_rom_si": med("knee_rom_si"), "elbow_rom_si": med("elbow_rom_si"),
        "heel_up_lr_lag_pct": med("heel_up_lr_lag_pct"),
        "asym_cycles": int(cycles["knee_rom_si"].notna().sum()) if n else 0,
        "vel_sd_median_tl_s": round(float(cycles["vel_sd"].median()), 3) if n else None,
        "params": asdict(p),
    }

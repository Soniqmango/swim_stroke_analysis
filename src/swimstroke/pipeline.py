"""End-to-end analysis of clips filmed from one camera setup.

Per clip (each step cached under outputs/<clip>/):
  1. stabilise onto the setup's reference image          (stabilize.py)
  2. motion blobs -> swimmer track                       (track.py)
  3. pose -> hand entries, only where the pose is on
     the tracked swimmer (rejects false detections)      (pose/, strokes.py)
Per session (all clips together):
  4. lane + virtual walls from every turn                (track.py, laps.py)
  5. push-off, turns, finish -> length times             (laps.py)
  6. strokes per length, stroke rate                     (strokes.py)
  7. if pool geometry is known: metres -> DPS, 5 m turn times (pool.py)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from swimstroke.laps import (
    detect_pushoff,
    length_times,
    segments,
    turn_vertices,
    wall_events,
    wall_positions,
)
from swimstroke.pool import PoolGeometry, distance_per_stroke, turn_times
from swimstroke.stabilize import CameraMotion, estimate_motion, to_reference
from swimstroke.strokes import detect_cycles, strokes_in_length, summarize_strokes
from swimstroke.track import find_candidates, track_swimmer, wall_activity
from swimstroke.video import iter_frames

SETUPS = Path("data/calib/setups")


@dataclass
class Setup:
    """One camera position. Learned values are saved back after each run."""

    name: str
    ref_video: str  # all clips are stabilised onto this video's ref_frame
    ref_frame: int = 0
    lane: list[float] = field(default_factory=list)  # y = lane[0] * x + lane[1], ref px
    walls_px: dict[str, float] = field(default_factory=dict)  # virtual wall x per side
    pool_file: str | None = None  # PoolGeometry JSON, enables metres

    @property
    def path(self) -> Path:
        return SETUPS / f"{self.name}.json"

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), indent=2))
        return self.path

    @classmethod
    def load_or_create(cls, name: str, ref_video: str | Path) -> "Setup":
        p = SETUPS / f"{name}.json"
        if p.exists():
            return cls(**json.loads(p.read_text()))
        return cls(name=name, ref_video=str(ref_video))

    def ref_image(self) -> np.ndarray:
        for f in iter_frames(self.ref_video, max_frames=self.ref_frame + 1):
            if f.idx == self.ref_frame:
                return f.image
        raise ValueError(f"ref_frame {self.ref_frame} not in {self.ref_video}")


@dataclass
class ClipData:
    name: str
    video: Path
    motion: CameraMotion
    track: pd.DataFrame  # frame, t, x, y (ref px)
    pose: pd.DataFrame | None = None  # gated pose table
    cycles: pd.DataFrame | None = None  # near-arm entries
    pose_on_swimmer: float = float("nan")  # share of frames with a pose on the swimmer


# ---------- per clip ----------


def track_clip(video: Path, setup: Setup, ref_image: np.ndarray, force: bool, log=print) -> ClipData:
    out = Path("outputs") / video.stem
    f_motion, f_blobs, f_track = out / "camera_motion_setup.npz", out / "blobs_setup.parquet", out / "track.parquet"
    if f_motion.exists() and not force:
        motion = CameraMotion.load(f_motion)
    else:
        log(f"  {video.stem}: stabilising onto {Path(setup.ref_video).stem}")
        motion = estimate_motion(video, ref_image=ref_image)
        motion.save(f_motion)
    if f_track.exists() and not force:
        track = pd.read_parquet(f_track)
    else:
        if f_blobs.exists() and not force:
            cands = pd.read_parquet(f_blobs)
        else:
            log(f"  {video.stem}: finding moving blobs")
            cands = find_candidates(video, motion)
            cands.to_parquet(f_blobs)
        track, _ = track_swimmer(cands, len(motion.t))
        track["t"] = motion.t
        track.to_parquet(f_track)
    return ClipData(video.stem, video, motion, track)


def gate_pose(pose: pd.DataFrame, motion: CameraMotion, track: pd.DataFrame, max_dist_px: float = 200.0) -> tuple[pd.DataFrame, float]:
    """Keep pose detections only where the body is on the tracked swimmer.

    The pose model happily finds 'people' in umbrellas and chairs; the motion
    track knows where the swimmer actually is. Hip midpoint is mapped into
    the reference image and compared with the track (interpolated over gaps
    up to 1 s). Rejected frames become undetected (NaN), like any miss.
    """
    gated = pose.copy()
    hips = (pose[["left_hip_x", "left_hip_y"]].to_numpy() + pose[["right_hip_x", "right_hip_y"]].to_numpy()) / 2
    hips_ref = np.array([to_reference(h[None], motion.H[i])[0] for i, h in enumerate(hips)])
    t = track["t"].to_numpy()
    ok = track["x"].notna().to_numpy()
    tx = np.interp(t, t[ok], track["x"].to_numpy()[ok]) if ok.any() else np.full(len(t), np.nan)
    ty = np.interp(t, t[ok], track["y"].to_numpy()[ok]) if ok.any() else np.full(len(t), np.nan)
    if ok.any():  # don't bridge long gaps
        last_seen = np.maximum.accumulate(np.where(ok, t, -np.inf))
        next_seen = np.minimum.accumulate(np.where(ok, t, np.inf)[::-1])[::-1]
        far = (t - last_seen > 1.0) & (next_seen - t > 1.0)
        tx[far], ty[far] = np.nan, np.nan
    n = min(len(pose), len(t))
    d = np.linalg.norm(hips_ref[:n] - np.c_[tx[:n], ty[:n]], axis=1)
    on = pose["detected"].to_numpy()[:n] & (d <= max_dist_px)
    reject = np.zeros(len(pose), bool)
    reject[:n] = ~on
    cols = [c for c in pose.columns if c.endswith(("_x", "_y", "_vis"))]
    gated.loc[reject, cols] = np.nan
    gated.loc[reject, "detected"] = False
    return gated, float(on.mean()) if n else float("nan")


def strokes_for_clip(clip: ClipData, force: bool, log=print) -> None:
    from swimstroke.pose import load_pose, save_pose, track_video
    from swimstroke.pose.mediapipe_pose import MediaPipePose

    cache = Path("outputs") / clip.name / "pose_mediapipe-heavy.parquet"
    if cache.exists() and not force:
        pose = load_pose(cache)
    else:
        log(f"  {clip.name}: pose estimation")
        with MediaPipePose("heavy") as est:
            pose = track_video(clip.video, est)
        save_pose(pose, cache)
    clip.pose, clip.pose_on_swimmer = gate_pose(pose, clip.motion, clip.track)
    clip.cycles = detect_cycles(clip.pose)


# ---------- per session ----------


def pushoff_time(clip: ClipData, segs, setup: Setup) -> float | None:
    """Detect the push-off at the start wall (see laps.detect_pushoff)."""
    if not segs or not setup.walls_px or not setup.lane:
        return None
    side = "left" if segs[0].direction > 0 else "right"  # it swims away from this wall
    if side not in setup.walls_px:
        return None
    W = setup.walls_px[side]
    ly = setup.lane[0] * W + setup.lane[1]
    sgn = 1 if side == "right" else -1  # region extends past the virtual wall towards the real wall
    xa, xb = sorted([W - sgn * 150, W + sgn * 90])
    w, h = clip.motion.image_size
    roi = (int(max(0, xa)), int(min(w, xb)), int(max(0, ly - 45)), int(min(h, ly + 35)))
    tr = clip.track
    other = setup.walls_px.get("left" if side == "right" else "right")
    far_x = abs(tr["x"] - W) > (abs(W - other) / 2 if other is not None else 600)
    empty = list(tr.index[far_x & tr["x"].notna()][::15][:15])
    if len(empty) < 5:
        return None
    act = wall_activity(clip.video, clip.motion, roi, empty, t_max=segs[0].t[0] + 1.0)
    return detect_pushoff(act["t"].to_numpy(), act["activity"].to_numpy())


def analyze_session(
    videos: list[Path],
    setup: Setup,
    strokes: bool = True,
    force: bool = False,
    log=print,
) -> dict:
    ref_image = setup.ref_image()
    clips = [track_clip(v, setup, ref_image, force, log) for v in videos]

    # lane: median over clips of each clip's robust lane fit (stored for reuse)
    from swimstroke.track import fit_lane
    lanes = [fit_lane(c.track) for c in clips if c.track["x"].notna().sum() > 100]
    if lanes:
        setup.lane = np.median(np.array(lanes), axis=0).tolist()

    segs = {c.name: segments(c.track["t"].to_numpy(), c.track["x"].to_numpy()) for c in clips}
    vertices = pd.concat([turn_vertices(s).assign(clip=n) for n, s in segs.items()], ignore_index=True)
    found = wall_positions(vertices) if len(vertices) else {}
    setup.walls_px = {**setup.walls_px, **found}  # new turns override, missing sides keep old values
    setup.save()

    pool = PoolGeometry.load(setup.pool_file) if setup.pool_file else None

    all_lengths, all_turns, per_clip = [], [], {}
    for c in clips:
        s = segs[c.name]
        t0 = pushoff_time(c, s, setup)
        events = wall_events(s, setup.walls_px, start_time=t0)
        lengths = length_times(events).assign(clip=c.name)
        lengths["start_method"] = np.where(lengths["from_event"] == "start",
                                           "pushoff" if t0 is not None else "extrapolated", "")

        stroke_info: dict = {}
        if strokes:
            strokes_for_clip(c, force, log)
            stroke_info = summarize_strokes(c.cycles, c.pose)
            stroke_info["pose_on_swimmer"] = round(c.pose_on_swimmer, 3)
            add_strokes_per_length(lengths, c, s)

        X = None
        if pool is not None:
            X = pool.to_pool(c.track[["x", "y"]].to_numpy())[:, 0]
            if strokes:
                lengths["dps_m"] = [
                    distance_per_stroke(c.track["t"].to_numpy(), X, _entries(c, r))
                    for _, r in lengths.iterrows()
                ]
            tt = turn_times(c.track["t"].to_numpy(), X, events, pool.pool_length_m).assign(clip=c.name)
            all_turns.append(tt)

        out = Path("outputs") / c.name
        events.to_csv(out / "events.csv", index=False)
        lengths.to_csv(out / "lengths.csv", index=False)
        all_lengths.append(lengths)
        per_clip[c.name] = {"events": events, "lengths": lengths, "strokes": stroke_info, "X": X}

    return {
        "clips": clips,
        "per_clip": per_clip,
        "lengths": pd.concat(all_lengths, ignore_index=True) if all_lengths else pd.DataFrame(),
        "turns": pd.concat(all_turns, ignore_index=True) if all_turns else pd.DataFrame(),
        "vertices": vertices,
        "setup": setup,
    }


def _entries(c: ClipData, row) -> np.ndarray:
    if c.cycles is None or c.cycles.empty:
        return np.array([])
    e = c.cycles["t"].to_numpy()
    return e[(e > row["t_start"]) & (e <= row["t_end"])]


def add_strokes_per_length(lengths: pd.DataFrame, c: ClipData, segs, min_coverage: float = 0.6) -> None:
    """Strokes and stroke rate per length, only where pose covered the length.

    Coverage = share of the length's swimming time with a usable stroke
    signal; below min_coverage the count would silently miss strokes, so it
    is reported as missing instead.
    """
    from swimstroke.strokes import StrokeParams, cycle_intervals, stroke_signal

    t_sig, y_sig = stroke_signal(c.pose)
    has = ~np.isnan(y_sig)
    counts, rates, cover = [], [], []
    for _, r in lengths.iterrows():
        seg = next((s for s in segs if s.t[0] >= r["t_start"] - 0.5 and s.t[-1] <= r["t_end"] + 0.5), None)
        swim_start = seg.t[0] if seg is not None else r["t_start"]
        span = (t_sig >= swim_start) & (t_sig <= r["t_end"])
        cov = float(has[span].mean()) if span.any() else 0.0
        e = _entries(c, r)
        seg_ids = c.cycles.loc[c.cycles["t"].isin(e), "segment"].to_numpy() if len(e) else np.array([])
        d = cycle_intervals(e, seg_ids, StrokeParams().max_cycle_s) if len(e) > 1 else np.array([])
        period = float(np.median(d)) if len(d) else float("nan")
        n = strokes_in_length(e, swim_start, r["t_end"], period) if cov >= min_coverage else None
        counts.append(n)
        rates.append(round(60 / period, 1) if cov >= min_coverage and np.isfinite(period) else None)
        cover.append(round(cov, 2))
    lengths["strokes"] = pd.array(counts, dtype="Int64")
    lengths["cycle_rate_per_min"] = rates
    lengths["pose_coverage"] = cover

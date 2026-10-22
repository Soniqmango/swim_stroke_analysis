# Pose model comparison: MediaPipe vs RTMPose (Sports2D)

First step of the kinematics milestone: before measuring joint angles,
check which pose model actually sees the swimmer.

- **MediaPipe Pose Landmarker (heavy)**, the model used so far.
- **RTMPose (balanced, HALPE-26 "body with feet")**, the model Sports2D uses,
  added as a second backend in `src/swimstroke/pose/rtmpose_pose.py`.

Run with `uv run python compare_pose.py VIDEO [VIDEO ...]`. Each model is
scored at its own recommended confidence threshold (MediaPipe 0.5, RTMPose
0.3 as in Sports2D). "Seen" = share of all frames where at least one side of
that joint is above the threshold.

## Clips

- **U1–U5**: underwater side view, breaststroke, 720x576 at 25 fps, fixed
  camera (NIE, provided by Assoc Prof John Komar). Swimmer names removed.
- **A1** (`IMG_0073`): above-water pool-deck phone video, freestyle, 1122 frames.

## Results

| Clip | Model | Shoulder | Elbow | Wrist | Hip | Knee | Ankle |
|---|---|---|---|---|---|---|---|
| U1 | MediaPipe | 0.41 | 0.16 | 0.14 | 0.41 | 0.09 | 0.35 |
| U1 | RTMPose | 0.84 | 0.84 | 0.78 | 0.90 | 0.86 | 0.94 |
| U2 | MediaPipe | 0.95 | 0.95 | 0.95 | 0.95 | 0.82 | 0.86 |
| U2 | RTMPose | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| U3 | MediaPipe | 0.86 | 0.75 | 0.68 | 0.86 | 0.83 | 0.83 |
| U3 | RTMPose | 0.98 | 0.98 | 0.97 | 0.99 | 0.99 | 0.99 |
| U4 | MediaPipe | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| U4 | RTMPose | 0.08 | 0.05 | 0.07 | 0.57* | 0.54* | 0.79* |
| U5 | MediaPipe | 0.93 | 0.93 | 0.93 | 0.93 | 0.34 | 0.29 |
| U5 | RTMPose | 0.99 | 0.99 | 0.98 | 0.99 | 0.99 | 0.99 |
| A1 | MediaPipe | 0.23 | 0.08 | 0.10 | 0.23 | 0.05 | 0.05 |
| A1 | RTMPose | 0.38 | 0.36 | 0.37 | 0.34 | 0.34 | 0.56 |

## What this shows

- **Underwater, RTMPose sees the whole body almost all the time** (U1–U3, U5:
  wrists 78–100%, knees and ankles 86–100%). MediaPipe often loses the legs
  (U5 knees 34%), which rules it out for kick frequency and arm-leg
  coordination.
- **Above water (A1), RTMPose sees joints about 1.5–7x more often** than
  MediaPipe, but still only about a third of the time: splash and the water
  surface hide the arms. Underwater video is the better source for kinematics.
- Spot checks of overlaid skeletons confirm RTMPose sits on the swimmer in
  U2 and A1 where MediaPipe found nothing.

## Known problem: false detections

RTMPose always returns *some* person (detection rate 100% on every clip),
even when the swimmer is not found. In **U4** (faint, distant swimmer) the
"skeleton" is mostly on the lane rope, so the starred numbers are largely
false positives. Next step: reject implausible poses (box size, limb lengths,
jumps between frames) and gate on the swimmer's track, before using the
joints for kinematics. Contrast enhancement may also help faint clips like U4.

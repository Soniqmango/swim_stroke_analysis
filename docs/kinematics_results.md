# Breaststroke kinematics: results so far

Underwater side-view breaststroke clips U1–U5 (same labels as
[pose_comparison.md](pose_comparison.md); swimmer names removed).

```bash
uv run python analyze_kinematics.py VIDEO [VIDEO ...] --overlay
```

Code: `src/swimstroke/kinematics.py`. Tests on a synthetic swimmer with
known timings: `tests/test_kinematics.py`.

**Status: automatic results, not yet checked against hand labels.** Spot
checks of video frames are described below; see "How to check by hand".

## Pipeline

1. **Pose:** RTMPose (HALPE-26, with heels and toes). The backend follows the
   same person from frame to frame.
2. **Plausibility gate:** a frame is kept only if shoulders and hips are
   confidently seen, the torso is 0.6–1.6x the clip's typical length, thigh
   and upper arm have human proportions, and the hips don't jump more than
   0.4 torso lengths between frames. Kept stretches under 0.5 s are dropped.
3. **Signals** along the swimmer's own body axis, in torso lengths (TL), so
   they don't depend on swim direction or camera distance:
   - *leg reach* = heels relative to the hips
   - *arm reach* = wrists relative to the shoulders

   A 0.2 s median filter first removes one-frame left/right label swaps.
4. **Cycles:** one leg recovery end (heels drawn up to the hips) to the next.
5. **Events** in each cycle. "Extended" = within 20% of the cycle's range of
   full extension, with the range taken between the 5th and 95th
   percentiles so one outlier frame can't set it:

   | Event | Definition |
   |---|---|
   | leg recovery end | heels closest to the hips (cycle start, 0%) |
   | kick end | first time the legs are extended |
   | leg recovery start | last time the legs are extended |
   | arm extension | first time the arms are extended after the previous pull-in |
   | pull start | last time the arms are extended before the pull-in |
   | pull-in | hands drawn back the furthest (end of insweep) |

   Arm events are only timed if arm reach swings at least 1.0 TL in that
   cycle; otherwise they are reported as not measurable.

6. **Phases** (% of cycle): legs = kick, glide, recovery; arms = glide, pull,
   recovery.
7. **Arm-leg coordination**, using the time-gap approach Chollet's group uses
   for breaststroke, in % of cycle:
   - **T1** = pull start − kick end (glide between leg and arm propulsion)
   - **T2** = leg recovery start − pull-in (arm recovery vs leg recovery)
   - **T3** = kick end − arm extension (arms vs legs reaching full extension)
   - **TTG** = T1 + T2 + T3

   Note: the paper John sent (Chollet, Chalies & Chatard 2000) defines the
   Index of Coordination and its four arm phases for **front crawl**; that is
   for the freestyle step. The breaststroke definitions above should be
   checked against the ones the lab uses (they follow the same idea, but the
   thresholds are ours: 80% extension rather than 90%, see Limitations).
8. **Kick frequency** from the intervals between leg recoveries alone, so it
   doesn't depend on the arms being measurable.
9. **Joint angles** (knee, hip, elbow, shoulder, per side) in the image
   plane, and **left/right asymmetry**: symmetry index of knee and elbow range
   of motion, 100 × (L − R) / mean, and the left-vs-right heel-up lag.
10. **Hip velocity** along the swim direction, TL/s, relative to the camera.

## Results

| | U2 | U3 | U5 |
|---|---|---|---|
| Frames kept | 94% | 79% | 89% |
| Cycles | 9 | 5 | 4 |
| Rate / kicks per min | 31.6 | 50.8 | 26.9 |
| Arm events measurable | 100% | 0% | 100% |
| **Leg phases** (kick / glide / recovery, %) | 15 / 58 / 26 | 26 / 46 / 28 | 14 / 62 / 25 |
| **Arm phases** (glide / pull / recovery, %) | 49 / 21 / 30 | – | 53 / 18 / 29 |
| Pull-in (% of cycle) | 76 | – | 78 |
| **T1** glide | 42 | – | 45 |
| **T2** | −3 | – | −0.5 |
| **T3** | 9 | – | 6 |
| **TTG** | 50 | – | 51 |
| Minimum knee angle (deg) | 54 | 54 | 52 |
| Knee ROM asymmetry (SI, %) | 2.8 | 2.7 | 0.6 |
| Elbow ROM asymmetry (SI, %) | 10.5 | −13.5 | −6.8 |
| Hip velocity SD within a cycle (TL/s) | 0.55 | 0.81 | 0.39 |

U1 and U4: no frames passed the gate (see below), so no results.

Cycle-to-cycle spread (SD) on U2 and U5: kick end ±1–4%, leg recovery start
±6%, pull start ±3–5%, pull-in ±2%, T1 ±5–6%, T2 ±5–6%. Arm extension on U5
±2%; on U2 ±11%, from one outlier cycle (38% vs 2–12% for the others).

## Checked by eye

- **U2, cycle events:** at each detected leg recovery end the heels are
  drawn up to the hips; 0.5 s later the legs are extended and the arms are
  in the glide; at the detected pull-in the hands are under the chest.
- **U2 overlay:** at 5.00 s the HUD shows legs KICK and arms GLIDE just after
  both arm extension and leg recovery end; at 6.60 s PULL IN with both limbs
  in recovery, matching the frames.
- **U3:** leg events are right. The arm signal swings only 0.5–0.85 TL (vs
  1.3–1.7 on U2/U5): the hands stay near the surface and the wrist labels
  swap, and earlier versions picked the glide as a "pull-in". Arm events are
  now reported as not measurable for U3 instead of wrong.
- **U1:** two swimmers overlap so the detector draws one box around both and
  mixes their joints; rejected. **U4:** faint swimmer, the pose sits on the
  lane rope; rejected.

## Limitations

- **Camera moves with the swimmer** (trolley): mean hip velocity relative to
  the camera is ~0. The velocity shape and spread within a cycle are real,
  but the mean speed isn't, so intracyclic velocity variation as SD / mean
  (%) can't be computed. Needs the trolley speed or a fixed camera.
- **No metre scale:** velocities in torso lengths per second.
- **Asymmetry from a side view is weak:** the far limbs are mostly hidden
  behind the near ones and the model often places both knees on the visible
  leg, so knee asymmetry close to 0 is partly an artefact. A frontal or rear
  underwater camera is needed for reliable left/right comparison.
- **Joint angles are 2D** (in the image plane); breaststroke leg movement is
  partly sideways, so the knee angle understates flexion when the knees open
  outwards.
- **80% instead of 90% extension:** with 90%, the noise in the video split
  the extended phase into pieces (event spread ±13–24% of a cycle); 80% gave
  ±1–6%.
- **Smoothing bias:** on the synthetic swimmer, events land ~2.5% of a cycle
  late (under 2 frames at 25 fps); time gaps combine two events, so up to ~5%.
- Breaststroke has one kick per cycle, so kick frequency equals the cycle
  rate here; it becomes informative for freestyle (2-, 4-, 6-beat kicks).

## How to check by hand

1. Run with `--overlay`. Open `outputs/<clip>/kinematics_overlay.mp4` and
   step frame by frame: the HUD shows the frame number, the current leg and
   arm phase, and flashes each event.
2. Note the frame where *you* see each event and add rows to
   `data/labels/kinematics_events.csv` (`t` = frame / 25). Event names and
   definitions are at the top of that file. It is not tracked in git, since
   clip names contain swimmers' names.
3. Re-run the script: `outputs/kinematics/validation.csv` lists, per event
   type, hits, misses, false detections and the mean timing offset
   (tolerance 3 frames).

## Next

- Freestyle: the Index of Coordination (Chollet et al. 2000), arm phases
  A–D, kick beats per cycle, left/right arm asymmetry.
- A fixed-camera recording for true velocity and metres.

# Breaststroke kinematics: first results

Second step of the kinematics milestone, on the five underwater side-view
breaststroke clips (U1–U5, same labels as [pose_comparison.md](pose_comparison.md)).
Run with:

```bash
uv run python analyze_kinematics.py VIDEO [VIDEO ...]
```

Code: `src/swimstroke/kinematics.py`. Tests on a synthetic swimmer:
`tests/test_kinematics.py`.

## Method

1. **Pose:** RTMPose (HALPE-26, with heels and toes). The backend now
   follows the same person from frame to frame, so a second swimmer in view
   doesn't steal the pick.
2. **Plausibility gate:** a frame is kept only if shoulders and hips are
   confidently seen, the torso is 0.6–1.6x the clip's typical length, the
   thigh and upper arm have human proportions, and the hips don't jump more
   than 0.4 torso lengths between frames. Kept stretches shorter than 0.5 s
   are dropped.
3. **Cycles from the legs:** heel position along the body axis peaks once
   per cycle, when the heels are drawn up to the hips just before the kick.
   One cycle = one leg recovery to the next.
4. **Arm check:** each cycle should hold one arm pull-in (hands drawn under
   the chest = dip in arm reach along the body axis). Its position in the
   cycle (% after leg recovery) is a first arm-leg timing measure.
5. **Hip velocity** along the swim direction, in torso lengths per second
   (TL/s), from the mid-hip position.

## Results

| Clip | Frames kept | Cycles | Rate (cycles/min) | Arm-leg agreement | Arm pull-in (% of cycle) | Hip velocity SD (TL/s) |
|---|---|---|---|---|---|---|
| U1 | 0% | – | – | – | – | – |
| U2 | 94% | 9 | 31.6 | 100% | 76 (71–77) | 0.55 |
| U3 | 79% | 5 | 51.7 | 80% | unreliable | 0.81 |
| U4 | 0% | – | – | – | – | – |
| U5 | 89% | 4 | 26.9 | 100% | 78 (75–80) | 0.39 |

Arm-leg agreement = share of cycles with exactly one arm pull-in.

## Checked by eye

- **U2:** frames at the detected events show the heels drawn up to the hips
  at each leg event, legs extended and arms in the glide 0.5 s later, and
  hands pulled in under the chest at the arm event. Cycle lengths are very
  regular (1.82–1.98 s).
- **U3:** leg events are correct, but arm events are not: one detected
  "pull-in" is actually the glide, and the timing jumps between cycles
  (9–67%). Arm-leg timing for U3 is not trusted yet.
- **U1:** two swimmers overlap so closely that the detector draws one box
  around both and mixes their joints. The gate correctly rejects every frame.
- **U4:** faint, distant swimmer; the "pose" is on the lane rope. Rejected.

## Limitations

- **The camera moves with the swimmer** (trolley): lane-rope sections change
  across the clip, and the mean hip velocity relative to the camera is ~0 in
  every clip. So hip velocity here is relative to the camera. Its shape and
  spread within a cycle are real, but the mean swimming speed is not, which
  means intracyclic velocity variation as a percentage (SD / mean) can't be
  computed yet. It needs the camera's speed or a fixed camera.
- No metre scale: velocities are in torso lengths per second.
- Arm events need work for swimmers whose arms stay close to the surface (U3).
- No hand-labelled reference yet: the next validation step is to mark leg
  recovery and arm pull-in times by hand on a few cycles and compare.

## Next

- Coordination and stroke phases as defined by Chollet et al. (2000).
- Kick frequency and left/right asymmetry.
- Freestyle clips, and a fixed-camera recording to get true speed.

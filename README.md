# Swim Stroke Analysis

Computer-vision tool that measures freestyle stroke rate, stroke count, splits,
distance per stroke and turn time from a pool-deck phone video, validated
against coach hand timing.

> Work in progress. Method, demo and accuracy results will be added as the milestones are completed.

## Setup

Requires [uv](https://docs.astral.sh/uv/). It installs the correct Python version and every dependency, including a bundled ffmpeg:

```bash
uv sync
uv run pytest
```

## Pose models

Two interchangeable pose backends (`src/swimstroke/pose/`): MediaPipe and
RTMPose, the model behind [Sports2D](https://github.com/davidpagnon/Sports2D).
Compare them on your own clips with:

```bash
uv run python compare_pose.py VIDEO [VIDEO ...]
```

Results so far: [docs/pose_comparison.md](docs/pose_comparison.md).

## Breaststroke kinematics (underwater side view)

Cleans the poses, detects cycles from the legs (arms as a check) and
measures hip velocity within each cycle:

```bash
uv run python analyze_kinematics.py VIDEO [VIDEO ...]
```

First results: [docs/kinematics_results.md](docs/kinematics_results.md).

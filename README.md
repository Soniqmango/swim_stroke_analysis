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

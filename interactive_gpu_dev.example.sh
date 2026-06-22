#!/bin/bash
# Example interactive GPU session for a live Oasis smoke test.
# Set `dev_run = true` and `device = "cuda"` in `experiment.toml` first.

set -euo pipefail

srun \
  --job-name="oasis-dev" \
  --partition=your-gpu-partition \
  --account=your-account \
  --gpus=1 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=16G \
  --time=01:00:00 \
  --pty bash -l -c '
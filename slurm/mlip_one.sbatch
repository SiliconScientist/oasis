#!/bin/bash
#SBATCH -J oasis-mlip
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o slurm_output/%x_%A_%a.out
#SBATCH -e slurm_output/%x_%A_%a.err

set -euo pipefail

TASKFILE="$1"
CONFIG="${2:-config.toml}"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKFILE")"
echo "TASK: $LINE"

# Use the driver env to parse config + dispatch
python -m oasis mlip run-one --line "$LINE" --config "$CONFIG"

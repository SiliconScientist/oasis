#!/bin/bash
#SBATCH --job-name="oasis-mlip"
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA40x4
#SBATCH --account=bchg-delta-gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=10G
#SBATCH -t 24:00:00
#SBATCH -o slurm_output/%x_%A_%a.out
#SBATCH -e slurm_output/%x_%A_%a.err
#SBATCH --mail-user=ahill15@tulane.edu
#SBATCH --mail-type=END


set -euo pipefail

TASKFILE="$1"
CONFIG="${2:-config.toml}"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKFILE")"
echo "TASK: $LINE"

# Use the driver env to parse config + dispatch
python -m oasis mlip run-one --line "$LINE" --config "$CONFIG"

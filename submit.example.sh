#!/bin/bash
# Example Slurm submission script for running Oasis on a GPU partition.

#SBATCH --job-name="oasis"
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --partition=your-gpu-partition
#SBATCH --account=your-account
#SBATCH --cpus-per-task=32
#SBATCH --mem=10G
#SBATCH -t 24:00:00
#SBATCH -o slurm_output/%x_%A_%a.out
#SBATCH -e slurm_output/%x_%A_%a.err
#SBATCH --mail-user=your-email@example.com
#SBATCH --mail-type=END

set -euo pipefail

source .venv/bin/activate
python -m oasis

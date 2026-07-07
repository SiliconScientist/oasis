#!/bin/bash
# Example Slurm submission script for running Oasis on a GPU partition.
# Submit this script directly: ./submit.example.sh [experiment.toml]

#SBATCH --job-name="oasis"
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --partition=your-gpu-partition
#SBATCH --account=your-account
#SBATCH --cpus-per-task=32
#SBATCH --mem=10G
#SBATCH -t 24:00:00
#SBATCH --mail-user=your-email@example.com
#SBATCH --mail-type=END

set -euo pipefail

source .venv/bin/activate

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  eval "$(python -m oasis.submission --run-tag oasis "$@")"
  sbatch \
    --output="${RUN_DIR}/slurm_%x_%j.out" \
    --error="${RUN_DIR}/slurm_%x_%j.err" \
    --export=ALL,OASIS_ARGV_FILE="${OASIS_ARGV_FILE}" \
    "$0"
  exit 0
fi

if [[ -z "${OASIS_ARGV_FILE:-}" ]]; then
  echo "Missing OASIS_ARGV_FILE. Submit with ./submit.example.sh so the config is frozen first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${OASIS_ARGV_FILE}"
python -m oasis "${OASIS_ARGV[@]}"

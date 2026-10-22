#!/bin/bash
#SBATCH --job-name=wlm-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Pass the array bounds at submission; do not submit all experiments before a pilot.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export WLM_PROJECT_DIR="$PWD"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
CONDA_BASE_DIR="$(conda info --base)"
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
conda activate wlm
cd "$WLM_PROJECT_DIR"
python -m wlm.sweep run --jobs "${1:-jobs.jsonl}" --index "$SLURM_ARRAY_TASK_ID" --device cuda

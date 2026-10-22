#!/bin/bash
#SBATCH --job-name=wlm-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
# Run from the extracted starter directory. Create logs/ BEFORE sbatch.
cd "$SLURM_SUBMIT_DIR"
export WLM_PROJECT_DIR="$PWD"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
CONDA_BASE_DIR="$(conda info --base)"
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
conda activate wlm
cd "$WLM_PROJECT_DIR"
nvidia-smi
# Supply the writer-specific paths with: sbatch slurm/train.sbatch BASE MANIFEST OUT
python -m wlm.train --base "$1" --manifest "$2" --output "$3" \
  --method decoder --steps 200 --batch-size 4 --seed 0 --device cuda

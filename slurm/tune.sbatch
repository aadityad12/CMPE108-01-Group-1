#!/bin/bash
#SBATCH --job-name=wlm-tune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.out
# Learning-rate tuning on a development writer (never an evaluation writer).
#
# One array task = one (method, learning rate, steps) setting. 3 methods x
# 3 learning rates x 3 step budgets = 27 tasks, so submit with --array=0-26.
# The pilot showed the default rates memorize 10 images and hurt unseen ones,
# so the grid searches downward from the defaults.
#
#   mkdir -p logs
#   sbatch --array=0-26%2 slurm/tune.sbatch SUPPORT.jsonl HELDOUT.jsonl runs/tune/NAME
#
# Each task writes OUT/METHOD_lrLR_sSTEPS/{model.pt,support/,heldout/}.
# Collect the scores afterwards with: python scripts/tune_summary.py runs/tune/NAME
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wlm

SUPPORT=$1
HELDOUT=$2
OUT=$3

METHODS=(lora decoder full)
LORA_LRS=(1e-4 3e-4 1e-3)
DECODER_LRS=(1e-5 2e-5 5e-5)
FULL_LRS=(2e-6 5e-6 1e-5)
STEPS_LIST=(25 50 200)

i=$SLURM_ARRAY_TASK_ID
method=${METHODS[$((i / 9))]}
lr_index=$(((i / 3) % 3))
steps=${STEPS_LIST[$((i % 3))]}
case $method in
  lora) lr=${LORA_LRS[$lr_index]} ;;
  decoder) lr=${DECODER_LRS[$lr_index]} ;;
  full) lr=${FULL_LRS[$lr_index]} ;;
esac

run=$OUT/${method}_lr${lr}_s${steps}
echo "task $i: $method lr=$lr steps=$steps -> $run"
python -m wlm.train --base checkpoints/base.pt --manifest "$SUPPORT" --output "$run" \
  --method "$method" --lr "$lr" --steps "$steps" --batch-size 4 --seed 0 --device cuda
for split in support heldout; do
  manifest=$SUPPORT
  [ "$split" = heldout ] && manifest=$HELDOUT
  python -m wlm.predict evaluate --checkpoint "$run/model.pt" --manifest "$manifest" \
    --output "$run/$split" --kind general --device cuda
done

#!/bin/bash
# slurm/05_gather.sbatch
#SBATCH --job-name=pf05-gather
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
$PYRAFAULT_PY scripts/05_sample_visual_diversity.py --config "$PYRAFAULT_CONFIG" \
              --data-root "$PYRAFAULT_DATA" --gather

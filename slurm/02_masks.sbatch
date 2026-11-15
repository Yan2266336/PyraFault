#!/bin/bash
# slurm/02_masks.sbatch  —  sbatch --array=0-9 slurm/02_masks.sbatch
#SBATCH --job-name=pf02-masks
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
SLIDE=$(slide_for_index "$SLURM_ARRAY_TASK_ID")
$PYRAFAULT_PY scripts/02_build_tissue_masks.py --config "$PYRAFAULT_CONFIG" \
              --data-root "$PYRAFAULT_DATA" --slides "$SLIDE"

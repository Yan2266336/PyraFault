#!/bin/bash
# slurm/07_mine.sbatch
#SBATCH --job-name=pf07-mine
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
$PYRAFAULT_PY scripts/07_mine_hard_negatives.py --config "$PYRAFAULT_CONFIG" \
                                                --data-root "$PYRAFAULT_DATA"

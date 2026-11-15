#!/bin/bash
# slurm/06_pairs.sbatch
#SBATCH --job-name=pf06-pairs
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
$PYRAFAULT_PY scripts/06_build_pairs.py --config "$PYRAFAULT_CONFIG" \
                                        --data-root "$PYRAFAULT_DATA"

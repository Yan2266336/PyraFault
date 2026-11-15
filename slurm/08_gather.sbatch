#!/bin/bash
# slurm/08_gather.sbatch
#SBATCH --job-name=pf08-gather
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
$PYRAFAULT_PY scripts/08_materialize_cache.py --config "$PYRAFAULT_CONFIG" \
              --data-root "$PYRAFAULT_DATA" --gather

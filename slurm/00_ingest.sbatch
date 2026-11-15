#!/bin/bash
# slurm/00_ingest.sbatch
#SBATCH --job-name=pf00-validate
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
$PYRAFAULT_PY scripts/00_ingest_source.py --config "$PYRAFAULT_CONFIG" \
                                                --data-root "$PYRAFAULT_DATA"

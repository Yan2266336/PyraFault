#!/bin/bash
# slurm/09_qc.sbatch — the acceptance gate. Non-zero exit means the benchmark is not shippable.
#SBATCH --job-name=pf09-qc
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
$PYRAFAULT_PY scripts/09_run_qc.py --config "$PYRAFAULT_CONFIG" \
                                   --data-root "$PYRAFAULT_DATA"

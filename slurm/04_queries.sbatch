#!/bin/bash
# slurm/04_queries.sbatch — no pixel reads; one task, one table.
#SBATCH --job-name=pf04-queries
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
$PYRAFAULT_PY scripts/04_build_query_candidates.py --config "$PYRAFAULT_CONFIG" \
                                                   --data-root "$PYRAFAULT_DATA"

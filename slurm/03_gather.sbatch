#!/bin/bash
# slurm/03_gather.sbatch — merges shards and writes splits.parquet. Must run after the array.
#SBATCH --job-name=pf03-gather
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
$PYRAFAULT_PY scripts/03_build_context_candidates.py --config "$PYRAFAULT_CONFIG" \
              --data-root "$PYRAFAULT_DATA" --gather

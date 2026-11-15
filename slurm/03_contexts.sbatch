#!/bin/bash
# slurm/03_contexts.sbatch  —  sbatch --array=0-9 slurm/03_contexts.sbatch
#SBATCH --job-name=pf03-contexts
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
SLIDE=$(slide_for_index "$SLURM_ARRAY_TASK_ID")
$PYRAFAULT_PY scripts/03_build_context_candidates.py --config "$PYRAFAULT_CONFIG" \
              --data-root "$PYRAFAULT_DATA" --slides "$SLIDE"

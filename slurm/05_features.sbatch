#!/bin/bash
# slurm/05_features.sbatch  —  sbatch --array=0-9 slurm/05_features.sbatch
# Set PYRAFAULT_EXTRACTOR=dinov2 in the environment until the UNI licence is accepted.
#SBATCH --job-name=pf05-features
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
SLIDE=$(slide_for_index "$SLURM_ARRAY_TASK_ID")
EXTRA=()
[ -n "${PYRAFAULT_EXTRACTOR:-}" ] && EXTRA=(--extractor "$PYRAFAULT_EXTRACTOR")
$PYRAFAULT_PY scripts/05_sample_visual_diversity.py --config "$PYRAFAULT_CONFIG" \
              --data-root "$PYRAFAULT_DATA" --slides "$SLIDE" "${EXTRA[@]}"

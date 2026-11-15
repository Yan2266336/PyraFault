#!/bin/bash
# slurm/01_scan.sbatch
#SBATCH --job-name=pf01-scan
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
$PYRAFAULT_PY scripts/01_scan_slides.py --config "$PYRAFAULT_CONFIG" \
                                        --data-root "$PYRAFAULT_DATA"
$PYRAFAULT_PY slurm/make_slide_list.py "$PYRAFAULT_DATA"

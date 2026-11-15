#!/bin/bash
# slurm/run_all.sh — submit stages 00–09 as one afterok chain.
#
#   bash slurm/run_all.sh                 # from stage 00
#   bash slurm/run_all.sh 05              # resume at stage 05
#   PYRAFAULT_EXTRACTOR=dinov2 bash slurm/run_all.sh 05
#
# Stage 00 must have run at least once before any array stage: the array size comes
# from $PYRAFAULT_DATA/slides.txt, which stage 01 writes.
set -euo pipefail
cd "$(dirname "$0")/.."
START="${1:-00}"

# Ruling AC: after() compares stage labels as strings ("00" < "01" < ... < "09"), so
# every valid label must be zero-padded -- a bare "5" compares GREATER than all of
# them lexicographically ("5" > "09"), meaning after() would return false for every
# single stage. That is not "stage 5 gets skipped": it is "the entire chain submits
# nothing at all", while this script still prints a normal-looking "submitting from
# stage 5" summary line and exits 0. Zero-pad a bare single digit, then validate
# against the known stage list and fail loudly rather than silently submitting an
# empty chain on any other unrecognised value.
VALID_STAGES=(00 01 02 03 04 05 06 07 08 09)
if [[ "$START" =~ ^[0-9]$ ]]; then
  START="0$START"
fi
valid=false
for s in "${VALID_STAGES[@]}"; do
  if [ "$START" = "$s" ]; then valid=true; break; fi
done
if [ "$valid" != true ]; then
  echo "error: unrecognised stage '$1' (resolved to '$START'); expected one of: ${VALID_STAGES[*]}" >&2
  exit 1
fi

DEP=""

submit() {           # submit <script> [extra sbatch args...]
  local script="$1"; shift
  local out
  out=$(sbatch --parsable ${DEP:+--dependency=afterok:$DEP} "$@" "$script")
  DEP="${out%%;*}"
  echo "  $(basename "$script")  -> job $DEP"
}

after() { [ "$START" \< "$1" ] || [ "$START" = "$1" ]; }

SLIDES_TXT="${PYRAFAULT_DATA:-data}/slides.txt"
if [ -f "$SLIDES_TXT" ]; then
  N=$(( $(wc -l < "$SLIDES_TXT") - 1 ))
else
  N=9
  echo "$SLIDES_TXT not found — assuming --array=0-9; rerun from stage 00 if wrong"
fi

echo "submitting from stage $START (array 0-$N)"
after 00 && submit slurm/00_ingest.sbatch
after 01 && submit slurm/01_scan.sbatch
after 02 && submit slurm/02_masks.sbatch      --array="0-$N"
after 03 && { submit slurm/03_contexts.sbatch --array="0-$N"; submit slurm/03_gather.sbatch; }
after 04 && submit slurm/04_queries.sbatch
after 05 && { submit slurm/05_features.sbatch --array="0-$N"; submit slurm/05_gather.sbatch; }
after 06 && submit slurm/06_pairs.sbatch
after 07 && submit slurm/07_mine.sbatch
after 08 && { submit slurm/08_materialize.sbatch --array="0-$N"; submit slurm/08_gather.sbatch; }
after 09 && submit slurm/09_qc.sbatch

echo "final job: $DEP    watch with:  squeue -u \$USER   /   tail -f logs/pf09-qc-*.out"

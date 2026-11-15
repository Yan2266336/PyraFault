# slurm/env.sh — sourced by every job. One place to configure paths.
#
# Nothing here is site-specific. PYRAFAULT_ROOT comes from wherever the job was
# submitted (run_all.sh submits from the repo root), and PYRAFAULT_PY defaults to
# whatever `python` is on PATH — so an activated conda env just works. Override
# either from the environment when that is not what you want:
#
#   export PYRAFAULT_PY=/path/to/your/conda/env/bin/python
#
# $(dirname "$0") is deliberately NOT used: sbatch copies the batch script to a
# spool directory, so $0 does not point into slurm/ at run time.
export PYRAFAULT_ROOT="${PYRAFAULT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYRAFAULT_PY="${PYRAFAULT_PY:-python}"

if ! "$PYRAFAULT_PY" -c "import openslide, timm, torch" 2>/dev/null; then
  echo "PYRAFAULT_PY ($PYRAFAULT_PY) cannot import openslide/timm/torch." >&2
  echo "Activate the environment, or set PYRAFAULT_PY to its python." >&2
  exit 1
fi
# pyrafault is not pip-installed; scripts/ would otherwise be sys.path[0], not the repo root.
export PYTHONPATH="$PYRAFAULT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Overridable, like PYRAFAULT_ROOT/PYRAFAULT_PY above: a new cohort is a new config
# and its own data root, and neither should require editing this file.
#   PYRAFAULT_CONFIG=configs/my_cohort.yaml PYRAFAULT_DATA=data_my_cohort \
#     bash slurm/run_all.sh
export PYRAFAULT_CONFIG="${PYRAFAULT_CONFIG:-configs/example10.yaml}"
export PYRAFAULT_DATA="${PYRAFAULT_DATA:-data}"
# The array slide list lives in the data root, written by stage 01, so two cohorts
# run side by side without one silently inheriting the other's slide list.
export SLIDE_LIST="$PYRAFAULT_DATA/slides.txt"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$PYRAFAULT_ROOT" || exit 1
mkdir -p logs

# $1 -> the slide_id for this array index, or exit 1 if the list is short.
slide_for_index() {
  local line
  line=$(sed -n "$(( $1 + 1 ))p" "$SLIDE_LIST")
  [ -n "$line" ] || { echo "no slide at index $1 in $SLIDE_LIST" >&2; exit 1; }
  echo "$line"
}

#!/bin/bash
#SBATCH --job-name=bl_spatial
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --exclude=a0105
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=08:00:00
#SBATCH --array=0-404
#SBATCH --output=logs/%x_%A_%a.out
#
# Published spatial AAD classifiers (DARNet, ListenNet, DBPNet; the authors' model code, fetched
# by baselines/spatial/fetch_thirdparty.sh) on the unified bundles: protocol A (CoupleFlow's
# content folds, subject-independent and per-listener) and protocol B (the papers' within-trial
# split).  One array index = one row of the task table:
#   python baselines/spatial/run.py --list          # number of tasks (405)
# Submit from the repo root (scripts/baselines/spatial.sh does this):
#   sbatch scripts/slurm/baseline_spatial.sbatch
# Extra arguments are passed to baselines/spatial/run.py.  run.py skips tasks whose output exists.
set -eo pipefail
# repo root: the submit directory under Slurm (submit from the repo root), else this file's ../..
ROOT=$(dirname "$0")/../..
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/coupleflow/paths.py" ]; then
    ROOT=$SLURM_SUBMIT_DIR
fi
cd "$ROOT"
source "${COUPLEFLOW_ENV:-scripts/env/osc.sh}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=8
echo "=== spatial task ${SLURM_ARRAY_TASK_ID:-0} :: $* ==="; date
nvidia-smi -L || true
python -u baselines/spatial/run.py --task "${SLURM_ARRAY_TASK_ID:-0}" "$@"
date

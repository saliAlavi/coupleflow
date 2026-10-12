#!/bin/bash
#SBATCH --job-name=cf_build
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --exclude=a0105
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#
# Build one preprocessed bundle from a raw release: `python -u "$@"` on a node sized for the
# multi-process readers (--workers W processes x 2 BLAS threads each).  Submit from the repo
# root; scripts/data/build_bundles.sh submits one job per bundle:
#   sbatch -J build_dtu scripts/slurm/build.sbatch coupleflow/preprocessing/build.py \
#       --dataset dtu --workers 12
#   sbatch -J build_megmasc_native --export=ALL,MEGMASC_NATIVE=1 scripts/slurm/build.sbatch \
#       coupleflow/preprocessing/build.py --dataset megmasc --workers 8 --name unified_native.pt
# Without Slurm the same body runs in the foreground:  bash scripts/slurm/build.sbatch <args>
# Environment file: $COUPLEFLOW_ENV (default scripts/env/osc.sh, the cluster the paper ran on).
set -eo pipefail
# repo root: the submit directory under Slurm (submit from the repo root), else this file's ../..
ROOT=$(dirname "$0")/../..
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/coupleflow/paths.py" ]; then
    ROOT=$SLURM_SUBMIT_DIR
fi
cd "$ROOT"
source "${COUPLEFLOW_ENV:-scripts/env/osc.sh}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2} \
    OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
echo "=== build: python $* (MEGMASC_NATIVE=${MEGMASC_NATIVE:-}) ==="; date
python -u "$@"
date

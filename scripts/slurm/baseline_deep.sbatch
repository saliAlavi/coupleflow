#!/bin/bash
#SBATCH --job-name=bl_deep
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --exclude=a0105
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:45:00
#SBATCH --array=0-119
#SBATCH --output=logs/%x_%A_%a.out
#
# Deep AAD baselines (VLAAI, DECAF, dilated match-mismatch) under CoupleFlow's content-disjoint
# folds: one subject-independent model per (model, dataset, fold).  Array index
#   i = model x 40 + dataset x 5 + fold
#   model   = (vlaai decaf mm)[i / 40]
#   dataset = (kuleuven dtu esaa uhd255 fuglsang2020 broderick_cp sparrkulee megmasc)[(i / 5) % 8]
#   fold    = i % 5      (MEG-MASC: 35-39 VLAAI, 75-79 DECAF, 115-119 MM)
# so baselines/deep/run.py itself takes --model/--dataset/--fold (it has no --task option).
# Folds beyond a dataset's number of content folds exit immediately (run.py checks).
# Submit from the repo root (scripts/baselines/deep.sh does this):
#   sbatch scripts/slurm/baseline_deep.sbatch                          # all 120 indices
#   sbatch --array=40-79 scripts/slurm/baseline_deep.sbatch            # DECAF only
#   sbatch --array=0-9,80-89 --export=ALL,OUT_SUBDIR=results_leak scripts/slurm/baseline_deep.sbatch \
#       --groups-name unified_groups_trialsplit.json                   # trial-split leakage control
# Environment:
#   OUT_SUBDIR   results directory under $COUPLEFLOW_BASELINES   (default: results)
# Extra arguments are passed to baselines/deep/run.py.  An existing output JSON is never
# overwritten (delete it to rerun that task).
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
MODELS=(vlaai decaf mm)
DATASETS=(kuleuven dtu esaa uhd255 fuglsang2020 broderick_cp sparrkulee megmasc)
I=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$((I / 40))]}
DS=${DATASETS[$(((I / 5) % 8))]}
FOLD=$((I % 5))
OUT_ROOT=${COUPLEFLOW_BASELINES:-$PWD/results/baselines}/${OUT_SUBDIR:-results}   # coupleflow/paths.py
OUT=$OUT_ROOT/$MODEL/${DS}_fold$FOLD.json
[ -f "$OUT" ] && { echo "$OUT exists"; exit 0; }
echo "=== deep baseline $MODEL $DS fold $FOLD -> $OUT_ROOT :: $* ==="; date
nvidia-smi -L || true
python -u baselines/deep/run.py --model "$MODEL" --dataset "$DS" --fold "$FOLD" \
    --out-root "$OUT_ROOT" "$@"
date

#!/bin/bash
#SBATCH --job-name=cf_cpu
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --exclude=a0105
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#
# Run one Python entry point on a CPU node: `python -u "$@"`.  Submit from the repo root
# (logs/ must exist):
#   sbatch scripts/slurm/cpu.sbatch baselines/linear/linear_baseline.py --dataset kuleuven
#   sbatch --mem=180G scripts/slurm/cpu.sbatch scripts/checks/check.py --dataset megmasc \
#       --name unified_native.pt --subjects 1000           # MEG-MASC (208 sensors) needs 180 GB
#   sbatch scripts/slurm/cpu.sbatch -m coupleflow.preprocessing.stimulus_groups --dataset dtu
# Without Slurm the same body runs in the foreground:  bash scripts/slurm/cpu.sbatch <args>
# Environment file: $COUPLEFLOW_ENV (default scripts/env/osc.sh, the cluster the paper ran on).
# Cluster settings above (account, partition, node exclusion) are those of the paper's runs.
set -eo pipefail
# repo root: the submit directory under Slurm (submit from the repo root), else this file's ../..
ROOT=$(dirname "$0")/../..
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/coupleflow/paths.py" ]; then
    ROOT=$SLURM_SUBMIT_DIR
fi
cd "$ROOT"
source "${COUPLEFLOW_ENV:-scripts/env/osc.sh}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
echo "=== cpu: python $* ==="; date
python -u "$@"
date

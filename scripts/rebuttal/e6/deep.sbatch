#!/bin/bash
#SBATCH --job-name=e6_deep
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --exclude=a0105
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:45:00
#SBATCH --array=0-83
#SBATCH --output=/fs/scratch/PAS2301/alialavi/rebuttal/e6/logs/%x_%A_%a.out
#
# E6 (rebuttal): VLAAI / dilated MM on MAESTRO, 4-way, both timings, both protocols.
#   i = combo x 21 + r;  combo = (vlaai corrected, vlaai release, mm corrected, mm release)[i / 21]
#   r < 5: within-listener content fold r;  r >= 5: LOSO fold r - 5 (16 listeners)
# Submit from the repo root:  sbatch scripts/rebuttal/e6/deep.sbatch
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
source scripts/env/osc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=8
MODELS=(vlaai vlaai mm mm)
TIMINGS=(corrected release corrected release)
I=${SLURM_ARRAY_TASK_ID:-0}
C=$((I / 21)); R=$((I % 21))
if [ "$R" -lt 5 ]; then PROTO=within; FOLD=$R; else PROTO=loso; FOLD=$((R - 5)); fi
echo "=== e6 deep ${MODELS[$C]} ${TIMINGS[$C]} $PROTO fold $FOLD :: $* ==="; date
nvidia-smi -L || true
python -u scripts/rebuttal/e6/deep_maestro.py --model "${MODELS[$C]}" --timing "${TIMINGS[$C]}" \
    --protocol "$PROTO" --fold "$FOLD" "$@"
date

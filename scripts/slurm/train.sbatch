#!/bin/bash
#SBATCH --job-name=cf_train
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --exclude=a0105
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#
# One CoupleFlow run per array index (= fold) with the paper recipe.  Submit from the repo root.
#   sbatch --export=ALL,DS=kuleuven,TAG=unified,SEED=0 scripts/slurm/train.sbatch
# scripts/train/*.sh submit every paper experiment through this file.
# Environment:
#   DS           dataset (required)
#   TAG          run tag -> $COUPLEFLOW_RUNS/<TAG>/<DS>_fold<k>/      (default: unified)
#   SEED         training seed                                          (default: 0)
#   STREAM       bundle file name                                       (default: unified_env64.pt)
#   GROUPS_FILE  content-group file                                     (default: unified_groups.json)
#   BANKS        sb (8- and 28-band sub-bands) | none (envelope only)   (default: sb)
#   SINGLE       1 for single-talker data (same-talker candidates)      (default: 0)
#   SPLIT        stimulus (array index = content fold) | loso (array index = held-out
#                listener, adds data.split_mode=loso)                   (default: stimulus)
#   MODE         train | val (re-evaluate an existing best.pt) | margins (write
#                conf_margins.npz of an existing run, scripts/results/dump_margins.py)
#                                                                       (default: train)
# Extra arguments after the script name are passed to main.py as config overrides.
# Cluster settings above (account, partition, node exclusion) are those of the paper's runs.
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
source scripts/env/osc.sh
export PYTHONHASHSEED=0
FOLD=${SLURM_ARRAY_TASK_ID:-0}
TAG=${TAG:-unified}
STREAM=${STREAM:-unified_env64.pt}
GROUPS_FILE=${GROUPS_FILE:-unified_groups.json}
SEED=${SEED:-0}
SPLITARGS=()
if [ "${SPLIT:-stimulus}" = loso ]; then  # one fold per listener (listener x content held out)
    N_LISTENERS=$(python -c "import json;print(len(json.load(open('$COUPLEFLOW_DATA/$DS/$GROUPS_FILE'))['stats']['groups_per_subject']))")
    if [ "$FOLD" -ge "$N_LISTENERS" ]; then echo "fold $FOLD >= n_listeners $N_LISTENERS"; exit 0; fi
    SPLITARGS=(data.split_mode=loso)
else
    N_GROUPS=$(python -c "import json;print(json.load(open('$COUPLEFLOW_DATA/$DS/$GROUPS_FILE'))['stats']['n_groups'])")
    if [ "$FOLD" -ge "$N_GROUPS" ]; then echo "fold $FOLD >= n_groups $N_GROUPS"; exit 0; fi
fi
if [ "${MODE:-train}" = margins ]; then   # per-window margins of a finished run (App. G)
    RD=$COUPLEFLOW_RUNS/$TAG/${DS}_fold$FOLD
    [ "$RD/conf_margins.npz" -nt "$RD/best.pt" ] && { echo "conf_margins.npz is current"; exit 0; }
    echo "=== margins $TAG $DS fold $FOLD ==="; date
    python -u scripts/results/dump_margins.py "$RD"
    date; exit 0
fi
if [ "${BANKS:-sb}" = none ]; then FB="data.feat_banks=[]"; else FB="data.feat_banks=[sb,sb28]"; fi
if [ "${SINGLE:-0}" = 1 ]; then
    CS="eval.cand_sets=[same_talker]"; VS="eval.val_set=same_talker"
else
    CS="eval.cand_sets=[competing,matched,same_talker]"; VS="eval.val_set=null"
fi
MODEARGS=(--mode train)
if [ "${MODE:-train}" = val ]; then       # re-evaluate an existing best.pt (no training)
    RD=$COUPLEFLOW_RUNS/$TAG/${DS}_fold$FOLD
    [ -f "$RD/results.json" ] && { echo "results.json exists"; exit 0; }
    MODEARGS=(--mode val --checkpoint "$RD/best.pt")
fi
echo "=== train ${MODE:-train} $TAG $DS fold $FOLD seed $SEED stream $STREAM :: $* ==="; date
python -u main.py "${MODEARGS[@]}" --dataset "$DS" fold="$FOLD" tag="$TAG" seed="$SEED" \
    data.stream_name="$STREAM" data.groups_name="$GROUPS_FILE" \
    "$FB" data.glimpse=true optim.epochs=100 optim.patience=20 eval.select_on=coupling \
    "eval.windows=[1.0,2.0,5.0,10.0,20.0,30.0,60.0]" "$CS" "$VS" "${SPLITARGS[@]}" "$@"
date

#!/bin/bash
# Trial split (stories may be shared between training and test): KU Leuven and DTU, seed 0,
# 125-ms lag window; every trial is its own group (unified_groups_trialsplit.json).
#
# Feeds: App. F.2 Table tab:leak (leak.tex/.json via leak_table.py, against unified seed 0 on
#   content-disjoint folds) and the macros \leakMin, \leakMax (make_numbers.py).
#
# Tag: leak_trial (KUL 5 folds, DTU 5 folds).
# Runs: 10; ~3 GPU-hours (paper runs 3.0 h).
# DRY_RUN=1 prints the sbatch commands.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/train/common.sh

for ds in kuleuven dtu; do
    submit "cf_leak_trial_${ds}" "DS=$ds,TAG=leak_trial,SEED=0,GROUPS_FILE=unified_groups_trialsplit.json" \
        0-4 -- $PIN_PRE_LOSO
done
echo "leakage.sh: $N_SUBMIT sbatch command(s)" >&2

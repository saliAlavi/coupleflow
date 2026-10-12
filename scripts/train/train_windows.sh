#!/bin/bash
# Training-window arms: the paper configuration (validation-chosen lag window, per-dataset
# settings) with only the training crop length changed; 3 seeds, 8 datasets.
#
# Feeds: App. G training-length analysis (confidence_appendix.py: conf_trainlen.tex,
#   figures/conf_trainlen.pdf, \confTl* macros, conf_trainlen_text.tex).  The 5-s arm is the
#   main run (final/final_s1/final_s2).  Needs dump_margins.sh afterwards (two-talker sets).
#
# Tags: twin_<arm>_s<seed>, seed 0/1/2:
#   w1   data.train_window_s=1.0          w10  data.train_window_s=10.0
#   w2   data.train_window_s=2.0          mix  data.train_window_s=5.0
#                                              data.train_window_choices=[1.0,2.0,5.0,10.0]
# MEG-MASC runs use the 208-sensor bundle (rerun under the same tags).
#
# Runs: 4 arms x 3 seeds x 35 folds = 420; ~425 GPU-hours (paper runs: w1 168 h, w2 108 h,
#   w10 54 h, mix 95 h; one SparrKULee fold takes up to 8.8 h with 1-s training).
# Needs paper/tables/lag_selection.json (scripts/results/select_lag.py).
# DRY_RUN=1 prints the sbatch commands.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/train/common.sh

arm_override() {
    case $1 in
        w1) echo "data.train_window_s=1.0" ;;
        w2) echo "data.train_window_s=2.0" ;;
        w10) echo "data.train_window_s=10.0" ;;
        mix) echo "data.train_window_s=5.0 data.train_window_choices=[1.0,2.0,5.0,10.0]" ;;
    esac
}
for arm in w1 w2 w10 mix; do
    for s in 0 1 2; do
        for ds in "${ALL_DS[@]}"; do
            lag=$(chosen_lag "$ds")
            submit "cf_twin_${arm}_s${s}_${ds}" "DS=$ds,TAG=twin_${arm}_s$s,SEED=$s$(ds_env "$ds")" \
                "$(range0 "${NFOLD[$ds]}")" $(ds_sbatch "$ds") -- \
                $(ds_overrides "$ds") $(lag_override "$lag") $(arm_override "$arm")
        done
    done
done
echo "train_windows.sh: $N_SUBMIT sbatch command(s)" >&2

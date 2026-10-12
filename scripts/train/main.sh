#!/bin/bash
# Main CoupleFlow runs: content-disjoint folds, both EEG lag windows, 3 seeds, 8 datasets.
#
# Feeds (through the symlink tags final/final_s1/final_s2 of scripts/results/make_final.py):
#   Table tab:main (compare.tex: 1/5/20 s), Table tab:track (tracking.tex), Fig. fig:law
#   (snr_law_final), Figs. fig:kappa/fig:kth, Table tab:flow (flow.tex), Table tab:spatial
#   (CoupleFlow row), App. D (arch.tex, net figures: final/kuleuven_fold0), App. E
#   (loss_terms.tex), App. F tab:app_baselines + windows.tex, App. G (conf_*), numbers*.tex.
# Directly: lag choice (select_lag.py: unified vs esaa_lag0, seed 0); tab:ablation base row
#   (unified, unified_s1, unified_s2 on KUL/DTU/MEG) and its "lag from 0 ms" row (esaa_lag0);
#   tab:app_variants (unified, esaa_lag0 columns); tab:channels and the native-channel kappa
#   macros (unified, via snr_law.py --tag unified); tab:leak content-disjoint column (unified);
#   \esaaLateFive (unified/esaa).
#
# Tags (window: 125 ms = model default, 0 ms = model.lag_offset=0):
#   unified, esaa_lag0               seed 0, all 8 datasets (both windows: the lag choice)
#   unified_s1, unified_s2           seeds 1, 2: datasets whose chosen window is 125 ms
#                                    (KUL, Fuglsang'20, Broderick) + the ablation sets DTU, MEG
#   esaa_lag0_s1, esaa_lag0_s2       seeds 1, 2: datasets whose chosen window is 0 ms
#                                    (DTU, ESAA, KUL-255ch, SparrKULee, MEG-MASC)
# MEG-MASC runs of these tags were trained under tags megnat_125[_s1|_s2] / megnat_0[_s1|_s2]
# (208-sensor rerun) and moved into them; their stored config.yaml keeps the old tag.
#
# Runs: 70 (STAGE=seed0) + 90 (STAGE=seeds) = 160; ~105 GPU-hours (paper runs: seed0 48.3 h,
#   seeds 56.4 h; one SparrKULee fold ~3.5 h, MEG-MASC ~1.1 h, the other datasets 7-23 min).
#
# Order:  STAGE=seed0 bash scripts/train/main.sh      # both windows, seed 0
#         python scripts/results/select_lag.py        # -> paper/tables/lag_selection.json
#         STAGE=seeds bash scripts/train/main.sh      # seeds 1, 2 of the chosen window
#         python scripts/results/make_final.py
# STAGE=all (default) submits both stages at once using the committed lag_selection.json.
# DRY_RUN=1 prints the sbatch commands.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/train/common.sh
STAGE=${STAGE:-all}

run() {   # TAG SEED LAG DS
    local tag=$1 seed=$2 lag=$3 ds=$4 pin=$PIN_PRE_LOSO
    [ "$tag" = unified ] && pin="$pin $PIN_PRE_TAIL"   # seed-0 125-ms runs predate both
    submit "cf_${tag}_${ds}" "DS=$ds,TAG=$tag,SEED=$seed$(ds_env "$ds")" "$(range0 "${NFOLD[$ds]}")" \
        $(ds_sbatch "$ds") -- $(pins "$pin" "$(ds_overrides "$ds")") $(ds_overrides "$ds") \
        $(lag_override "$lag")
}

if [ "$STAGE" = seed0 ] || [ "$STAGE" = all ]; then
    for ds in "${ALL_DS[@]}"; do
        run unified 0 125 "$ds"
        run esaa_lag0 0 0 "$ds"
    done
fi
if [ "$STAGE" = seeds ] || [ "$STAGE" = all ]; then
    for s in 1 2; do
        for ds in "${ALL_DS[@]}"; do
            lag=$(chosen_lag "$ds")
            if [ "$lag" = 125 ] || [[ " ${ABL_DS[*]} " == *" $ds "* ]]; then
                run "unified_s$s" "$s" 125 "$ds"
            fi
            if [ "$lag" = 0 ]; then
                run "esaa_lag0_s$s" "$s" 0 "$ds"
            fi
        done
    done
fi
echo "main.sh: $N_SUBMIT sbatch command(s) (STAGE=$STAGE)" >&2

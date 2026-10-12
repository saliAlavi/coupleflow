#!/bin/bash
# MAESTRO (16 listeners, 4 of 6 talkers attendable): 4-way candidates, 0-ms lag window.
#
# Feeds: Table tab:maestro (maestro_main.tex/.json via maestro_paper.py), the MAESTRO macros
#   \mae* (make_numbers.py) and \maMainNullMin/Max (compare_table.py).
#
# Tags (all: data.cand_idx=[0,1,2,3], eval.cand_sets=[cands_matched,cands,same_talker3,
#   same_talker], eval.windows=[1,2,5,10,20], model.lag_offset=0):
#   maestro_within_lag0{,_s1,_s2}   content-disjoint folds (5), seeds 0/1/2
#   maestro_loso_lag0{,_s1,_s2}     LOSO (16 listeners), seeds 0/1/2
#   maestro_pkg_within{,_s1,_s2}    release timing (unified_pkg.pt + unified_groups_pkg.json)
#   maestro_pkg_loso{,_s1,_s2}      release timing, LOSO
#   maestro_within_pre50            compensated timing, model.lag_offset=-50 (EEG window
#                                   50 samples = 781 ms earlier; maestro_main.json
#                                   "compensated_pre_window", macro \maePreTrial)
# seed 0 of maestro_within_lag0 / maestro_loso_lag0 was evaluated in a second job (MODE=val,
# results_val.json) after an evaluation crash fixed since; a fresh run writes results.json.
#
# Runs: 3 x (5 + 16 + 5 + 16) + 5 = 131; ~18 GPU-hours (paper runs 18.4 h, ~8 min per run).
# DRY_RUN=1 prints the sbatch commands.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/train/common.sh

PKG=",STREAM=unified_pkg.pt,GROUPS_FILE=unified_groups_pkg.json"
for s in 0 1 2; do
    sfx=$([ "$s" = 0 ] && echo "" || echo "_s$s")
    submit "cf_maestro_within_lag0$sfx" "DS=maestro,TAG=maestro_within_lag0$sfx,SEED=$s" \
        "$(range0 "${NFOLD[maestro]}")" -- $MAESTRO_OV $(lag_override 0)
    submit "cf_maestro_loso_lag0$sfx" "DS=maestro,TAG=maestro_loso_lag0$sfx,SEED=$s,SPLIT=loso" \
        "$(range0 "${NLISTENER[maestro]}")" -- $MAESTRO_OV $(lag_override 0)
    submit "cf_maestro_pkg_within$sfx" "DS=maestro,TAG=maestro_pkg_within$sfx,SEED=$s$PKG" \
        "$(range0 "${NFOLD[maestro]}")" -- $MAESTRO_OV $(lag_override 0)
    submit "cf_maestro_pkg_loso$sfx" "DS=maestro,TAG=maestro_pkg_loso$sfx,SEED=$s,SPLIT=loso$PKG" \
        "$(range0 "${NLISTENER[maestro]}")" -- $MAESTRO_OV $(lag_override 0)
done
submit cf_maestro_within_pre50 "DS=maestro,TAG=maestro_within_pre50,SEED=0" \
    "$(range0 "${NFOLD[maestro]}")" -- $MAESTRO_OV model.lag_offset=-50
echo "maestro.sh: $N_SUBMIT sbatch command(s)" >&2

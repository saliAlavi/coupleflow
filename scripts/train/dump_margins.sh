#!/bin/bash
# Per-window test/validation margins (conf_margins.npz) for App. G, written next to best.pt by
# scripts/results/dump_margins.py (train.sbatch MODE=margins; no training).
#
# confidence_appendix.py reads conf_margins.npz of
#   final, final_s1, final_s2      all 8 datasets (calibration, sequential decisions, overlap)
#   twin_<arm>_s<seed>             the six two-talker datasets (sequential decisions per arm)
# Run after main.sh + select_lag.py + make_final.py (final* are symlinks to the chosen runs)
# and after train_windows.sh.  Jobs whose conf_margins.npz is newer than best.pt exit at once.
#
# Jobs: 105 (final*) + 12 x 25 (twin, two-talker folds) = 405 GPU jobs, ~10 min each
#   (paper twin jobs: ~9-12 min after training) -> ~70 GPU-hours.
# DRY_RUN=1 prints the sbatch commands.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/train/common.sh

opts() { [ "$1" = megmasc ] && echo "--time=02:00:00 --mem=120G" || echo "--time=02:00:00"; }
for t in final final_s1 final_s2; do
    for ds in "${ALL_DS[@]}"; do
        submit "cf_margins_${t}_${ds}" "DS=$ds,TAG=$t,MODE=margins" "$(range0 "${NFOLD[$ds]}")" \
            $(opts "$ds") --
    done
done
for arm in w1 w2 w10 mix; do
    for s in 0 1 2; do
        for ds in "${TWO_DS[@]}"; do
            submit "cf_margins_twin_${arm}_s${s}_${ds}" "DS=$ds,TAG=twin_${arm}_s$s,MODE=margins" \
                "$(range0 "${NFOLD[$ds]}")" $(opts "$ds") --
        done
    done
done
echo "dump_margins.sh: $N_SUBMIT sbatch command(s)" >&2

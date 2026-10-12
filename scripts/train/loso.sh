#!/bin/bash
# Leave-one-listener-out (LOSO): one listener AND one content fold held out, seed 0 only.
#
# Feeds: Table tab:main (LOSO, 5 s column), App. F.6 Table tab:loso (loso.tex) and
#   tables/numbers_loso.tex (\loso<DS>Five, \losoDropMin/Max); the PCA-input MEG-MASC runs feed
#   \losoMEGpca in App. C (datasets_appendix.tex).
#
# Tags:
#   loso_s0                         8 datasets, each with its validation-chosen lag window
#                                   (paper/tables/lag_selection.json) and per-dataset settings;
#                                   the source runs were final/<ds>_fold0 (old loso.sbatch).
#                                   MEG-MASC was trained as loso_megnat_0 and moved in.
#   _pca_megmasc/loso_s0            MEG-MASC on the abandoned 64-component PCA bundle
#                                   (unified_env64.pt), 0-ms window; <runs>/_pca_megmasc/loso_s0,
#                                   read by loso_table.py.
# Array index = held-out listener (train.sbatch SPLIT=loso).
#
# Runs: 270 (loso_s0: KUL 16, DTU 18, Fuglsang'20 44, ESAA 17, KUL-255ch 30, Broderick 33,
#   SparrKULee 85, MEG-MASC 27) + 27 (PCA) = 297; ~256 GPU-hours (paper runs: loso_s0 242.6 h,
#   of which SparrKULee 85 x ~2.7 h; PCA 13.6 h).
# Needs paper/tables/lag_selection.json (scripts/results/select_lag.py after main.sh seed 0).
# DRY_RUN=1 prints the sbatch commands.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/train/common.sh

for ds in "${ALL_DS[@]}"; do
    lag=$(chosen_lag "$ds")
    submit "cf_loso_s0_${ds}" "DS=$ds,TAG=loso_s0,SEED=0,SPLIT=loso$(ds_env "$ds")" \
        "$(range0 "${NLISTENER[$ds]}")" $(ds_sbatch "$ds") -- \
        $(ds_overrides "$ds") $(lag_override "$lag")
done
# MEG-MASC, 64-component PCA input (default bundle unified_env64.pt), 0-ms window.
submit cf_loso_pca_megmasc "DS=megmasc,TAG=_pca_megmasc/loso_s0,SEED=0,SPLIT=loso,SINGLE=1" \
    "$(range0 "${NLISTENER[megmasc]}")" -- $(lag_override 0)
echo "loso.sh: $N_SUBMIT sbatch command(s)" >&2

#!/bin/bash
# One-component ablations on KU Leuven, DTU and MEG-MASC (seed 0, 125-ms lag window).
#
# Feeds: App. F.1 Table tab:ablation (ablation.tex via make_tables.py; base row = mean of
#   unified/unified_s1/unified_s2, "lag window from 0 ms" row = esaa_lag0, both from main.sh)
#   and the macros \ablRank*, \ablOtherMax, \lagCtl*, \lagCtlAcc* (make_numbers.py).
#
# Tags (override):
#   abl_noflow     model.use_flow=false losses.flow=0      abl_noadv    losses.adv=0
#   abl_norank     model.rank=0                            abl_nohinge  losses.null_hinge=0
#   abl_nomatch    data.matched=false                      abl_nocon    losses.con=0
#   abl_noglimpse  data.glimpse=false                      abl_selacc   eval.select_metric=acc
#   abl_noside     model.use_side=false losses.side=0      abl_noaux    losses.env=0 losses.col=0
#   abl_before     model.lag_mode=before (lag control)
# MEG-MASC runs use the 208-sensor bundle (rerun under the same tags).
#
# Runs: 11 variants x 14 folds (KUL 4, DTU 5, MEG 5) = 154; ~67 GPU-hours (paper runs 66.6 h).
# DRY_RUN=1 prints the sbatch commands.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/train/common.sh

for v in noflow norank nomatch noglimpse noadv nohinge nocon selacc noside noaux before; do
    for ds in "${ABL_DS[@]}"; do
        submit "cf_abl_${v}_${ds}" "DS=$ds,TAG=abl_$v,SEED=0$(ds_env "$ds")" \
            "$(range0 "${NFOLD[$ds]}")" $(ds_sbatch "$ds") -- \
            $(pins "$PIN_PRE_LOSO" "$(ds_overrides "$ds")") $(ds_overrides "$ds") \
            $(abl_override "$v")
    done
done
echo "ablations.sh: $N_SUBMIT sbatch command(s)" >&2

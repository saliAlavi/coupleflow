#!/bin/bash
# Rebuttal E3: the "simpler model" (huh7 Q4, AC item 4).
#
# CoupleFlow's decision path (listener correction, spatial projection, EEG and audio encoders,
# coupling score) trained with the decoding terms only.  Every other training term is switched
# off; the data features (sub-bands, glimpse, matched candidates), validation selection
# (eval.select_metric=contrib, eval.select_on=coupling) and lag window are those of the paper.
#
#   reb_core   CE + InfoNCE                        (losses.py: L = task + con)
#   reb_ce     CE only (reb_core + losses.con=0)   seed 0
#
# KU Leuven, DTU, MEG-MASC: as scripts/train/ablations.sh (125-ms window, PIN_PRE_LOSO).
#   reb_core, reb_core_s1, reb_core_s2, reb_ce
# MAESTRO, within-listener content-disjoint folds: as scripts/train/maestro_ablations.sh
#   (MAESTRO_OV, model.lag_offset=0).   reb_mcore, reb_mcore_s1, reb_mcore_s2, reb_mce
#
# Runs: 4 x 14 + 4 x 5 = 76.  DRY_RUN=1 prints the sbatch commands.
# Results: python scripts/rebuttal/e3/aggregate.py
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/train/common.sh

CORE="model.use_flow=false losses.flow=0 model.use_side=false losses.side=0 losses.adv=0 \
losses.null_hinge=0 losses.env=0 losses.col=0"
CE="$CORE losses.con=0"

sfx() { [ "$1" = 0 ] && echo "" || echo "_s$1"; }

for s in 0 1 2; do
    for ds in "${ABL_DS[@]}"; do
        submit "cf_reb_core$(sfx "$s")_${ds}" "DS=$ds,TAG=reb_core$(sfx "$s"),SEED=$s$(ds_env "$ds")" \
            "$(range0 "${NFOLD[$ds]}")" $(ds_sbatch "$ds") -- \
            $(pins "$PIN_PRE_LOSO" "$(ds_overrides "$ds")") $(ds_overrides "$ds") $CORE
    done
    submit "cf_reb_mcore$(sfx "$s")" "DS=maestro,TAG=reb_mcore$(sfx "$s"),SEED=$s" \
        "$(range0 "${NFOLD[maestro]}")" -- $MAESTRO_OV $(lag_override 0) $CORE
done
for ds in "${ABL_DS[@]}"; do
    submit "cf_reb_ce_${ds}" "DS=$ds,TAG=reb_ce,SEED=0$(ds_env "$ds")" \
        "$(range0 "${NFOLD[$ds]}")" $(ds_sbatch "$ds") -- \
        $(pins "$PIN_PRE_LOSO" "$(ds_overrides "$ds")") $(ds_overrides "$ds") $CE
done
submit cf_reb_mce "DS=maestro,TAG=reb_mce,SEED=0" "$(range0 "${NFOLD[maestro]}")" -- \
    $MAESTRO_OV $(lag_override 0) $CE
echo "e3/launch.sh: $N_SUBMIT sbatch command(s)" >&2

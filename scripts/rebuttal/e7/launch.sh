#!/bin/bash
# Rebuttal E7 (ocular control): retrain with the frontal channels zeroed (data.drop_frontal=true:
# Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 F7 F8, coupleflow/data.py FRONTAL_IDX; applied at load time, so
# in training, validation and test alike).
#
# Tags:
#   reb_nofrontal                  KU Leuven, DTU; seed 0, settings of scripts/train/ablations.sh
#                                  (125-ms lag window); compare with unified (seed 0)
#   reb_mnofrontal{,_s1,_s2}       MAESTRO within listeners, settings of scripts/train/maestro.sh
#                                  (4-way candidates, 0-ms lag window); compare with
#                                  maestro_within_lag0{,_s1,_s2}
# Runs: 4 + 5 + 3 x 5 = 24.  DRY_RUN=1 prints the sbatch commands.
#   python scripts/rebuttal/e7/aggregate.py    # results vs the full-channel runs
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/train/common.sh

for ds in kuleuven dtu; do
    submit "cf_reb_nofrontal_${ds}" "DS=$ds,TAG=reb_nofrontal,SEED=0$(ds_env "$ds")" \
        "$(range0 "${NFOLD[$ds]}")" $(ds_sbatch "$ds") -- \
        $(pins "$PIN_PRE_LOSO" "$(ds_overrides "$ds")") $(ds_overrides "$ds") data.drop_frontal=true
done
for s in 0 1 2; do
    sfx=$([ "$s" = 0 ] && echo "" || echo "_s$s")
    submit "cf_reb_mnofrontal$sfx" "DS=maestro,TAG=reb_mnofrontal$sfx,SEED=$s" \
        "$(range0 "${NFOLD[maestro]}")" -- $MAESTRO_OV $(lag_override 0) data.drop_frontal=true
done
echo "e7/launch.sh: $N_SUBMIT sbatch command(s)" >&2

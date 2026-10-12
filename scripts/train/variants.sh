#!/bin/bash
# CoupleFlow variants (seed 0, content-disjoint folds, 125-ms lag window).
#
# Feeds: App. F Table tab:app_variants (app_variants.tex via appendix_tables.py; base columns
#   unified / esaa_lag0 come from main.sh); unified_native also feeds App. C Table tab:channels
#   (channels.tex via make_tables.py) and the native-channel kappa macros \kN*, \tN*
#   (snr_law.py --tag unified_native -> figures/snr_law_unified_native.json -> make_numbers.py).
#
# Tags:
#   unified_envonly  broadband envelope only (BANKS=none): KUL, DTU, ESAA, KUL-255ch, SparrKULee,
#                    MEG-MASC (Fuglsang'20 and Broderick are envelope-only already: "n/a")
#   unified_native   native channels (unified_native.pt): Broderick (128), KUL-255ch (253)
#   esaa_rank8       listener correction of rank 8 (model.rank=8): KUL, DTU, ESAA
#   esaa_sft         per-listener fine-tuning after pooled training (subject_ft.enabled=true):
#                    KUL, DTU, ESAA
# MEG-MASC on its 208 sensors is not a variant: every MEG-MASC run of every tag uses it
# (the former megnat_* tags were moved into unified*, esaa_lag0* and loso_s0).
#
# Runs: 26 + 7 + 14 + 14 = 61; ~22 GPU-hours (paper runs 21.6 h).
# DRY_RUN=1 prints the sbatch commands.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/train/common.sh

run() {   # TAG DS EXTRA_EXPORTS PIN [OVERRIDE ...]
    local tag=$1 ds=$2 ex=$3 pin=$4
    shift 4
    submit "cf_${tag}_${ds}" "DS=$ds,TAG=$tag,SEED=0$(ds_env "$ds")$ex" "$(range0 "${NFOLD[$ds]}")" \
        $(ds_sbatch "$ds") -- $(pins "$pin" "$(ds_overrides "$ds")") $(ds_overrides "$ds") "$@"
}
for ds in kuleuven dtu esaa uhd255 sparrkulee megmasc; do
    run unified_envonly "$ds" ",BANKS=none" "$PIN_PRE_LOSO $PIN_PRE_TAIL"
done
for ds in broderick_cp uhd255; do
    run unified_native "$ds" ",STREAM=unified_native.pt" "$PIN_PRE_LOSO $PIN_PRE_TAIL"
done
for ds in kuleuven dtu esaa; do
    run esaa_rank8 "$ds" "" "$PIN_PRE_LOSO" model.rank=8
done
for ds in kuleuven dtu esaa; do
    run esaa_sft "$ds" "" "$PIN_PRE_LOSO" subject_ft.enabled=true
done
echo "variants.sh: $N_SUBMIT sbatch command(s)" >&2

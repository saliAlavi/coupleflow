#!/bin/bash
# One-component ablations on MAESTRO (seed 0, content-disjoint folds, 0-ms lag window).
#
# Feeds: App. F.3 Table tab:mabl (maestro_ablation.tex/.json via maestro_ablation.py; full-model
#   row = maestro_within_lag0 from maestro.sh) and the macros \maAbl*, \maBefore*, \maSafeMax*,
#   \maNullMin/Max, \maUnmatchedGap (make_numbers.py).
#
# Tags: mabl_<v> for v in noflow norank nomatch noglimpse nobanks noadv nohinge nocon selacc
#   noside noaux before (overrides: scripts/train/common.sh abl_override), each on top of the
#   MAESTRO settings of maestro.sh (4-way candidates, model.lag_offset=0).
#
# Runs: 12 x 5 folds = 60; ~10 GPU-hours (paper runs 10.4 h).
# DRY_RUN=1 prints the sbatch commands.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/train/common.sh

for v in noflow norank nomatch noglimpse nobanks noadv nohinge nocon selacc noside noaux before; do
    submit "cf_mabl_$v" "DS=maestro,TAG=mabl_$v,SEED=0" "$(range0 "${NFOLD[maestro]}")" -- \
        $MAESTRO_OV $(lag_override 0) $(abl_override "$v")
done
echo "maestro_ablations.sh: $N_SUBMIT sbatch command(s)" >&2

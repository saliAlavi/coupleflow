#!/bin/bash
# Rebuild every table, macro and figure of the paper from the shipped results, then the PDF.
#
#   bash scripts/build_paper.sh              # tables + figures + abstract + PDF
#   NO_PDF=1 bash scripts/build_paper.sh     # skip LaTeX
#   WITH_DATA=1 bash scripts/build_paper.sh  # also redraw App. D data thumbnails (needs the
#                                            # KU Leuven bundle in $COUPLEFLOW_DATA/kuleuven/)
#
# Inputs (all in the repository unless the COUPLEFLOW_* variables point elsewhere):
#   runs/<tag>/<ds>_fold<k>/results*.json, best.pt, config.yaml   CoupleFlow runs
#   runs/<tag>/<ds>_fold<k>/conf_margins.npz                      App. G only; fetch with
#                                                                 scripts/fetch_release_assets.sh
#   data/<ds>/*.json, linear_margins.npz                          baselines, checks, content groups
#   results/baselines/, results/earlier/, results/offset_scan/    per-fold baseline outputs etc.
# Outputs: paper/tables/*, figures/*, paper/figures/*, paper/abstract.md, paper/main.pdf.
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg PYTHONHASHSEED=0 SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1791331200}  # fixed PDF dates: rebuilds are byte-identical
run() { echo "== $*"; python "$@"; }

# pooled baseline results -> data/<ds>/baseline_*.json
run baselines/deep/aggregate.py
# lag window per dataset (validation only) -> runs/final{,_s1,_s2} symlinks
run scripts/results/select_lag.py
run scripts/results/make_final.py
# tables (make_tables.py first: later scripts read tables/tables.json)
run scripts/tables/make_tables.py --out paper/tables
run scripts/figures/snr_law.py --tag final
run scripts/figures/kappa_fig.py
run scripts/figures/kappa_linear.py
run scripts/figures/kappa_theory.py
run scripts/figures/kappa_theory_fig.py
run scripts/tables/leak_table.py
run scripts/tables/spatial_table.py --cf-tag final
run scripts/tables/appendix_tables.py
run scripts/tables/maestro_paper.py
run scripts/tables/maestro_ablation.py
run scripts/figures/fig_architecture.py
run scripts/tables/arch_table.py
run scripts/tables/groups_table.py
run scripts/tables/earlier_split_table.py
run scripts/tables/loso_table.py
if ls runs/final/*/conf_margins.npz >/dev/null 2>&1; then
    run scripts/tables/confidence_appendix.py      # App. G (~15 min on CPU)
else
    echo "== skip confidence_appendix.py: no conf_margins.npz (bash scripts/fetch_release_assets.sh)"
fi
if [ "${WITH_DATA:-0}" = 1 ]; then
    run scripts/figures/fig_net_thumbs.py          # App. D thumbnails + tables/net_values.tex
else
    echo "== skip fig_net_thumbs.py (needs the KU Leuven bundle; WITH_DATA=1)"
fi
run scripts/figures/fig_net_encoders.py
run scripts/tables/compare_table.py
run scripts/tables/loss_table.py
run scripts/tables/make_numbers.py                 # last among the table scripts
run scripts/paper/sync_paper_figs.py
run scripts/paper/make_abstract.py
[ "${NO_PDF:-0}" = 1 ] || bash scripts/paper/build_pdf.sh

#!/usr/bin/env bash
# Linear backward-model (mTRF-style ridge) baselines on CoupleFlow's folds, windows and inputs.
# Every script reads coupleflow/preprocessing/run_config.py (bundle, validation settings).
#
# Produces, under $COUPLEFLOW_DATA/<dataset>/:
#   linear_baseline.json            per-listener decoder, content folds, windows 1-60 s, competing
#                                   and same-talker rules, trial accuracy, 5-s accuracy per listener.
#                                   Two-talker sets as is; SparrKULee and MEG-MASC with --single.
#                                   Read by scripts/tables/compare_table.py (main Table 1
#                                   compare.tex, numbers_compare.tex), make_tables.py (main.tex),
#                                   appendix_tables.py (app_baselines.tex, App. F), leak_table.py,
#                                   scripts/figures/kappa_linear.py (figures/snr_law_linear.json)
#                                   and baselines/deep/aggregate.py (RESULTS.md).
#   linear_baseline_trialsplit.json kuleuven, dtu: the same on the leaky trial split
#                                   (unified_groups_trialsplit.json) -> leak_table.py (tables/leak.tex).
#   linear_loso.json                two-talker sets: one decoder pooled over the training listeners
#                                   of each leave-one-listener-out fold, 5-s windows ->
#                                   compare_table.py (Table 1, LOSO column), loso_table.py
#                                   (tables/loso.tex, App. F.6).
#   linear_margins.npz              two-talker sets: per-window margins r(att) - r(comp), 1/2/5-s
#                                   windows, test and out-of-sample validation ->
#                                   scripts/tables/confidence_appendix.py (App. G tables, figures).
#   (STEPS=persub) linear_baseline_persub.json  kuleuven, esaa: an earlier per-listener export of
#                                   the same run, read by no table (linear_baseline.json now carries
#                                   the per-listener 5-s accuracies); only to recreate that file.
# Needs: scripts/data/build_bundles.sh and scripts/data/stimulus_groups.sh.
#
# Compute (CPU only, scripts/slurm/cpu.sbatch, 16 cores): linear_baseline 5-30 min per dataset
# (SparrKULee ~70 min; MEG-MASC ~40 min and 180 GB), linear_loso 15 min-2.2 h (Fuglsang'20 the
# longest), linear_margins 10-35 min; about 12 CPU-node-hours for all 22 jobs.
#
# Usage:
#   bash scripts/baselines/linear.sh                     # submit everything
#   DRY_RUN=1 bash scripts/baselines/linear.sh           # print the commands only
#   STEPS="baseline loso" DATASETS="kuleuven dtu" bash scripts/baselines/linear.sh
#   MODE=local bash scripts/baselines/linear.sh          # no Slurm: run here in turn
# Environment: DRY_RUN, MODE (slurm|local), STEPS ("baseline trialsplit loso margins" default;
# also "persub"), DATASETS, DEPEND (extra Slurm dependency), COUPLEFLOW_ENV.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- launcher helpers (same block in scripts/data/*.sh and scripts/baselines/*.sh) -------------
DRY_RUN=${DRY_RUN:-0}   # 1: print every command, run and submit nothing
MODE=${MODE:-slurm}     # slurm: submit jobs with sbatch | local: run each job here, one after another
DEPEND=${DEPEND:-}      # extra Slurm dependency for every job, e.g. afterok:1234567
LAST=""                 # id of the last submitted job (its name in angle brackets in a dry run)

say() {  # print a command as one copy-pasteable shell line
    local s="" a
    for a in "$@"; do
        if [[ $a =~ ^[][A-Za-z0-9_./:=,@%+{}-]+$ ]]; then s+="$a "; else s+="'${a//\'/\'\\\'\'}' "; fi
    done
    echo "${s% }"
}

expand_array() {  # "0-3,5,7-8" -> 0 1 2 3 5 7 8  (a Slurm throttle "%N" is ignored)
    local p a=${1%%\%*}
    for p in ${a//,/ }; do
        if [[ $p == *-* ]]; then seq "${p%-*}" "${p#*-}"; else echo "$p"; fi
    done
}

# job NAME SCRIPT [sbatch options] -- [arguments]
#   slurm: sbatch --parsable -J NAME [options] SCRIPT [arguments]   (LAST = job id)
#   local: bash SCRIPT [arguments] with the --export variables set, once per --array index
job() {
    local name=$1 script=$2 e i
    shift 2
    local opts=() envs=() deps=() array=""
    while [ $# -gt 0 ] && [ "$1" != -- ]; do
        case $1 in
            --export=*) e=${1#--export=}; e=${e#ALL}; e=${e#,}
                        [ -n "$e" ] && IFS=, read -ra envs <<<"$e"; opts+=("$1") ;;
            --array=*) array=${1#--array=}; opts+=("$1") ;;
            --dependency=*) deps+=("${1#--dependency=}") ;;
            *) opts+=("$1") ;;
        esac
        shift
    done
    [ $# -gt 0 ] && shift
    [ -n "$DEPEND" ] && deps+=("$DEPEND")
    if [ "$MODE" = local ]; then
        local ids=("") pre=()
        if [ -n "$array" ]; then
            ids=($(expand_array "$array"))
            pre=("SLURM_ARRAY_TASK_ID=<each of $array>")
        fi
        if [ ${#envs[@]} -gt 0 ] || [ -n "$array" ]; then
            pre=(env ${envs[@]+"${envs[@]}"} ${pre[@]+"${pre[@]}"})
        fi
        if [ "$DRY_RUN" = 1 ]; then
            say ${pre[@]+"${pre[@]}"} bash "$script" "$@"
        else
            for i in "${ids[@]}"; do
                echo "### $name${i:+ [$i]}"
                env ${envs[@]+"${envs[@]}"} ${i:+SLURM_ARRAY_TASK_ID=$i} bash "$script" "$@"
            done
        fi
        LAST=""
        return
    fi
    local dep=()
    [ ${#deps[@]} -gt 0 ] && dep=("--dependency=$(IFS=,; echo "${deps[*]}")")
    local cmd=(sbatch --parsable -J "$name" ${dep[@]+"${dep[@]}"} ${opts[@]+"${opts[@]}"} "$script" "$@")
    if [ "$DRY_RUN" = 1 ]; then
        say "${cmd[@]}"
        LAST="<$name>"
    else
        LAST=$("${cmd[@]}")
        LAST=${LAST%%;*}
        echo "submitted $name: job $LAST"
    fi
}

act() {  # act CMD...: a launcher-side command (printed in a dry run)
    if [ "$DRY_RUN" = 1 ]; then say "$@"; else "$@"; fi
}

if [ "$DRY_RUN" != 1 ] && [ "$MODE" = slurm ]; then
    command -v sbatch >/dev/null || { echo "sbatch not found: use MODE=local" >&2; exit 1; }
    mkdir -p logs
fi
# -------------------------------------------------------------------------------------------------

DATASETS=${DATASETS:-"kuleuven dtu fuglsang2020 esaa uhd255 broderick_cp sparrkulee megmasc"}
STEPS=${STEPS:-"baseline trialsplit loso margins"}
C=scripts/slurm/cpu.sbatch
SINGLE=" sparrkulee megmasc "            # single-talker: same-talker rule only, no LOSO / margins
has() { [[ " $1 " == *" $2 "* ]]; }

for ds in $DATASETS; do
    single=0
    if has "$SINGLE" "$ds"; then single=1; fi
    if has "$STEPS" baseline; then
        flag=() mem=()
        if [ $single = 1 ]; then flag=(--single); fi
        if [ "$ds" = megmasc ]; then mem=(--mem=180G); fi   # 208-sensor bundle
        job "linear_$ds" $C ${mem[@]+"${mem[@]}"} --time=08:00:00 -- \
            baselines/linear/linear_baseline.py --dataset "$ds" ${flag[@]+"${flag[@]}"}
    fi
    if has "$STEPS" trialsplit && has "kuleuven dtu" "$ds"; then
        job "linear_trialsplit_$ds" $C --time=04:00:00 -- \
            baselines/linear/linear_baseline.py --dataset "$ds" \
            --groups-name unified_groups_trialsplit.json --out-name linear_baseline_trialsplit.json
    fi
    if has "$STEPS" persub && has "kuleuven esaa" "$ds"; then
        job "linear_persub_$ds" $C --time=04:00:00 -- \
            baselines/linear/linear_baseline.py --dataset "$ds" --out-name linear_baseline_persub.json
    fi
    if [ $single = 1 ]; then continue; fi
    if has "$STEPS" loso; then
        job "linear_loso_$ds" $C --time=08:00:00 -- baselines/linear/linear_loso.py --dataset "$ds"
    fi
    if has "$STEPS" margins; then
        job "linear_margins_$ds" $C --time=06:00:00 -- \
            baselines/linear/linear_margins.py --dataset "$ds"
    fi
done

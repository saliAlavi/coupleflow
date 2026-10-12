#!/usr/bin/env bash
# Deep AAD baselines -- VLAAI, DECAF and the dilated match-mismatch (MM) model -- one
# subject-independent model per (model, dataset, content fold), plus the trial-split leakage runs.
#
# Produces:
#   $COUPLEFLOW_BASELINES/results/{vlaai,decaf,mm}/<ds>_fold<k>.json        (STEPS=main)
#       scripts/slurm/baseline_deep.sbatch array, index = model x 40 + dataset x 5 + fold over
#       (vlaai decaf mm) x (kuleuven dtu esaa uhd255 fuglsang2020 broderick_cp sparrkulee megmasc);
#       only existing folds are submitted (KU Leuven 4, KUL-255ch 2, Fuglsang'20 4, others 5), e.g.
#       MEG-MASC = 35-39 (VLAAI), 75-79 (DECAF), 115-119 (MM), on its 208-sensor bundle.
#   $COUPLEFLOW_DATA/<ds>/baseline_{vlaai,decaf,decaf_eeg,mm}.json and baselines/deep/RESULTS.md
#       (STEPS=aggregate, baselines/deep/aggregate.py after the main array): folds pooled.
#       Read by scripts/tables/compare_table.py (main Table 1), make_tables.py (main.tex),
#       appendix_tables.py (app_baselines.tex, App. F; also DECAF) and leak_table.py.
#   $COUPLEFLOW_BASELINES/results_leak/{vlaai,mm}/{kuleuven,dtu}_fold<k>.json   (STEPS=leak)
#       the same models on the leaky trial split (unified_groups_trialsplit.json, 5 folds) ->
#       scripts/tables/leak_table.py (tables/leak.tex, App. F.2).
# Needs: bundles, unified_groups.json and (leak) unified_groups_trialsplit.json
# (scripts/data/{build_bundles,stimulus_groups}.sh).
#
# Compute (GPU, one A100 per task, 110 GB, <= 2 h 45 min): 101 main tasks of 1-20 min (~10
# GPU-hours; SparrKULee and MEG-MASC the longest) + 20 leakage tasks (~1 GPU-hour); the
# aggregation is a few seconds on CPU.  Existing per-fold JSONs are kept (delete to rerun).
#
# Usage:
#   bash scripts/baselines/deep.sh                       # main array + leakage array + aggregate
#   DRY_RUN=1 bash scripts/baselines/deep.sh             # print the commands (and the index table)
#   MODELS="vlaai" DATASETS="megmasc" bash scripts/baselines/deep.sh    # e.g. indices 35-39 only
#   STEPS=aggregate bash scripts/baselines/deep.sh       # re-pool existing results only
#   MODE=local bash scripts/baselines/deep.sh            # no Slurm: every task here, in turn
# Environment: DRY_RUN, MODE (slurm|local), STEPS ("main leak aggregate"), MODELS, DATASETS,
# DEPEND (extra Slurm dependency), COUPLEFLOW_ENV (default scripts/env/osc.sh).
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

ALL_MODELS=(vlaai decaf mm)                 # index order of scripts/slurm/baseline_deep.sbatch
ALL_DATASETS=(kuleuven dtu esaa uhd255 fuglsang2020 broderick_cp sparrkulee megmasc)
MODELS=${MODELS:-"${ALL_MODELS[*]}"}
DATASETS=${DATASETS:-"${ALL_DATASETS[*]}"}
STEPS=${STEPS:-"main leak aggregate"}
has() { [[ " $1 " == *" $2 "* ]]; }
pick() {  # the words of $1 that are also in $2, space-separated
    local x o=""
    for x in $1; do if has "$2" "$x"; then o+="$x "; fi; done
    echo "${o% }"
}
nfolds() {  # content folds per dataset (= baselines/deep/aggregate.py N_FOLDS)
    case $1 in kuleuven|fuglsang2020) echo 4 ;; uhd255) echo 2 ;; *) echo 5 ;; esac
}

spec() {  # spec "<models>" "<datasets>" [fixed fold count] -> array spec of model x dataset x fold
    local m d s="" n
    for m in "${!ALL_MODELS[@]}"; do
        has "$1" "${ALL_MODELS[$m]}" || continue
        for d in "${!ALL_DATASETS[@]}"; do
            has "$2" "${ALL_DATASETS[$d]}" || continue
            n=${3:-$(nfolds "${ALL_DATASETS[$d]}")}
            s+="$((m * 40 + d * 5))-$((m * 40 + d * 5 + n - 1)),"
        done
    done
    echo "${s%,}"
}

table() {  # print what each range of an array spec runs (dry run)
    local r a b
    for r in ${1//,/ }; do
        a=${r%-*}; b=${r#*-}
        printf '#   %-8s %-5s %-12s folds %d-%d\n' "$r" "${ALL_MODELS[$((a / 40))]}" \
            "${ALL_DATASETS[$(((a / 5) % 8))]}" $((a % 5)) $((b % 5))
    done
}

if has "$STEPS" main; then
    main=$(spec "$MODELS" "$DATASETS")
    if [ -n "$main" ]; then
        if [ "$DRY_RUN" = 1 ]; then echo "# main array ($main):"; table "$main"; fi
        job bl_deep scripts/slurm/baseline_deep.sbatch --array="$main" --
    fi
fi
main_job=$LAST

if has "$STEPS" leak; then
    # trial split: one group per trial, so KU Leuven has 5 folds here too
    leak=$(spec "$(pick "vlaai mm" "$MODELS")" "$(pick "kuleuven dtu" "$DATASETS")" 5)
    if [ -n "$leak" ]; then
        if [ "$DRY_RUN" = 1 ]; then echo "# leakage array ($leak):"; table "$leak"; fi
        job bl_deep_leak scripts/slurm/baseline_deep.sbatch --array="$leak" \
            --export=ALL,OUT_SUBDIR=results_leak -- --groups-name unified_groups_trialsplit.json
    fi
fi

if has "$STEPS" aggregate; then
    dep=()
    if [ -n "$main_job" ]; then dep=(--dependency=afterany:$main_job); fi
    job bl_deep_aggregate scripts/slurm/cpu.sbatch ${dep[@]+"${dep[@]}"} --cpus-per-task=2 \
        --mem=16G --time=00:30:00 -- -m baselines.deep.aggregate
fi

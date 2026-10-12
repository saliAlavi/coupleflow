#!/usr/bin/env bash
# Published spatial AAD classifiers -- DARNet, DBPNet, ListenNet (the authors' model code) -- on
# the unified bundles under two protocols: A = CoupleFlow's content folds (subject-independent and
# per-listener), B = the papers' within-trial split (last 10% of every trial is the test set).
#
# Produces:
#   $COUPLEFLOW_THIRDPARTY/{darnet,DBPNet,ListenNet}   (STEPS=fetch, baselines/spatial/fetch_thirdparty.sh)
#   $COUPLEFLOW_BASELINES/spatial_results/<model>/<ds>_<protocol>_<w>s_fold<k>.json   (STEPS=tasks)
#       scripts/slurm/baseline_spatial.sbatch, array 0-404 = the 405 rows of
#       `python baselines/spatial/run.py --list`: 3 models x 5 two-talker datasets (kuleuven dtu
#       esaa fuglsang2020 uhd255) x 3 windows (1, 2, 5 s) x {A_indep, A_dep per fold; B once}.
#   $COUPLEFLOW_DATA/<ds>/spatial_{darnet,dbpnet,listennet}.json   (STEPS=aggregate,
#       baselines/spatial/aggregate.py; its Markdown tables land in the job log).
#       Read by scripts/tables/spatial_table.py (tables/spatial.tex, Sec. 5 side classifiers) and
#       scripts/tables/appendix_tables.py (tables/app_spatial.tex, App. F).
# Needs: bundles and unified_groups.json (scripts/data/{build_bundles,stimulus_groups}.sh).
#
# Compute (GPU, one A100 per task, 110 GB, <= 8 h): 405 tasks of 3-52 min, ~77 GPU-hours in all;
# aggregation is seconds on CPU.  run.py skips tasks whose output exists (delete to rerun).
#
# Usage:
#   bash scripts/baselines/spatial.sh                    # fetch, submit the array, then aggregate
#   DRY_RUN=1 bash scripts/baselines/spatial.sh          # print the commands only
#   TASKS=0-44 bash scripts/baselines/spatial.sh         # a subset of the task table
#   STEPS=aggregate bash scripts/baselines/spatial.sh    # re-pool existing results only
#   MODE=local bash scripts/baselines/spatial.sh         # no Slurm: every task here, in turn
# Environment: DRY_RUN, MODE (slurm|local), STEPS ("fetch tasks aggregate"), TASKS (default
# 0-404), DEPEND (extra Slurm dependency), COUPLEFLOW_ENV (default scripts/env/osc.sh; also read
# here for COUPLEFLOW_THIRDPARTY).
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

STEPS=${STEPS:-"fetch tasks aggregate"}
TASKS=${TASKS:-0-404}
has() { [[ " $1 " == *" $2 "* ]]; }

if has "$STEPS" fetch; then
    # same environment as the jobs, so the clones land where run.py looks (COUPLEFLOW_THIRDPARTY)
    env_file=${COUPLEFLOW_ENV:-scripts/env/osc.sh}
    [ -f "$env_file" ] || { echo "environment file $env_file not found" >&2; exit 1; }
    (
        set +eu
        source "$env_file" >/dev/null
        set -eu
        if [ "$DRY_RUN" = 1 ]; then
            echo "# fetch_thirdparty.sh -> ${COUPLEFLOW_THIRDPARTY:-baselines/spatial/thirdparty}:"
        fi
        DRY_RUN=$DRY_RUN bash baselines/spatial/fetch_thirdparty.sh
    )
fi

if has "$STEPS" tasks; then
    job bl_spatial scripts/slurm/baseline_spatial.sbatch --array="$TASKS" --
fi
tasks_job=$LAST

if has "$STEPS" aggregate; then
    dep=()
    if [ -n "$tasks_job" ]; then dep=(--dependency=afterany:$tasks_job); fi
    job bl_spatial_aggregate scripts/slurm/cpu.sbatch ${dep[@]+"${dep[@]}"} --cpus-per-task=2 \
        --mem=16G --time=00:30:00 -- baselines/spatial/aggregate.py
fi

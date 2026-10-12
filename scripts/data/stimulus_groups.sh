#!/usr/bin/env bash
# Recover the content groups (trials sharing stimulus audio) of every paper dataset, and the
# derived group files used by the leakage control and the MAESTRO timing runs.
#
# Produces, under $COUPLEFLOW_DATA/<dataset>/:
#   unified_groups.json             content groups from the audio envelopes of unified_env64.pt
#                                   (coupleflow/preprocessing/stimulus_groups.py: 10-s chunks matched
#                                   at 8 Hz, r > 0.9, confirmed at 64 Hz, r > 0.95).  Folds are unions
#                                   of groups: read by every training run, baseline and check.
#   unified_groups_v2.json          copy of unified_groups.json.  The 10-01 audit re-derived the
#                                   groups with the current algorithm under this name; it reproduces
#                                   them exactly (Fuglsang'20: same groups, min. verified r 0.977
#                                   instead of 0.966).  Read by scripts/tables/groups_table.py
#                                   (tables/groups.tex, numbers_groups.tex, App. C) and as the
#                                   reference of scripts/checks/check_fold_leak.py.
#   unified_groups_trialsplit.json  kuleuven, dtu: one group per trial (the leaky "trial split",
#                                   stories shared across folds).  Read by the leakage runs of
#                                   scripts/baselines/{linear,deep}.sh and the CoupleFlow
#                                   leak_trial runs -> scripts/tables/leak_table.py (tables/leak.tex,
#                                   App. F.2).
#   unified_groups_pkg.json         maestro: copy of unified_groups.json for the maestro_pkg_* runs
#                                   (release timing), so both timings use identical folds.
# Needs: scripts/data/build_bundles.sh.  Next: scripts/data/checks.sh.
#
# Compute (CPU, scripts/slurm/cpu.sbatch, 96 GB): 9 jobs of 1-4 min (SparrKULee needs
# --block 512 to fit in 96 GB) + one small job for the derived files.
#
# Usage:
#   bash scripts/data/stimulus_groups.sh                  # submit
#   DRY_RUN=1 bash scripts/data/stimulus_groups.sh        # print the commands only
#   DATASETS="kuleuven dtu" bash scripts/data/stimulus_groups.sh
#   MODE=local bash scripts/data/stimulus_groups.sh       # no Slurm: run here in turn
# Environment: DRY_RUN, MODE (slurm|local), DATASETS, DEPEND (extra Slurm dependency, e.g. the
# build jobs), COUPLEFLOW_ENV (environment file the jobs source; default scripts/env/osc.sh).
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

DATASETS=${DATASETS:-"kuleuven dtu fuglsang2020 esaa uhd255 broderick_cp sparrkulee megmasc maestro"}
C=scripts/slurm/cpu.sbatch

# Derived files: v2 copy (all), one group per trial (kuleuven, dtu), MAESTRO package-timing copy.
DERIVE_PY='import json, shutil, sys
from coupleflow import paths
for ds in sys.argv[1:]:
    d = paths.DATA / ds
    src = d / "unified_groups.json"
    shutil.copyfile(src, d / "unified_groups_v2.json")
    if ds == "maestro":
        shutil.copyfile(src, d / "unified_groups_pkg.json")
    if ds in ("kuleuven", "dtu"):
        g = json.loads(src.read_text())["groups"]
        out = {"dataset": ds, "groups": {k: i for i, k in enumerate(g)},
               "stats": {"n_groups": len(g), "note": "LEAKY control: one group per trial "
                                                     "(content shared across folds)"}}
        (d / "unified_groups_trialsplit.json").write_text(json.dumps(out, indent=1))
    print("derived group files written for", ds)'

ids=()
for ds in $DATASETS; do
    blk=()
    if [ "$ds" = sparrkulee ]; then blk=(--block 512); fi   # 4096 (default) needs > 96 GB
    job "groups_$ds" $C --time=01:00:00 -- -m coupleflow.preprocessing.stimulus_groups \
        --dataset "$ds" --stream-name unified_env64.pt --out-name unified_groups.json \
        ${blk[@]+"${blk[@]}"}
    if [ -n "$LAST" ]; then ids+=("$LAST"); fi
done
dep=()
if [ ${#ids[@]} -gt 0 ]; then dep=("--dependency=afterok:$(IFS=:; echo "${ids[*]}")"); fi
# shellcheck disable=SC2086
job groups_derive $C ${dep[@]+"${dep[@]}"} --cpus-per-task=1 --mem=8G --time=00:15:00 -- \
    -c "$DERIVE_PY" $DATASETS

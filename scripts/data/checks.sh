#!/usr/bin/env bash
# Data checks: per-listener envelope tracking of every bundle, and the split audit.
#
# Produces:
#   $COUPLEFLOW_DATA/<ds>/unified_env64.check.json   (scripts/checks/check.py) per-listener ridge
#   $COUPLEFLOW_DATA/<ds>/unified_native.check.json  backward model, leave-trials-out: r with the
#                                     attended / competing / time-shifted envelope, 30-s accuracy.
#                                     Read through run_config.check_name() (MEG-MASC: the 208-sensor
#                                     unified_native.check.json) by scripts/tables/compare_table.py
#                                     (main Table 2, tables/tracking.tex) and
#                                     scripts/figures/kappa_theory.py (tables/kappa_theory.tex,
#                                     figures/kappa_theory.pdf).  The other checks (MEG-MASC PCA,
#                                     KUL-255ch / Broderick native channels) are sanity checks only.
#   paper/tables/split_audit.jsonl    (scripts/checks/check_splits.py) one line per dataset and mode
#                                     (content folds, LOSO): trial-, content-, stimulus-name- and
#                                     sample-disjointness, all with run_config.py's settings.  The
#                                     record behind the protocol statements of Sec. 2 / App. C; no
#                                     table script reads it.  Emptied first when all datasets are
#                                     audited (HANDOFF advice); the jobs run one after another so the
#                                     lines come in a fixed order.
#   results/offset_scan/<ds>_unified_env64.pt.json  (scripts/checks/offset_scan.py) held-out
#                                     backward-model r at global EEG-audio offsets of -4..+4 s; a
#                                     correctly aligned bundle peaks at 0.  ESAA's scan feeds a macro
#                                     of scripts/tables/compare_table.py (App. C onset estimate).
#   (STEPS=leak) scripts/checks/check_fold_leak.py: test trials whose unified_groups_v2.json group
#                                     also has training data, per fold; log output only.
# Needs: scripts/data/build_bundles.sh and scripts/data/stimulus_groups.sh.
#
# Compute (CPU, scripts/slurm/cpu.sbatch): tracking 1-7 min per bundle at 96 GB, MEG-MASC
# 208 sensors ~25 min at 180 GB; audit < 1 min per job (bundle loading), 18 jobs at 180 GB.
#
# Usage:
#   bash scripts/data/checks.sh                          # tracking checks + split audit
#   DRY_RUN=1 bash scripts/data/checks.sh                # print the commands only
#   STEPS="audit" DATASETS="kuleuven" bash scripts/data/checks.sh   # appends, does not empty
#   MODE=local bash scripts/data/checks.sh               # no Slurm: run here in turn
# Environment: DRY_RUN, MODE (slurm|local), STEPS ("track audit offset" default; also "leak"), DATASETS,
# RESET_AUDIT (1: empty split_audit.jsonl first; default 1 only when all 9 datasets are audited),
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

ALL="kuleuven dtu fuglsang2020 esaa uhd255 broderick_cp sparrkulee megmasc maestro"
DATASETS=${DATASETS:-$ALL}
STEPS=${STEPS:-"track audit offset"}
C=scripts/slurm/cpu.sbatch
AUDIT=paper/tables/split_audit.jsonl
has() { [[ " $1 " == *" $2 "* ]]; }

if has "$STEPS" track; then
    for ds in $DATASETS; do
        # --subjects 1000 = every listener (the largest dataset has 85)
        job "check_$ds" $C --time=04:00:00 -- \
            scripts/checks/check.py --dataset "$ds" --subjects 1000
        case $ds in
        megmasc)            # the 208-sensor bundle every MEG-MASC model reads
            job check_megmasc_native $C --mem=180G --time=08:00:00 -- \
                scripts/checks/check.py --dataset megmasc --name unified_native.pt --subjects 1000 ;;
        uhd255|broderick_cp) # native-channel variants (253 / 128 channels)
            job "check_${ds}_native" $C --time=04:00:00 -- \
                scripts/checks/check.py --dataset "$ds" --name unified_native.pt --subjects 1000 ;;
        esac
    done
fi

if has "$STEPS" audit; then
    reset=${RESET_AUDIT:-$([ "$DATASETS" = "$ALL" ] && echo 1 || echo 0)}
    if [ "$reset" = 1 ]; then act truncate -s 0 "$AUDIT"; fi
    prev=""
    for ds in $DATASETS; do
        for mode in "" --loso; do
            dep=()
            if [ -n "$prev" ]; then dep=(--dependency=afterany:$prev); fi
            job "audit_$ds${mode:+_loso}" $C ${dep[@]+"${dep[@]}"} --mem=180G --time=01:00:00 -- \
                scripts/checks/check_splits.py --dataset "$ds" $mode
            prev=$LAST
        done
    done
fi

if has "$STEPS" offset; then
    for ds in $DATASETS; do
        # --subjects 1000 = every listener (the paper's ESAA scan used all 17)
        job "offset_$ds" $C --time=02:00:00 -- \
            scripts/checks/offset_scan.py --dataset "$ds" --stream-name unified_env64.pt --subjects 1000
    done
fi

if has "$STEPS" leak; then
    for ds in $DATASETS; do
        job "foldleak_$ds" $C --time=01:00:00 -- scripts/checks/check_fold_leak.py \
            --dataset "$ds" --groups unified_groups.json --ref unified_groups_v2.json
    done
fi

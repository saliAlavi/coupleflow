#!/usr/bin/env bash
# Build the preprocessed bundles of the 9 paper datasets from their raw releases.
#
# Produces, under $COUPLEFLOW_DATA/<dataset>/ (scripts/env/osc.sh; default <repo>/data):
#   unified_env64.pt   kuleuven dtu fuglsang2020 esaa uhd255 broderick_cp sparrkulee maestro:
#                      64 canonical channels, 0.5-32 Hz, 64 Hz, envelopes + 8/28 sub-bands.
#                      megmasc: 64-component per-listener PCA (only the "PCA" LOSO comparison
#                      \losoMEGpca in App. C uses it; its envelopes also define the content groups)
#   unified_native.pt  megmasc: all 208 KIT sensors -- the MEG-MASC input of every model and
#                      baseline in the paper (coupleflow/preprocessing/run_config.py);
#                      uhd255 (253 ch) and broderick_cp (128 ch): native-channel variants
#   unified_pkg.pt     maestro with the release's own timing (no 0.781-s latency correction)
# Read by: every training run, baseline and check (via run_config.py); unified_native.pt of
# uhd255 / broderick_cp by the `unified_native` runs (tables/channels.tex, tables/app_variants.tex);
# unified_pkg.pt by the maestro_pkg_* runs (tables/maestro_main.tex, App. C timing comparison).
# Next: scripts/data/stimulus_groups.sh, then scripts/data/checks.sh.
#
# Compute (CPU only; scripts/slurm/build.sbatch = 32 cores, 180 GB): 13 jobs of 1-35 min each
# (SparrKULee ~35 min, MEG-MASC ~10 min per bundle, the rest < 10 min), about 2.5 node-hours in
# all; ~31 GB of bundles.  Raw releases: see scripts/data/README.md.
#
# Usage (from anywhere; runs from the repo root):
#   bash scripts/data/build_bundles.sh                    # submit every build job
#   DRY_RUN=1 bash scripts/data/build_bundles.sh          # print the commands only
#   DATASETS="kuleuven dtu" bash scripts/data/build_bundles.sh
#   SMOKE=1 bash scripts/data/build_bundles.sh            # one listener each -> *.smoke.pt
#   MODE=local bash scripts/data/build_bundles.sh         # no Slurm: run the jobs here in turn
# Environment: DRY_RUN, MODE (slurm|local), DATASETS, SMOKE, DEPEND (extra Slurm dependency),
# COUPLEFLOW_ENV (environment file the jobs source; default scripts/env/osc.sh).
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
SMOKE=${SMOKE:-0}
S=scripts/slurm/build.sbatch
B=coupleflow/preprocessing/build.py
LIM=()
if [ "$SMOKE" = 1 ]; then LIM=(--limit 1); fi    # build.py and the reader modules write *.smoke.pt

for ds in $DATASETS; do
    case $ds in
    kuleuven|dtu|esaa|sparrkulee)
        # readers with read(): the generic builder (16 workers for the two largest releases)
        case $ds in kuleuven|sparrkulee) w=16 ;; *) w=12 ;; esac
        job "build_$ds" $S -- $B --dataset "$ds" --workers $w ${LIM[@]+"${LIM[@]}"} ;;
    uhd255)
        # the native job re-reads the CURRY files that the first job extracts from S<k>.tar.gz
        job build_uhd255 $S -- $B --dataset uhd255 --workers 12 ${LIM[@]+"${LIM[@]}"}
        dep=()
        if [ -n "$LAST" ]; then dep=(--dependency=afterok:$LAST); fi
        job build_uhd255_native $S ${dep[@]+"${dep[@]}"} -- \
            -m coupleflow.preprocessing.readers.uhd255 --native --workers 10 ${LIM[@]+"${LIM[@]}"} ;;
    megmasc)
        job build_megmasc $S -- $B --dataset megmasc --workers 12 ${LIM[@]+"${LIM[@]}"}
        job build_megmasc_native $S --export=ALL,MEGMASC_NATIVE=1 -- \
            $B --dataset megmasc --workers 8 --name unified_native.pt ${LIM[@]+"${LIM[@]}"} ;;
    fuglsang2020)
        # envelope-only release: the reader module builds the bundle itself
        job build_fuglsang2020 $S -- \
            -m coupleflow.preprocessing.readers.fuglsang2020 --workers 12 ${LIM[@]+"${LIM[@]}"} ;;
    broderick_cp)
        # envelope-only release: the reader module builds the bundle itself
        job build_broderick_cp $S -- \
            -m coupleflow.preprocessing.readers.broderick_cp --workers 16 ${LIM[@]+"${LIM[@]}"}
        job build_broderick_cp_native $S -- \
            -m coupleflow.preprocessing.readers.broderick_cp --native --workers 16 ${LIM[@]+"${LIM[@]}"} ;;
    maestro)
        # six talkers: the reader module builds the bundle itself.  Its --limit counts eval
        # trials and keeps the bundle name, so the smoke build names its output explicitly.
        m=(-m coupleflow.preprocessing.readers.maestro --workers 16)
        if [ "$SMOKE" = 1 ]; then
            job build_maestro $S -- "${m[@]}" --subjects S01 --limit 2 --name unified_env64.smoke.pt
            job build_maestro_pkg $S -- "${m[@]}" --subjects S01 --limit 2 --latency 0 \
                --name unified_pkg.smoke.pt
        else
            job build_maestro $S -- "${m[@]}"
            job build_maestro_pkg $S -- "${m[@]}" --latency 0 --name unified_pkg.pt
        fi ;;
    *)
        echo "unknown dataset $ds" >&2; exit 1 ;;
    esac
done

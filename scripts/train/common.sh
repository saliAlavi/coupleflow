# Shared settings for scripts/train/*.sh (sourced, not run).  Each experiment script does
#     set -euo pipefail; cd "$(dirname "$0")/../.."; source scripts/train/common.sh
# and calls `submit` once per (tag, dataset).  DRY_RUN=1 prints the sbatch commands instead of
# submitting them (scripts/train/verify_configs.py parses that output).
set -f   # overrides such as eval.cand_sets=[a,b] must never be glob-expanded

ALL_DS=(kuleuven dtu fuglsang2020 esaa uhd255 broderick_cp sparrkulee megmasc)
TWO_DS=(kuleuven dtu fuglsang2020 esaa uhd255 broderick_cp)   # two-talker EEG
ABL_DS=(kuleuven dtu megmasc)                                    # ablation datasets

# Content-disjoint folds = min(5, content groups) and listeners (= LOSO folds), from
# stats.n_groups / stats.groups_per_subject of <data>/<ds>/unified_groups.json.  train.sbatch
# exits for an index past either count, so these only avoid allocating idle GPUs.
declare -A NFOLD=([kuleuven]=4 [dtu]=5 [fuglsang2020]=4 [esaa]=5 [uhd255]=2 [broderick_cp]=5
                  [sparrkulee]=5 [megmasc]=5 [maestro]=5)
declare -A NLISTENER=([kuleuven]=16 [dtu]=18 [fuglsang2020]=44 [esaa]=17 [uhd255]=30
                      [broderick_cp]=33 [sparrkulee]=85 [megmasc]=27 [maestro]=16)

# Per-dataset run settings (identical in every experiment):
#   Fuglsang'20, Broderick   envelope-only releases                  -> BANKS=none
#   SparrKULee, MEG-MASC     single talker (match-mismatch)          -> SINGLE=1
#   MEG-MASC                 all 208 sensors                         -> STREAM=unified_native.pt
#   Fuglsang'20              validation = last 20% of trials >= 30 s (run_config.split_kw)
ds_env() {        # extra --export entries (leading comma)
    case $1 in
        fuglsang2020|broderick_cp) echo ",BANKS=none" ;;
        sparrkulee) echo ",SINGLE=1" ;;
        megmasc) echo ",SINGLE=1,STREAM=unified_native.pt" ;;
        *) echo "" ;;
    esac
}
ds_overrides() {  # extra main.py overrides
    case $1 in
        fuglsang2020) echo "data.tail_min_s=30 data.val_frac=0.2" ;;
        *) echo "" ;;
    esac
}
ds_sbatch() {     # sbatch resources beyond train.sbatch's 96 GB / 16 h
    case $1 in
        megmasc) echo "--mem=120G" ;;        # 6.7-GB 208-sensor bundle (paper runs: 120 GB)
        sparrkulee) echo "--time=24:00:00" ;; # 85 listeners; longest paper run 8.8 h (1-s training)
        *) echo "" ;;
    esac
}

# EEG lag window.  125 ms (+125..+1109 ms after the sound) is the model default
# (model.lag_offset falls back to 8 samples at 64 Hz); 0 ms adds model.lag_offset=0.
lag_override() { [ "$1" = 0 ] && echo "model.lag_offset=0" || echo ""; }
# Window chosen on validation by scripts/results/select_lag.py (paper/tables/lag_selection.json).
LAG_JSON=${LAG_JSON:-paper/tables/lag_selection.json}
chosen_lag() {
    python3 -c "import json,sys; c=json.load(open(sys.argv[1]))[sys.argv[2]]['choice']; \
print({'125ms': 125, '0ms': 0}[c])" "$LAG_JSON" "$1"
}

# Keys added to config.yaml after some paper runs were made.  Those runs' stored config.yaml
# lacks them and the code used its cfg.get(key, fallback), which equals today's default.  The
# launches of the affected tags pin the fallback explicitly, so a later change of the default
# cannot silently change them (verify_configs.py checks key absence + pinned value).
PIN_PRE_LOSO="data.split_mode=stimulus data.cand_idx=null"   # before LOSO and MAESTRO
PIN_PRE_TAIL="data.tail_min_s=200.0 eval.select_metric=contrib"   # before the tail rule
pins() {   # pins PINS OVERRIDES -> the pins whose keys OVERRIDES does not set itself
    local p o out=()
    for p in $1; do
        for o in $2; do [ "${p%%=*}" = "${o%%=*}" ] && continue 2; done
        out+=("$p")
    done
    echo "${out[*]}"
}

# One-component ablations (scripts/train/ablations.sh, maestro_ablations.sh).
abl_override() {
    case $1 in
        noflow) echo "model.use_flow=false losses.flow=0" ;;
        norank) echo "model.rank=0" ;;                 # no listener correction
        nomatch) echo "data.matched=false" ;;
        noglimpse) echo "data.glimpse=false" ;;
        nobanks) echo "data.feat_banks=[]" ;;          # MAESTRO only
        noadv) echo "losses.adv=0" ;;
        nohinge) echo "losses.null_hinge=0" ;;
        nocon) echo "losses.con=0" ;;                  # no InfoNCE
        selacc) echo "eval.select_metric=acc" ;;
        noside) echo "model.use_side=false losses.side=0" ;;
        noaux) echo "losses.env=0 losses.col=0" ;;
        before) echo "model.lag_mode=before" ;;        # lag control: EEG before the sound
        *) echo "unknown ablation $1" >&2; return 1 ;;
    esac
}

# MAESTRO (4 attendable talkers): K-way candidates and their candidate sets / windows.
MAESTRO_OV="data.cand_idx=[0,1,2,3] eval.cand_sets=[cands_matched,cands,same_talker3,same_talker] eval.windows=[1.0,2.0,5.0,10.0,20.0]"

N_SUBMIT=0
# submit JOB_NAME EXPORTS ARRAY [SBATCH_OPTION ...] -- [OVERRIDE ...]
#   EXPORTS: comma list appended to --export=ALL (DS=..,TAG=..,SEED=..[,SINGLE=1,...])
submit() {
    local name=$1 exports=$2 array=$3
    shift 3
    local opts=()
    while [ $# -gt 0 ] && [ "$1" != -- ]; do opts+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    local cmd=(sbatch "--job-name=$name" "--export=ALL,$exports" "--array=$array" "${opts[@]}"
               scripts/slurm/train.sbatch "$@")
    N_SUBMIT=$((N_SUBMIT + 1))
    if [ "${DRY_RUN:-0}" = 1 ]; then
        local a out=()
        for a in "${cmd[@]}"; do   # single-quote anything beyond plain characters
            if [[ $a =~ ^[A-Za-z0-9_./=,:@%+-]+$ ]]; then out+=("$a"); else out+=("'${a//\'/\'\\\'\'}'"); fi
        done
        echo "${out[*]}"
    else
        mkdir -p logs   # train.sbatch writes logs/%x_%A_%a.out; slurm needs the directory first
        "${cmd[@]}"
    fi
}
range0() { echo "0-$(($1 - 1))"; }   # array spec 0..n-1

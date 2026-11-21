#!/bin/bash
# Submit the E8 CPU jobs (scripts/slurm/cpu.sbatch, one per dataset x mode x gap).
# DRY_RUN=1 prints the commands.  Logs: /fs/scratch/PAS2301/alialavi/rebuttal/e8/logs/
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG=/fs/scratch/PAS2301/alialavi/rebuttal/e8/logs
mkdir -p "$LOG" logs
S=scripts/rebuttal/e8/lagctl.py
mem() { case $1 in megmasc) echo "--mem=180G" ;; sparrkulee) echo "--mem=150G" ;; *) echo "--mem=96G" ;; esac; }
run() {  # run DS JOBNAME ARGS...
    local ds=$1 jn=$2; shift 2
    local cmd=(sbatch --job-name="e8_$jn" $(mem "$ds") --output="$LOG/%x_%j.out"
               scripts/slurm/cpu.sbatch "$S" --dataset "$ds" "$@")
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}
for ds in kuleuven dtu sparrkulee megmasc maestro; do
    run "$ds" "${ds}_stim" --mode stim --gaps 125 250 500 1000
done
for ds in kuleuven megmasc dtu sparrkulee maestro; do
    run "$ds" "${ds}_eeg_after" --mode eeg --after
    gaps="125 250 500 1000"; [ "$ds" = maestro ] && gaps="0 $gaps"
    for g in $gaps; do run "$ds" "${ds}_eeg_g$g" --mode eeg --gaps "$g"; done
done

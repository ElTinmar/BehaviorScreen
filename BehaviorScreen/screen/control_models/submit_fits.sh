#!/bin/bash
# Fits of every experiment in experiments.txt, or of those named, each with the resources its ladder needs:
#   bash BehaviorScreen/screen/control_models/submit_fits.sh [experiment ...]
# GPU nodes throughout (the GPUs stay idle: easier to get than CPU nodes), at most 18 cores per GPU:
# renewal rungs on long trials or large grids get a full node, the other renewal ladders half of one,
# ladders without renewal rungs a quarter (run.sbatch's one GPU).
cd "$(dirname "$0")/../../.." || exit 1
LIST=BehaviorScreen/screen/control_models/experiments.txt
HEAVY=" after_looming phototaxis "
LIGHT=" prey_capture_convergent_saccade prey_capture_divergent_saccade okr_conjugate_saccades looming looming_divergent_saccade dark_flash "

mapfile -t ALL < <(tr -d '\r' < "$LIST")
EXPERIMENTS=("$@")
[ ${#EXPERIMENTS[@]} -eq 0 ] && EXPERIMENTS=("${ALL[@]}")
for EXP in "${EXPERIMENTS[@]}"; do
    INDEX=$(printf '%s\n' "${ALL[@]}" | grep -nx "$EXP" | cut -d: -f1)
    [ -z "$INDEX" ] && { echo "Not in $LIST: $EXP"; continue; }
    if [[ $HEAVY == *" $EXP "* ]]; then
        RESOURCES=(--constraint=gpu --gres=gpu:a100:4 --cpus-per-task=72 --mem=500000); N_JOBS=48
    elif [[ $LIGHT == *" $EXP "* ]]; then
        RESOURCES=(--cpus-per-task=18 --mem=32G); N_JOBS=18
    else
        RESOURCES=(--gres=gpu:a100:2 --cpus-per-task=36 --mem=160G); N_JOBS=32
    fi
    echo -n "$EXP: "
    sbatch --array="$INDEX" "${RESOURCES[@]}" --export=ALL,N_JOBS=$N_JOBS BehaviorScreen/screen/control_models/run.sbatch
done

#!/bin/bash
# The preprocessing pipeline on every <line>/<condition> folder of a screen, one job per folder:
#   bash BehaviorScreen/preprocessing/submit.sh SCREEN_ROOT [pipeline options, e.g. --steps qc heatmaps]
# A GPU when the steps include pose, eye_pose or bouts (all steps by default); 12 GB for qc and heatmaps alone.
cd "$(dirname "$0")/../.." || exit 1
ROOT=$1
shift
if [[ " $* " != *" --steps "* || " $* " =~ " "(pose|eye_pose|bouts)" " ]]; then
    RESOURCES=(--constraint=gpu --gres=gpu:a100:1)
elif [[ " $* " =~ " "(export|eye_crop|saccades)" " ]]; then
    RESOURCES=()
else
    RESOURCES=(--mem=12000)
fi
for FOLDER in "$ROOT"/*/*/; do
    echo -n "${FOLDER%/}: "
    sbatch "${RESOURCES[@]}" BehaviorScreen/preprocessing/run.sbatch "${FOLDER%/}" "$@"
done

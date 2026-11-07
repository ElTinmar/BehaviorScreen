#!/bin/bash
# Tier 1, one behavior per array task (the lines of experiments.txt):
#   sbatch BehaviorScreen/point_process/analysis/tier1.sbatch
# then, once they are done, their tables and maps merged:
#   sbatch --array=1 BehaviorScreen/point_process/analysis/tier1.sbatch --combine
# A resubmitted task skips the cells it has already saved.
#SBATCH -o /u/mapri/logs/tier1.out.%A_%a
#SBATCH -e /u/mapri/logs/tier1.err.%A_%a
#SBATCH -D /u/mapri/Code/BehaviorScreen
#SBATCH --array=1-14
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00

PROJECT_ROOT="/raven/u/mapri/Code/BehaviorScreen"
cd "$PROJECT_ROOT" || exit 1
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ "$1" = "--combine" ]; then
    [ "$SLURM_ARRAY_TASK_ID" -gt 1 ] && exit 0  # submitted without --array=1: one task combines
    MODE=(--combine)
else
    BEHAVIOR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" BehaviorScreen/point_process/analysis/experiments.txt | tr -d '\r')
    scontrol update JobId=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} Name=tier1_$BEHAVIOR
    MODE=(--behavior "$BEHAVIOR")
fi

srun conda run --no-capture-output -n BehaviorScreen \
    python -m BehaviorScreen.ablation_screen.run_screen "${MODE[@]}" \
    --data-root /ptmp/mapri --fits-root ./figures --output-dir ./tier1_results

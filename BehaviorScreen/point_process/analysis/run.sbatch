#!/bin/bash
#SBATCH -o /u/mapri/logs/job.out.%j
#SBATCH -e /u/mapri/logs/job.err.%j
#SBATCH -D /u/mapri/Code/BehaviorScreen
#SBATCH --array=1-14
#SBATCH --cpus-per-task=72
#SBATCH --mem=240G
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00

mkdir -p logs

PROJECT_ROOT="/raven/u/mapri/Code/BehaviorScreen"
cd "$PROJECT_ROOT" || exit 1

EXP_FILE="$PROJECT_ROOT/BehaviorScreen/point_process/analysis/experiments.txt"

# Extract key corresponding to the Slurm Array Task ID
EXP_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXP_FILE" | tr -d '\r')
scontrol update JobId=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} Name=$EXP_NAME

echo "Starting task $SLURM_ARRAY_TASK_ID: $EXP_NAME"

# Path options (update if needed)
DATA_DIR="/ptmp/mapri"
OUT_DIR="./figures"

# One fit worker per allocated CPU; each holds the data and its renewal grid (2 GB or more for phototaxis).
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Run Model Fitting & Bootstrapping exclusively on this node
srun conda run --no-capture-output -n BehaviorScreen \
    python -m BehaviorScreen.point_process.analysis.run_analysis \
    --exp "$EXP_NAME" \
    --mode fit \
    --data-root "$DATA_DIR" \
    --output-dir "$OUT_DIR" \
    --n-jobs "$SLURM_CPUS_PER_TASK"
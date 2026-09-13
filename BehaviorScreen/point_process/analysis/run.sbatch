#!/bin/bash
#SBATCH --job-name=point_proc
#SBATCH -o /u/mapri/logs/job.out.%j
#SBATCH -e /u/mapri/logs/job.err.%j
#SBATCH -D /u/mapri/Code/BehaviorScreen
#SBATCH --array=1-15
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=24:00:00

mkdir -p logs

PROJECT_ROOT="/raven/u/mapri/Code/BehaviorScreen"
cd "$PROJECT_ROOT" || exit 1

EXP_FILE="$PROJECT_ROOT/experiments.txt"

# Extract key corresponding to the Slurm Array Task ID
EXP_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXP_FILE" | tr -d '\r')

echo "Starting task $SLURM_ARRAY_TASK_ID: $EXP_NAME"

# Path options (update if needed)
DATA_DIR="/ptmp/mapri"
OUT_DIR="./figures"

# Run Model Fitting & Bootstrapping exclusively on this node
srun conda run --no-capture-output -n BehaviorScreen \
    python -m BehaviorScreen.point_process.analysis.run_analysis \
    --exp "$EXP_NAME" \
    --mode fit \
    --data-root "$DATA_DIR" \
    --output-dir "$OUT_DIR"
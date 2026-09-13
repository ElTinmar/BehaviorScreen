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

# Extract key corresponding to the Slurm Array Task ID
EXP_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments.txt)

echo "Starting task $SLURM_ARRAY_TASK_ID: $EXP_NAME"

# Run Diagnostics first (fast)
python run_analysis.py --exp "$EXP_NAME" --mode diagnostics

# Run Model Fitting & Bootstrapping (compute-intensive)
python run_analysis.py --exp "$EXP_NAME" --mode fit
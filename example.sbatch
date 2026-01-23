#!/bin/bash -l
#SBATCH -o /u/mapri/logs/%x.out.%j
#SBATCH -e /u/mapri/logs/%x.err.%j
#SBATCH -D /u/mapri/Code/BehaviorScreen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32000
#SBATCH --time=24:00:00

module purge
module load anaconda/3/2023.03 cuda/13.0

INPUT_PATH=$1
JOB_NAME=$(basename "$INPUT_PATH")
scontrol update JobId=$SLURM_JOB_ID Name=$JOB_NAME

srun conda run -n BehaviorScreen python -m BehaviorScreen "$INPUT_PATH" 8Jan2026/model

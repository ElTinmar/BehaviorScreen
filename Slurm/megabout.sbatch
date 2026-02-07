#!/bin/bash -l
#SBATCH -o /u/mapri/logs/job.out.%j
#SBATCH -e /u/mapri/logs/job.err.%j
#SBATCH -D /u/mapri/Code/BehaviorScreen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48000
#SBATCH --time=24:00:00
#SBATCH --unbuffered

module purge
module load anaconda/3/2023.03 cuda/13.0

INPUT_PATH=$1
JOB_NAME=$(basename "$INPUT_PATH")
scontrol update JobId=$SLURM_JOB_ID Name=$JOB_NAME

srun conda run -n BehaviorScreen python -u -m BehaviorScreen.megabouts "$INPUT_PATH" --metadata='results' --stimuli='results' --tracking='results' --lightning-pose='lightning_pose' --video='results' --video-timestamp='results'

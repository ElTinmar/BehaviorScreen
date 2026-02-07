#!/bin/bash -l
#SBATCH -o /u/mapri/logs/job.out.%j
#SBATCH -e /u/mapri/logs/job.err.%j
#SBATCH -D /u/mapri/Code/BehaviorScreen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000
#SBATCH --time=24:00:00

module purge
module load anaconda/3/2023.03 

INPUT_PATH=$1
JOB_NAME=$(basename "$INPUT_PATH")
scontrol update JobId=$SLURM_JOB_ID Name=$JOB_NAME

srun --unbuffered conda run -n BehaviorScreen python -m BehaviorScreen.overlay "$INPUT_PATH" --metadata='results' --stimuli='results' --tracking='results' --video='results' --video-timestamp='results'
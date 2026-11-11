#!/bin/bash -l
# The preprocessing pipeline on one <line>/<condition> folder (submit.sh: every folder of a screen):
#   sbatch [--constraint=gpu --gres=gpu:a100:1] BehaviorScreen/preprocessing/run.sbatch FOLDER [pipeline options]
# The steps pose, eye_pose and bouts need the GPU.
#SBATCH -o /u/mapri/logs/job.out.%j
#SBATCH -e /u/mapri/logs/job.err.%j
#SBATCH -D /u/mapri/Code/BehaviorScreen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000
#SBATCH --time=24:00:00

module purge
module load anaconda/3/2023.03 cuda/13.0

scontrol update JobId=$SLURM_JOB_ID Name="preprocessing_$(basename "$(dirname "$1")")/$(basename "$1")"

srun conda run --no-capture-output -n BehaviorScreen python -m BehaviorScreen.preprocessing "$@"

#!/bin/bash -l
#SBATCH -o /u/mapri/logs/job.out.%j
#SBATCH -e /u/mapri/logs/job.err.%j
#SBATCH -D /u/mapri/Code/BehaviorScreen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=48000
#SBATCH --time=24:00:00

module purge
module load anaconda/3/2023.03 cuda/13.0

srun conda run --no-capture-output -n BehaviorScreen python -m BehaviorScreen.point_process
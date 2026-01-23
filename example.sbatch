#!/bin/bash -l
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D /u/mapri/Code/BehaviorScreen
#SBATCH -J bscreen_cort
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32000
#SBATCH --time=24:00:00

module purge
module load anaconda/3/2023.03 cuda/13.0

srun conda run -n BehaviorScreen python -m BehaviorScreen /ptmp/mapri/cort/cort-vehicle 8Jan2026/model

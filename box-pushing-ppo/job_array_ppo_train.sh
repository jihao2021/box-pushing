#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=10:00:00
#SBATCH --array=1-100
#SBATCH --job-name=random_seed_check
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Activate conda
source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate tf2

# Set seed from SLURM array ID
SEED=$SLURM_ARRAY_TASK_ID

# Run your actual training
python train_ppo.py -number_agent 5 -random_seed $SEED
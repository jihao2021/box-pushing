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

# Load only required modules
module purge
module load gcc/13.3.0
module load cuda/12.6.3

# Set environment variable for XLA CUDA path
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw

# Activate conda
source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate tf2

# Set seed from SLURM array ID
SEED=$SLURM_ARRAY_TASK_ID

# Run your actual training
python train.py -number_agent 5 -random_seed $SEED -number_agent_trained_model 5 -random_seed_trained_model 5
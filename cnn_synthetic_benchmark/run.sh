#!/bin/bash -l

#SBATCH --job-name=benchmark_cnn
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --reservation=pytorchcourse1
#SBATCH --account=class02

module load daint-gpu
module load PyTorch

export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1

srun python cnn_distr.py

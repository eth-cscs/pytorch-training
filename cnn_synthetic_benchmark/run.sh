#!/bin/bash -l

#SBATCH --job-name=benchmark-cnn
#SBATCH --time=00:10:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --account=<account>

module load cray-python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
. /apps/daint/UES/6.0.UP04/sandboxes/sarafael/hpcpython2022/bin/activate

export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0

srun python 3-train_cnn_synthetic_data_ddp.py
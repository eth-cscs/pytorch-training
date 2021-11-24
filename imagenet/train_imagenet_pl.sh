#!/bin/bash -l
#SBATCH --job-name=imagenet
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --account=class02
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu

module load daint-gpu PyTorch PyExtensions
source /apps/daint/UES/6.0.UP04/sandboxes/sarafael/deepspeed_env/bin/activate

export NCCL_DEBUG=WARN
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_ALGO=Tree
export NCCL_NTHREADS=512
export PMI_NO_PREINITIALIZE=1

DIR=$SCRATCH/runs/imagenet/pl$SLURM_JOBID

set -x
mkdir -p $DIR
srun -ul python train_imagenet_pl.py \
    /scratch/snx3000/datasets/imagenet/ILSVRC2012_1k \
    --root-dir $DIR \
    $@
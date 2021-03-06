#!/bin/bash -l
## Usage:
## > sbatch -N4 train_imagenet.sh --epochs 4 --gpu-aug <additional arguments>

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

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NNODES
export LOCAL_RANK=0

CHECKPOINT_DIR=$SCRATCH/runs/imagenet/$SLURM_JOBID/
CMD="\
  python train_imagenet.py /scratch/snx3000/datasets/imagenet/ILSVRC2012_1k \
  --save-path $CHECKPOINT_DIR \
  $@ \
"

set -x
mkdir -p $CHECKPOINT_DIR
srun -ul bash -c "RANK=\$SLURM_NODEID $CMD"

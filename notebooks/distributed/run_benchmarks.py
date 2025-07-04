import os
import math
import subprocess

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)
os.makedirs("jobs", exist_ok=True)

# Configs to benchmark
configs = [
    {"method": "ddp", "gpus": 1},
    {"method": "ddp", "gpus": 2},
    {"method": "ddp", "gpus": 4},
    {"method": "ddp", "gpus": 8},
    {"method": "fsdp", "gpus": 1},
    {"method": "fsdp", "gpus": 2},
    {"method": "fsdp", "gpus": 4},
    {"method": "fsdp", "gpus": 8},
]

# Basic sbatch header template
SBATCH_HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/{job_name}.out
#SBATCH --error=logs/{job_name}.err
#SBATCH --time=00:10:00
#SBATCH --account=crs01
#SBATCH --reservation=pytorch
#SBATCH --no-requeue
#SBATCH --exclusive

export DISTRIBUTED_INITIALIZATION_METHOD=SLURM
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export MPICH_GPU_SUPPORT_ENABLED=0
export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=64

srun -ul --environment=./edf.toml bash -c \"
    export RANK=\\$SLURM_PROCID
    export LOCAL_RANK=\\$SLURM_LOCALID
    python main.py --method {method} --epochs 10 --batch_size 256
\"
"""

for cfg in configs:
    method = cfg["method"]
    gpus = cfg["gpus"]
    nodes = math.ceil(gpus / 4)
    job_name = f"{method}_{gpus}gpu"
    sbatch_script_path = f"jobs/{job_name}.sbatch"

    if gpus < 4:
        tasks_per_node=gpus
    else:
        if gpus % 4 != 0:
            raise ValueError(f"Invalid GPU count ({gpus}) for multi-node run: must be a multiple of 4.")
        tasks_per_node=4

    # Write sbatch script using heredoc-style string formatting
    with open(sbatch_script_path, "w") as f:
        f.write(SBATCH_HEADER.format(
            job_name=job_name,
            method=method,
            gpus=gpus,
            nodes=nodes,
            tasks_per_node=tasks_per_node,
        ))

    # Submit job
    print(f"Submitting {job_name} ({gpus} GPUs on {nodes} node(s))")
    subprocess.run(["sbatch", sbatch_script_path])

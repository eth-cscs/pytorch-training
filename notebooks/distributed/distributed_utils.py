import os
import torch
import torch.distributed as dist


def setup():
    """
    Initializes torch.distributed using environment variables (SLURM compatible).
    Assumes:
        - MASTER_ADDR and MASTER_PORT, RANK, LOCAL_RANK and WORLD_SIZE are set in the environment.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not dist.is_initialized():
        ### TODO: Initialize the default process group
        ### Use NCCL backend and supply the correct device_id
        # dist.init_process_group(_______)

    ### TODO: Set the current CUDA device to the local rank
    # torch.cuda.set_device(_______)

    return local_rank

def cleanup():
    dist.destroy_process_group()

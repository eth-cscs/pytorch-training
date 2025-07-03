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
        # Automatically picks up env variables
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

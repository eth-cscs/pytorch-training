import torch
import torch.distributed as dist
from model import CNN
import torch.nn as nn


def get_ddp_model(device_id):

    if dist.get_rank() == 0:
        world_size = dist.get_world_size()
        print(f"[Rank 0] Running in DDP mode with {world_size} processes")

    model = CNN().to(torch.device(f"cuda:{device_id}"))
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])

    return model

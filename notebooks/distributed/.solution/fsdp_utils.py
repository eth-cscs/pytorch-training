import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, CPUOffload, ShardingStrategy
from model import CNN
import torch.nn as nn


def custom_auto_wrap_policy(module, recurse, nonwrapped_numel):
    return isinstance(module, (nn.Conv2d, nn.Linear, nn.Sequential))

def get_fsdp_model(device_id):

    if dist.get_rank() == 0:
        world_size = dist.get_world_size()
        print(f"[Rank 0] Running in FSDP mode with {world_size} processes")

    model = CNN().to(torch.device(f"cuda:{device_id}"))

    mixed_precision = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    model = FSDP(
        model,
        device_id=device_id,
        auto_wrap_policy=custom_auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=False)
    )

    return model

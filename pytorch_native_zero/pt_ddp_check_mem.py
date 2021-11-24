# This script is adapted from the example here
# https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html

import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from pt_distr_env import setup_distr_env


parser = argparse.ArgumentParser(description='PyTorch: check GPU memory usage')
parser.add_argument('--use-zero', action='store_true',
                    help='Use ZeRO optimization')
args = parser.parse_args()


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


torch.manual_seed(0)
torch.cuda.manual_seed(0)

# create a model model
device_id = 0
features = 2000
num_layers = 20
model = nn.Sequential(*[nn.Linear(features, features).to(device_id)
                        for _ in range(num_layers)])
print_peak_memory("Max memory allocated after creating local model", device_id)

# print the number of parameters of the model
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f' * {params:,} parameters')

setup_distr_env()
dist.init_process_group(backend="nccl")

# construct DDP model
ddp_model = DDP(model, device_ids=[device_id])
print_peak_memory("Max memory allocated after creating DDP", device_id)

# define loss function and optimizer
loss_fn = nn.MSELoss()
if args.use_zero:
    optimizer = ZeroRedundancyOptimizer(
        ddp_model.parameters(),
        optimizer_class=torch.optim.Adam,
        lr=0.01
    )
else:
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01)

# forward pass
outputs = ddp_model(torch.randn(20, features).to(device_id))
labels = torch.randn(20, features).to(device_id)

# backward pass
loss_fn(outputs, labels).backward()

# update parameters
print_peak_memory("Max memory allocated before optimizer step()", device_id)
optimizer.step()
print_peak_memory("Max memory allocated after optimizer step()", device_id)

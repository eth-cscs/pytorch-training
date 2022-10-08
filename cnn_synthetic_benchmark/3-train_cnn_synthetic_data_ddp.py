import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import models
from pt_distr_env import DistributedEnviron


distr_env = DistributedEnviron()
dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()
device = distr_env.local_rank

batch_size_per_gpu = 128
num_iters = 10

class SyntheticDataset(Dataset):
    def __getitem__(self, idx):
        data = torch.randn(3, 224, 224)
        target = random.randint(0, 999)
        return (data, target)

    def __len__(self):
        return batch_size_per_gpu * num_iters * world_size


train_set = SyntheticDataset()

train_sampler = DistributedSampler(
    train_set,
    num_replicas=world_size,
    rank=rank,
    shuffle=False,
    seed=42
)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    sampler=train_sampler,
    num_workers=12
)

_model = models.resnet50()
_model.to(device);

ddp_model = DistributedDataParallel(_model, device_ids=[device])

optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)


def benchmark_step(model, imgs, labels):
    optimizer.zero_grad()
    output = model(imgs)
    loss = F.cross_entropy(output, labels)
    loss.backward()
    optimizer.step()


num_epochs = 5
imgs_sec = []
for epoch in range(num_epochs):
    t0 = time.time()
    for step, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        benchmark_step(ddp_model, imgs, labels)

    dt = time.time() - t0
    imgs_sec.append(batch_size_per_gpu * num_iters / dt)

    if rank == 0:
        print(f' * Epoch {epoch:2d}: '
              f'{imgs_sec[epoch]:.2f} images/sec per GPU')

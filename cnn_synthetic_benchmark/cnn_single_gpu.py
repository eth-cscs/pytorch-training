import random
import time
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import models


device = 0
batch_size = 64
num_iters = 5
model_name = 'resnet50'

model = getattr(models, model_name)()

optimizer = optim.SGD(model.parameters(), lr=0.01)


class SyntheticDataset(Dataset):
    def __getitem__(self, idx):
        data = torch.randn(3, 224, 224)
        target = random.randint(0, 999)
        return (data, target)

    def __len__(self):
        return batch_size * num_iters


train_loader = DataLoader(SyntheticDataset(),
                          batch_size=batch_size)

model.to(device)


def benchmark_step(model, imgs, labels):
    optimizer.zero_grad()
    output = model(imgs.to(device))
    loss = F.cross_entropy(output, labels.to(device))
    loss.backward()
    optimizer.step()


num_epochs = 5
for epoch in range(num_epochs):
    t0 = time.time()
    for step, (imgs, labels) in enumerate(train_loader):
        benchmark_step(model, imgs, labels)

    dt = time.time() - t0
    imgs_sec = batch_size * num_iters / dt
    print(f' * Epoch {epoch:2d}: {imgs_sec:.2f} images/sec')

import torch
import torch.nn.functional as F
from tqdm.auto import trange, tqdm


def train(model, train_loader, optimizer, scheduler, logger, log_every=10):
    losses = RunningAverage('Loss')
    acc = RunningAverage('Acc')
    model.train()
    
    for i, (batch,) in enumerate(tqdm(train_loader, leave=False)):
        imgs, target = batch['data'], batch['label'].ravel()

        # Calculate CE loss
        logits = F.log_softmax(model(imgs), dim=1)
        loss = F.nll_loss(logits, target)

        # Zero the parameter gradients
        optimizer.zero_grad()
        # Calculate gradients
        loss.backward()
        # Training step
        optimizer.step()
        # Update LR
        scheduler.step()

        # Update metrics
        with torch.no_grad():
            losses.update(loss.item())
            acc.update((logits.argmax(dim=1) == target).float().mean().item())
            if i % log_every == 0:
                logger(f'{acc} {losses} lr:{optimizer.param_groups[0]["lr"]:.01e}')


@torch.no_grad()
def validate(model, valid_loader):
    losses = AverageMeter('Loss')
    acc = AverageMeter('Acc')
    model.eval()

    for i, (batch,) in enumerate(tqdm(valid_loader, leave=False)):
        imgs, target = batch['data'], batch['label'].ravel()
        # Calculate CE loss
        logits = F.log_softmax(model(imgs), dim=1)
        loss = F.nll_loss(logits, target)
        # Update metrics
        losses.update(loss.item(), len(target))
        acc.update((logits.argmax(dim=1) == target).float().mean().item(), len(target))

    return losses, acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':.03f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self, device='cuda'):
        total = torch.FloatTensor([self.sum, self.count], device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

    
class RunningAverage(object):
    """Computes and stores the running average of the given value"""
    def __init__(self, name, fmt=':.03f', beta=0.98):
        self.name = name
        self.fmt = fmt
        self.beta = beta
        self.reset()

    def reset(self):
        self.avg = None

    def update(self, val):
        if self.avg is None:
            self.avg = val
        self.avg = self.beta*self.avg + (1-self.beta)*val

    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

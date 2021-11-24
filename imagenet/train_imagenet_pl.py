import os
import argparse
from glob import glob

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.models as models

from torchinfo import summary
import pytorch_lightning as pl

from dali_pt_dataloader import dali_dataloader


class LtngModel(pl.LightningModule):
    def __init__(self,
                 data_path='/scratch/snx3000/datasets/imagenet/ILSVRC2012_1k',
                 arch='resnet50',
                 epochs=10,
                 warmup_epochs=0.1,
                 optimizer='SGD',
                 learning_rate=0.05,
                 momentum=0.9,
                 weight_decay=1e-4,
                 dropout=0.4,
                 batch_size=256,
                 image_size=224,
                 image_resize=256,
                 num_threads=12,
                 gpu_aug=True,
                 **kwargs):
        super().__init__()
        # torchvision.models.<arch>
        self.net = models.__dict__[arch]() 
        # save all locals()
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def step(self, batch, log_prefix=''):
        """ forward step, loss and pl logging """
        x, y = batch[0]['data'], batch[0]['label'].ravel()
        y_hat = self.net(x)
        loss = F.cross_entropy(y_hat, y)
        with torch.no_grad():
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            self.log(f'{log_prefix}_loss', loss.item(), prog_bar=False)
            self.log(f'{log_prefix}_acc1', acc.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        return self.step(batch, 'val')

    def test_step(self, batch, batch_nb):
        return self.step(batch, 'test')

    def training_step(self, batch, batch_nb):
        self.log_optimizer_params()
        return self.step(batch, 'train')

    def log_optimizer_params(self):
        params = self.optimizer.param_groups[0]
        self.log('lr', params['lr'], prog_bar=True)
        if 'betas' in params:
            mom = params['betas'][0] # Adam like opt
        elif 'momentum' in params:
            mom = params['momentum'] # SGD like opt
        else:
            return # no momentum
        self.log('mom', mom, prog_bar=True)

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        if self.hparams.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(),
                                        lr=self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay,
                                        nesterov=True)
        else:
            self.optimizer = optim.AdamW(self.parameters(),
                                        lr=self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay,
                                        betas=(self.hparams.momentum, 0.999))

        self.lr_scheduler = {'scheduler': optim.lr_scheduler.OneCycleLR(
                                        self.optimizer,
                                        max_lr=self.hparams.learning_rate,
                                        steps_per_epoch=len(self.train_dataloader()),
                                        pct_start=(self.hparams.warmup_epochs / self.hparams.epochs),
                                        epochs=self.hparams.epochs,
                                        anneal_strategy="cos",
                                        cycle_momentum=False,
                                        div_factor = 100,
                                        final_div_factor = 100,
                                    ),
                                'name': 'learning_rate',
                                'interval':'step',
                                'frequency': 1}

        return [self.optimizer], [self.lr_scheduler]

    def train_dataloader(self):
        self.train_loader = dali_dataloader(
            batch_size=self.hparams.batch_size,
            num_threads=self.hparams.num_threads,
            tfrec_filenames=sorted(glob(f'{self.hparams.data_path}/train/*')),
            tfrec_idx_filenames=sorted(glob(f'{self.hparams.data_path}/idx_files/train/*')),
            resize=self.hparams.image_resize,
            crop=self.hparams.image_size,
            shard_id=self.global_rank,
            num_shards=self.trainer.world_size,
            gpu_aug=self.hparams.gpu_aug,
            gpu_out=True,
            training=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_loader = dali_dataloader(
            batch_size=self.hparams.batch_size,
            num_threads=self.hparams.num_threads,
            tfrec_filenames=sorted(glob(f'{self.hparams.data_path}/validation/*')),
            tfrec_idx_filenames=sorted(glob(f'{self.hparams.data_path}/idx_files/validation/*')),
            resize=self.hparams.image_resize,
            crop=self.hparams.image_resize, # don't crop validation
            shard_id=self.global_rank,
            num_shards=self.trainer.world_size,
            gpu_aug=self.hparams.gpu_aug,
            gpu_out=True,
            training=False,
        )
        return self.val_loader


def parse_args():
    """ define and parse args with argparse """
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data_path', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-t', '--num_threads', default=12, type=int,
                        help='number of data loading threads (default: 4)')
    parser.add_argument('-e', '--epochs', default=90, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=80, type=int,
                        help='mini-batch size per node (default: 80)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate', dest='learning_rate')
    parser.add_argument('-m', '--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=3e-4, type=float,
                        help='weight decay (default: 3e-4)', dest='weight_decay')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='dropout probability (default: 0.3)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--root-dir', default='', type=str,
                        help='directory to save logs and latest checkpoints (default: none)')
    parser.add_argument('--gpu-aug', action='store_true',
                        help='Use GPU for image decoding and augmentation.')

    # List torchvision.models
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help=f"model architecture: {' | '.join(model_names)} (default: resnet50)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    num_nodes = int(os.environ['SLURM_NNODES'])
    rank = int(os.environ['SLURM_NODEID'])
    project = f"ImageNet_DALI_{args.arch}"
    print(project, rank, num_nodes, args)

    model = LtngModel(**args.__dict__)
    if rank == 0:
        summary(model, input_size=(model.hparams.batch_size, 3, model.hparams.image_size, model.hparams.image_size))
        print(model.hparams)

    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint

    wandb_logger = WandbLogger(
        project=project,
        save_dir=args.root_dir,
        log_model=True,
    )
    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log="all")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc1',
        dirpath=args.root_dir,
        filename=project + '-{epoch:02d}-{val_loss:.2f}.pt'
    )

    trainer = pl.Trainer(
        default_root_dir=args.root_dir,
        gpus=1, num_nodes=num_nodes,
        max_epochs=model.hparams.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        strategy="ddp",
        # strategy="deepspeed_stage_2",  # Enable DeepSpeed ZeRO stage 2
        # precision=16,                 # Enable torch.cuda.amp
        # amp_backend="apex", amp_level="O2",  # Uses Nvidia APEX AMP instead
        # strategy="fsdp",              # Enable Fully Sharded Data Parallel
        replace_sampler_ddp=False,      # disable sampler as DALI shards the data itself
        progress_bar_refresh_rate=(args.print_freq if rank==0 else 0),
        enable_progress_bar=(rank==0),
    )
    trainer.fit(model)


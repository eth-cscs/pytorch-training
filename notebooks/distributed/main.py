import torch
import torch.optim as optim
import torch.nn as nn
from model import CNN
from train import train
from data import get_dataloaders
from distributed_utils import setup, cleanup
from ddp_utils import get_ddp_model
from fsdp_utils import get_fsdp_model
import argparse

def run(args):
    ### TODO: Initialize distributed training (hint: distributed_utils.py)
    # local_rank = _____
    device = torch.device(f"cuda:{local_rank}")

    trainloader, validloader = get_dataloaders(global_batch_size=args.batch_size)

    if args.method == "ddp":
        model = get_ddp_model(local_rank)
    elif args.method == "fsdp":
        model = get_fsdp_model(local_rank)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # The optimizer and the loss function are identical to the serial training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # In contrast to the serial version, we the negative log likelihood with simple summation instead
    # of averaging. The reason is that we will reduce the loss from all participating workers later
    # in the training and the simple summation makes the loss scale invariant.
    loss_function = nn.NLLLoss(reduction='sum')

    ### TODO: Launch training
    # train(_____)

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["ddp", "fsdp"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256, help="Global batch size across all workers")
    parser.add_argument("--print_every", type=int, default=32, help="Global batch size across all workers")
    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()

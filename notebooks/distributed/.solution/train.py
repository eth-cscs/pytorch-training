import torch
import torch.distributed as dist
import time
from utils import log_training_step


def train(model, optimizer, loss_function, epochs, device, trainloader, validloader, print_every):

    model.to(device)
    model.train()

    train_losses, valid_losses, accuracies = [], [], []
    
    global_step = 0
    total_steps = epochs * len(trainloader)
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Optional: get distributed sampler to reshuffle data each epoch
    train_sampler = getattr(trainloader, "sampler", None)
    if isinstance(train_sampler, torch.utils.data.distributed.DistributedSampler):
        has_distributed_sampler = True
    else:
        has_distributed_sampler = False

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_start = time.time()

        # Required to shuffle differently at each epoch
        if has_distributed_sampler:
            train_sampler.set_epoch(epoch)

        # Training loop for one epoch
        for images, labels in trainloader:
            step_start = time.time()

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)

            # DDP will all-reduce grads automatically during .backward()
            loss.backward()

            optimizer.step()

            # Log scalar loss (detach to avoid side effects)
            epoch_loss += loss.detach().item()

            log_training_step(
                iteration=global_step,
                total_train_iters=total_steps,
                loss_tensor=loss,
                model=model,
                optimizer=optimizer,
                step_start_time=step_start,
                print_every=print_every,
            )

            global_step += 1

        epoch_duration = time.time() - epoch_start

        # Validation
        model.eval()
        valid_loss = torch.tensor(0.0, device=device)
        correct_preds = torch.tensor(0.0, device=device)
        total_samples = torch.tensor(0.0, device=device)

        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)

                # Accumulate validation loss
                valid_loss += loss_function(output, labels).detach()

                # Accuracy calculation
                probs = torch.exp(output)
                top_p, top_class = probs.topk(1, dim=1)
                correct = (top_class == labels.view(-1, 1)).type(torch.FloatTensor)
                correct_preds += correct.sum()
                total_samples += labels.size(0)

        # Synchronize metrics across all workers
        if dist.is_initialized():
            dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_preds, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        avg_valid_loss = valid_loss.item() / len(validloader.dataset)
        avg_accuracy = 100.0 * correct_preds.item() / total_samples.item()

        if rank == 0:
            train_losses.append(epoch_loss / len(trainloader))
            valid_losses.append(avg_valid_loss)
            accuracies.append(avg_accuracy)
            
            print(f"[Epoch {epoch+1}] "
                  f"Val Loss: {avg_valid_loss:.4f} | "
                  f"Accuracy: {avg_accuracy:.2f}%  | "
                  f"Train Loss: {train_losses[-1]:.4f}")

            # Structured log for metrics
            epoch_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"[METRIC] epoch={epoch+1} "
                  f"time={epoch_duration:.2f} "
                  f"val_loss={avg_valid_loss:.4f} "
                  f"acc={avg_accuracy:.2f} "
                  f"peak_mem={epoch_peak_mem:.0f}MB")

        model.train()

    return train_losses, valid_losses, accuracies

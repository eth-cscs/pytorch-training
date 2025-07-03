import torch
import torch.distributed as dist
import time


def all_reduce_avg(tensor):
    """
    Averages the input tensor across all distributed ranks.
    """
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


class color:
    """
    ANSI color codes for terminal output formatting.
    courtesy - https://gist.github.com/nazwadi/ca00352cd0d20b640efd
    """
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def pretty_log(
    iteration,
    total_train_iters,
    train_loss,
    elapsed_time_per_iteration,
    grad_norm,
    learning_rate,
):
    """
    Constructs a colored log string with memory and gradient stats.
    """

    log_string = f"{color.BOLD}{color.CYAN}> global batch {iteration:8d}/{total_train_iters:8d} {color.END}|"
    log_string += f" elapsed time per batch (ms): {elapsed_time_per_iteration:.1f} |"
    log_string += f" learning rate: {learning_rate:.3E} |"
    log_string += f"{color.GREEN} loss: {train_loss:.5f}{color.END} |"

    curr_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    log_string += f" memory used: {curr_mem:.3f} GB (peak {peak_mem:.3f} GB) |"
    log_string += f" grad_norm: {grad_norm:.2f}"

    return log_string


def log_training_step(
    iteration,
    total_train_iters,
    loss_tensor,
    model,
    optimizer,
    step_start_time,
    print_every=10,
):
    """
    Logs training info every `print_every` steps.
    Reduces and logs training stats across ranks.
    Should be called inside the training loop.
    """

    # Only log every `print_every` steps
    if iteration % print_every != 0 and iteration != total_train_iters - 1:
        return

    # Compute grad norm (sync across model replicas)
    total_norm = torch.norm(
        torch.stack([
            p.grad.detach().data.norm(2)
            for p in model.parameters() if p.grad is not None
        ])
    ).to(loss_tensor.device)

    loss_avg = all_reduce_avg(loss_tensor.clone().detach())
    norm_avg = all_reduce_avg(total_norm.clone().detach())

    # Elapsed time
    elapsed = (time.time() - step_start_time) * 1000  # in ms

    if dist.get_rank() == 0:
        lr = optimizer.param_groups[0]['lr']
        print(pretty_log(
            iteration,
            total_train_iters,
            loss_avg.item(),
            elapsed,
            norm_avg.item(),
            lr
        ))


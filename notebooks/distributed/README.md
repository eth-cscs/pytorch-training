# Distributed Training with PyTorch

In this exercise, you'll transition from a working **single-GPU CNN model** to a **scalable multi-GPU** training setup using **Distributed Data Parallel (DDP)** and **Fully Sharded Data Parallel (FSDP)** in PyTorch.

You’ve already trained a CNN on a single GPU using JupyterLab. Now, we will scale this pipeline to efficiently use multiple GPUs and nodes.


## Files Overview

| File                   | Description                                                                                |
|------------------------|--------------------------------------------------------------------------------------------|
| `model.py`             | CNN model definition. Previously used on single GPU. Now adapted for flexible batch sizes. |
| `train.py`             | Training & evaluation loop. DDP/FSDP-compatible. Tracks timing, memory, accuracy.          |
| `data.py`              | CIFAR-10 dataset loader. Downloads on rank 0 only.                                         |
| `main.py`              | Entrypoint. Initializes distributed context, sets up model/training.                       |
| `distributed_utils.py` | Utility to initialize `torch.distributed` from environment (e.g., via `srun`).             |
| `ddp_utils.py`         | Wraps model in DistributedDataParallel (DDP).                                              |
| `fsdp_utils.py`        | Wraps model in FSDP.                                                                       |
| `utils.py`             | Logging, printing                                                                          |
| `run.sh`               | Main SLURM script to launch experiments.                                                   |
| `edf.toml`             | EDF container config (image and hooks).                                                    |
| `run_benchmarks.py`    | Automates batch SLURM submissions for scalability studies.                                 |
| `collect_results.py`   | Parses `[METRIC]` lines from logs and compiles a CSV summary.                              |
| `plot_results.py`      | Generates runtime, loss, and memory plots from CSV.                                        |
| `plot_results.sh`      | Convenience script to call plotting & open output.                                         |

> `run_benchmarks.py`, `plot_results.py`, `plot_results.sh`, and `collect_results.py` are used in the final scaling experiment.


## Learning Goal

Transition your CNN training pipeline from a single GPU to a distributed setup using DDP and FSDP — and understand the design, tradeoffs, and performance implications of each.

---


## Section 1: Review  `model.py`

The CNN model in `model.py` is similar to what you used earlier, with one key difference: it now supports flexible batch sizes, necessary for distributed training. The model also includes assertions to help debug shape mismatches.


### Exercise 1.1: Understand Output Shapes

Look at the following assertion in the model:

```python
assert x.shape[1:] == (3, 32, 32)
```

Why do we use x.shape[1:] instead of x.shape == (64, 3, 32, 32) as you did in your previous exercise?

---


## Section 2: Distributed Setup (`distributed_utils.py`)

PyTorch’s `torch.distributed` module enables multi-process parallel training across GPUs and nodes. It provides:
- Process groups for communication between workers
- Backends like nccl, gloo, mpi
- DDP and FSDP relies on the distributed module

When launching with `srun` (or torchrun), environment variables like:
- `RANK` → global rank of the process
- `LOCAL_RANK` → rank on the current node
- `WORLD_SIZE` → total number of processes
must be set, such that `distributed.init_process_group` can understand and connect the workers.

If you're curious of how these variables are set exactly, have a look at `run.sh`.


### Exercise 2.1: Complete the Setup Function

Complete the missing parts in `distributed_utils.py`.

---


## Section 3: Data Loading with DistributedSampler (`data.py`)

When training in parallel with DDP or FSDP, each process (or GPU) runs its own independent training loop. To prevent data duplication, we use `torch.utils.data.DistributedSampler`, which:
- Splits the dataset into equal-sized chunks
- Ensures each process only sees its own shard
- Optionally shuffles shards (at each epoch)


### Exercise 3.1: Complete the Loader Setup

- Complete the missing parts in `data.py`.
- Why do we call sync after downloading?
- What would happen if we didn't use a DistributedSampler in DDP/FSDP training?

---


## Section 4: Wrapping Models for Parallel Training

Files: `ddp_utils.py` and `fsdp_utils.py`

In distributed training, wrapping the model is how we enable efficient gradient synchronization and memory optimizations:
- DDP (DistributedDataParallel) synchronizes gradients across processes at each backward pass.
- FSDP (FullyShardedDataParallel) shards the model’s weights, gradients, and optimizer states across GPUs for memory savings.


### Exercise 4.1: Understanding `get_ddp_model()`

- complete the missing parts in `ddp_utils.py`
- In this code, is the model replicated or sharded across GPUs?

### Exercise 4.2: FSDP & Memory Optimization

Have a look at `fsdp_utils.py`:

- What does `auto_wrap_policy` do in FSDP?
- What does `MixedPrecision(...)` help with?
- What does `ShardingStrategy.FULL_SHARD` mean? How does it differ from DDP?

Try to look up the documentation for to answer these questions.

### Exerecise 4.3: DDP vs. FSDP

- When would you choose FSDP over DDP?
- What are the tradeoffs in performance and complexity?

---


## Section 5: Putting It All Together (`main.py`)

The `main.py` script is the central controller for:
- Initializing distributed training
- Loading data
- Choosing DDP or FSDP
- Launching training
-  Cleaning up

### Exercise 5.1: Try a Dry Run (no training)

Complete the missing parts in `main.py`, but leave the call to `train(...)` commented out for now.
We can now run the training with slurm:

  $ sbatch run.sh

This will use 1 node (4 GPUs per node) by default, and you will be able to see from the output stored in `train.log` and `train.err` whether your code works as expected.
If you want to reduce the number of tasks per node (i.e. use only 1 GPU, for example) you can

  $ sbatch --ntasks-per-node=1 run.sh

Similarly, we can increase the number of nodes using

  $ sbatch --nodes=2 run.sh

The command line options override the `SBATCH` defaults specified in `run.sh`.

Try to answer some of these questions:
- What happens if `cleanup()` is omitted at the end?
- Why do we always use `local_rank` when setting the device?
- Could we also use `torchrun` instead of `srun`?

---


## Section 6: Training Loop and Evaluation (`train.py`)

The training function:
- Initializes the model on the right GPU (device)
- Loops over epochs
- In each epoch:
    - Runs a training loop over batches
    - Computes loss and gradients
    - Logs training metrics
- At the end of the epoch:
    - Evaluates the model on validation data
    - Averages metrics across all workers
    - Logs output

### Exercise 6.1: Complete the training code

- complete the missing parts in `train.py`
- What would happen if we forgot to call `optimizer.zero_grad()` before `loss.backward()`?
- Why is validation done with `torch.no_grad()`?
- What is the purpose of the all-reduce after validation?
- Why do we switch to model.train() at the end of the epoch?

### Exercise 6.2: Run the Training

Now that the full training loop is implemented, it's time to run the code using slurm and verify everything works as expected. Submit a job using your existing run.sh:

  $ sbatch run.sh

Wait for the job to complete. Check the output log (e.g., train.log). You can also peek into the file as it is being written using

  $ tail -f train.log

- Does the training loss decrease over time?
- Does the validation loss decrease as well?
- Does the accuracy improve each epoch?
- Is there a noticeable difference between DDP and FSDP runs?

Change between DDP and FSDP - do you notice a difference?

---


## Section 7: Run and Analyze Benchmarks

The purpose of this exercise is to evaluate how well our training setup scales with:
- Increasing number of GPUs
- Different parallelization methods (DDP vs FSDP)

We'll measure:
- Training speed (time per epoch)
- Validation performance (loss and accuracy)
- GPU memory usage

### Exercise 7.1: Run the Benchmarks

You’ll run several training jobs with different configurations:

  $ python run_benchmarks.py

This script will:
- Automatically generate temporary SLURM scripts using sbatch
- Launch jobs with different GPU counts (1, 2, 4, 8)
- Try both DDP and FSDP

You can follow the job logs in `logs/`.

### Exercise 7.2: Visualize the Results

Once all jobs have finished run:

  $ ./plot_results.sh

This will:
- Parse the output logs with `collect_results.py`
- Create a CSV summary of metrics
- Generate plots via `plot_results.py`

Three plots will be produced, each with number of GPUs on the x-axis:

1. Time per epoch (log-scaled)
    - Shows how training speed improves (or doesn’t) with more GPUs
    - Ideal: time decreases ~linearly

2. Validation loss and accuracy
    - Shows whether scaling affects model convergence
    - Loss should not increase / accuracy decrease with more GPUs

3. Peak memory per GPU
    - Shows how much memory is used
    - Ideal: FSDP uses less memory than DDP


- Which method (DDP or FSDP) achieves better training speed?
- Do you see ideal scaling (linear speedup) as you add more GPUs?
- Does validation loss or accuracy degrade when using more GPUs?
- Which method is more memory-efficient?

Exploration:
- Change the global batch size in `run_benchmarks.py` and observe what happens
- Add configurations with more epochs or mixed precision turned off

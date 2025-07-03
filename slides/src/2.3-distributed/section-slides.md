---
layout: section
---

# Distributed Training

---
level: 2
---

# Why distributed training?

- Training deep neural networks is computationally expensive
- Models can be very large 
  - e.g. GPT-3 has 175 billion parameters
  - We need to store the model parameters in memory
  - Will not fit in a single GPU memory


---
level: 2
---

# Data Parallelism

<img src="/images/ddp1.png" class="w-300"/>
<br></br>

- Split the data across multiple GPUs
- Each GPU has a copy of the model
- Simplest way to distribute training: Distributed Data Parallel (DDP)


---
level: 2
transition: none
---

# Data Parallelism: DDP

<img src="/images/ddp2.png" class="w-300"/>

---
level: 2
---

# Data Parallelism: DDP

<img src="/images/ddp3.png" class="w-300"/>

---
level: 2
transition: none
---


<img src="/images/ddp4.png" class="w-300"/>

---
level: 2
transition: none
---


<img src="/images/ddp5.png" class="w-300"/>

---
level: 2
transition: none
---


<img src="/images/ddp6.png" class="w-300"/>

---
level: 2
transition: none
---


<img src="/images/ddp7.png" class="w-300"/>

---
level: 2
transition: none
---


<img src="/images/ddp8.png" class="w-300"/>

---
level: 2
---


<img src="/images/ddp9.png" class="w-300"/>


---
level: 2
---

# Data Parallelism: DDP


<div grid="~ cols-2 gap-4">
<div class="col-span-1">


### Pros:


- Easy to implement
- Scales well with the number of GPUs
- Can be used with any model
- Good computation/communication ratio
- Speedup is close to linear with the number of GPUs


</div>
<div class="col-span-1">


### Cons:


- Large models do not fit in a single GPU memory


</div>
</div>

---
level: 2
---

# Can we compress the model?

- Observations:
  1. Model states (in particular optimizer states) use the most memory
  2. Some states can be stored in lower precision (e.g. 16-bit floats)


---
level: 2
---

# Optimizer State: Adam

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

- Maintains two moving averages:
  - **First moment** (mean) estimate
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \frac{\partial J}{\partial w_t}$$
  - **Second moment** (uncentered variance) estimate
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) \left(\frac{\partial J}{\partial w_t}\right)^2$$

</div>
<div class="col-span-1">

- Bias correction for both estimates:
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2}
$$

- Update rule for weights:
$$
w_{t+1} = w_t - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} 
$$

- Mixed precision training:
  - Store gradients and weights in 16-bit floats
  - Use 32-bit floats for bias-corrected estimates during computation

</div>
</div>

---
level: 2
---

# Data Parallelism: Zero Redundancy Optimizer (ZeRO)

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

- Three stages of ZeRO:
    1. Optimizer State Partitioning (Pos)
        - 4x memory reduction
        - same communication volume
    2. Gradient Partitioning (Pos+g)
        - 8x memory reduction
        - same communication volume
    3. Parameter Partitioning (Pos+g+p)
        - Memory reduction is linear with data parallelism degree (Nd)
        - 1.5x increase in communication volume

</div>
<div class="col-span-1">

- Developed by Microsoft Research (DeepSpeed)
- Additional features:
    - CPU offloading
    - Mitigates memory fragmentation

</div>
</div>


---
level: 2
---


<div class="mx-auto w-200">
    <img src="/images/ZeRO.png" />
    <div class="text-xs text-gray-500 italic mt-1">
        Rajbhandari, Samyam, et al. "Zero: Memory optimizations toward training trillion parameter models." SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 2020.
    </div>
</div>

- $\Psi$ : Number of parameters
- $K$: Optimizer state memory overhead factor ($K = 12$ for Adam)


---
level: 2
---

# Fully Sharded Data Parallel (FSDP)

- PyTorch native implementation of ZeRO-3

**Forward Pass**
- Each GPU holds **only its shard** of model weights
- Before computation, each layer's full weights are **gathered ("all-gather")** from shards
- Forward pass runs as usual
- Afterward, weights are optionally **discarded** to save memory


**Backward Pass**
- Gradients are computed **locally**
- Gradients are **reduced-scattered** across devices (each device keeps only the gradient shard it needs)
- Optimizer updates are done **shard-wise** (no full model needed)


---
level: 2
---

# Fully Sharded Data Parallel (FSDP)

<div class="mx-auto w-200">
    <img src="/images/fsdp.png" />
    <div class="text-xs text-gray-500 italic mt-1">
        <a href="https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/" >https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/</a>
    </div>
</div>

---
level: 2
---

# Data Parallelism: ZeRO/FSDP

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

Pros:
- Scales to large models
- Memory savings
- Supports mixed precision training
- Can be used with any model
- Good compute/communication ratio (1,5x DDP communication volume)

</div>
<div class="col-span-1">

Cons:
- More complex to implement & debug
- Overhead from frequent all-gather/reduce-scatter
- Very large layers: 
  - may not fit in memory even with sharding
  - computation is not parallelized across GPUs

</div>
</div>

---
level: 2
---

# Model Parallelism: Tensor/Operator Parallelism

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

- Each GPU holds a part of the model
- Layers of the model are split across multiple GPUs
    - Not only data but also computation is split
    - Parallel (dense) linear algebra operations
- Requires more communication between GPUs
- Example: Megatron-LM, DeepSpeed, Axonn

</div>
<div class="col-span-1">

<div class="mx-auto w-100">
    <img src="/images/operator_parallelism.png" />
    <div class="text-xs text-gray-500 italic mt-1">
        By Hoefler, Torsten, et al. "HammingMesh: A Network Topology for Large-Scale Deep Learning." Communications of the ACM 67.12 (2024): 97-105.
    </div>
</div>

</div>
</div>

---
level: 2
---

# Model Parallelism: Tensor/Operator Parallelism

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

- Pros:
  - Can handle very large models that do not fit in a single GPU memory
  - Good for models with large layers (e.g. transformers)

</div>
<div class="col-span-1">

- Cons:
  - More complex to implement: specialized frameworks needed
    - Only support a limited of distributed operators
    - often requires adapting the model
  - Requires more communication between GPUs
  - Requires low-latency fabric: Efficiency degrades quickly beyond a single node

</div>
</div>

---
level: 2
---

# Model Parallelism: Pipeline Parallelism

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

- Split the model into stages (groups of layers)
- Each stage is executed on a different GPU
- Use for very deep models (e.g. transformers)
- Communication: 
  - Forward pass: send activations from one stage to the next
  - Backward pass: send gradients back through the stages
- Works with any model

</div>
<div class="col-span-1">

<div class="mx-auto w-120">
    <img src="/images/pipeline_parallelism.png" />
    <div class="text-xs text-gray-500 italic mt-1">
        By Hoefler, Torsten, et al. "HammingMesh: A Network Topology for Large-Scale Deep Learning." Communications of the ACM 67.12 (2024): 97-105.
    </div>
</div>

</div>
</div>

---
level: 2
---

# Model Parallelism: Pipeline Parallelism

<div class="mx-auto w-200">
    <img src="/images/pipeline_ds_2.png" />
    <div class="text-xs text-gray-500 italic mt-1">
        <a href="https://www.deepspeed.ai/tutorials/pipeline/" >https://www.deepspeed.ai/tutorials/pipeline/</a>
    </div>
</div>


<br></br>

- 2-stage pipeline parallelism
- Combined with 2-way data parallelism
- Training of 4 mini-batches


---
level: 2
---

# 3D Parallelism

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

<div class="mx-auto w-90">
    <img src="/images/data_parallelism.png" />
</div>

</div>
<div class="col-span-1">

<div class="mx-auto w-60">
    <img src="/images/operator_parallelism.png" />
</div>

</div>
</div>


<div grid="~ cols-2 gap-4">
<div class="col-span-1">

<div class="mx-auto w-60">
    <img src="/images/pipeline_parallelism.png" />
</div>

</div>
<div class="col-span-1">

<div class="mx-auto w-110">
    <img src="/images/3d_parallelism.png" />
</div>

</div>
</div>

<div class="text-xs text-gray-500 italic mt-1">
By Hoefler, Torsten, et al. "HammingMesh: A Network Topology for Large-Scale Deep Learning." Communications of the ACM 67.12 (2024): 97-105.
</div>

---
level: 2
---

# NVIDIA Collective Communication Library (NCCL)

- inter-GPU (collective) communication primitives

| Broadcast     | Reduce | AllReduce | AllGather |   ReduceScatter  | Send/Recv |
|---|---|---|---|---|---|

<br></br>

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

- single kernel implementation for communication and computation
- topology-aware, supports NVLINK
- API is similar to MPI (non-blocking version, e.g. `MPI_Isend`)
- bootstrap for parallel environment is out-of-band (not provided)

</div>
<div class="col-span-1">

- can be used in combination with MPI
- collectives may be faster than GPU-aware MPI
- often used in AI applications (PyTorch, Keras or TensorFlow)
- AMD's version: RCCL

</div>
</div>

---
level: 2
---

# HPE Slingshot Network


<div grid="~ cols-4 gap-4">
<div class="col-span-3">

- Built from Cassini NIC (PCIe Gen4) and Rosetta Switch (64-port).
    - one NIC (Cassini) per GH200 module (4 per node)
- Uses 200 Gbps Ethernet links for dragonfly networks
    - 25 GB/s unidirectional peak throughput per NIC
- Supports 64 to 250,000+ endpoints (largest system: ~85,000 endpoints).
- Optimized Ethernet protocol
- Adaptive routing & congestion management optimize performance.
- support remote memory operations

</div>
<div class="col-span-1">

<img src="/images/nw-local.png" class="w-50"/>

</div>
</div>

---
level: 2
---

# Dragonfly Network Topology

<div grid="~ cols-3 gap-4">
<div class="col-span-2">

- Low-diameter “direct” network with no external fat-tree top switches.
- Fewer optical links needed → reduces cost for large systems.
- Groups of routers act as high-radix virtual routers to improve scalability.
- Three levels of connectivity:
    - Edge links (L0): Nodes to local routers.
    - Intra-group links (L1): Routers within a group.
    - Inter-group links (L2): Optical connections between groups.
- Scales linearly in cost while providing high bandwidth.
- Max three hops between components in the network.

</div>
<div class="col-span-1">

<img src="/images/nw-topology.png" class="w-100"/>

</div>
</div>

---
level: 2
---

# Alps Network Stack

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

## Libfabric

- Open Fabrics Interfaces (OFI)
- low-level communication library
- abstraction over different network technologies
- network specifics are implemented as providers

</div>
<div class="col-span-1">

## CXI Provider

- enables libfabric on HPE/Cray's Slingshot network
- makes use of proprietary extensions of the network protocol
- message matching offload to NIC
- RMA and active messages
- GPU-to-NIC RDMA
- Fabric-accelerated small reductions

</div>
</div>

---
level: 2
---


<div grid="~ cols-4 gap-4">
<div class="col-span-3">

# aws-ofi-plugin

- enables NCCL to use libfabric as network provider
- maps NCCLs connection-oriented transport APIs to libfabric's connection-less reliable interface
    - reliable message support
    - OS bypass
- originally developed by NVIDIA and AWS to support Elastic Fabric Adapter (EFA)
- HPE worked together with NVIDIA and AWS to support Slingshot
- open source: https://github.com/aws/aws-ofi-nccl
- fabric and NIC offload capabilities in HPE Slingshot: overlap the compute and communication

</div>
<div class="col-span-1">

<img src="/images/ofi.png" class="w-50"/>

</div>
</div>

---
level: 2
---

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

# `torch.distributed`

- Core PyTorch package for distributed training
- Initializes communication between processes
- Common backends:
  - `nccl` (recommended for GPU)
  - `gloo` (CPU and fallback)


</div>
<div class="col-span-1">

```python
import torch.distributed as dist

dist.init_process_group(
  backend="nccl",
  device_id=torch.device(f"cuda:{GPU_ID}"))
torch.cuda.set_device(GPU_ID)
```

- By default, PyTorch uses environment variables
  - `RANK`: global rank of the process
  - `LOCAL_RANK`: rank on the current node
  - `WORLD_SIZE`: total number of processes
  - `MASTER_ADDR`: address of the master node
  - `MASTER_PORT`: port for communication

</div>
</div>

---
level: 2
---

# `DistributedSampler`

- Ensures each process gets a unique slice of the dataset
- Avoids overlapping data between workers
- Should be used with `DataLoader` in distributed training

```python
from torch.utils.data import DataLoader, DistributedSampler

# Same arguments as serial DataLoader
sampler = DistributedSampler(train_dataset, shuffle=True)
# Watch out for the batch size!
loader = DataLoader(trainset, batch_size=local_batch_size, sampler=sampler)
```

---
level: 2
---

# `torch.nn.parallel.DistributedDataParallel` (DDP)

- Wraps the model for multi-GPU training
- Each process gets its own model replica

```python
import torch.nn as nn
model = nn.parallel.DistributedDataParallel(model, device_ids=[GPU_ID])
```

---
level: 2
---

# `torch.distributed.fsdp.FullyShardedDataParallel` (FSDP)

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

- Advanced parallelism: shards parameters, gradients, and optimizer states
- Reduces memory usage significantly
- Useful for large models
- Wraps your model to enable FSDP using `auto_wrap_policy`
```python
def custom_auto_wrap_policy(
    module, recurse, nonwrapped_numel):
    ...
```

</div>
<div class="col-span-1">

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, CPUOffload, ShardingStrategy

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
```

</div>
</div>

---
level: 2
---

# Training Loop Structure


- Loop over epochs
    - Shuffle data using `DistributedSampler.set_epoch(epoch)`
    - Loop over batches (same as serial version!)
        - Forward pass
        - Compute loss
        - Backward pass
        - Optimizer step

    - Validation
        - validation loss: synchronization using `dist.all_reduce`




---
level: 2
---

# Lab: Distributed Training of CNN


- From a working **single-GPU CNN model** to a **scalable multi-GPU** training
    - **Distributed Data Parallel (DDP)**
    - **Fully Sharded Data Parallel (FSDP)**
- In the end: Strong scaling study (fixed **global batch size**)
    - How does the training time change with the number of GPUs
- Slurm job submission (no JupyterLab)

---
level: 2
---

# SSH Access to CSCS Cluster

<div grid="~ cols-12 gap-4">
<div class="col-span-4">

- For a convenience, it is recommended to add the following lines to your `~/.ssh/config` file

```
Host ela
    HostName ela.cscs.ch
    User <YOUR_USERNAME>
    ForwardAgent yes
    ForwardX11 yes

Host daint
    HostName daint.cscs.ch
    User <YOUR_USERNAME>
    ProxyJump ela
    ForwardAgent yes
    ForwardX11 yes
```

</div>
<div class="col-span-7">


- After that, you can connect to the cluster using the command below

```bash
$ ssh <YOUR_USERNAME>@daint
```

- You should be dropped into the `bash` shell in your home directory
- For launching jobs and we use the scratch directory

```bash
$ cd $SCRATCH/pytorch-training/
```

- pull the latest changes

```bash
$ git pull
$ cd notebooks/distributed
```

</div>
</div>

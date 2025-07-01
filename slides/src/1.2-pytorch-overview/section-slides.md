---
layout: section
---

# PyTorch Overview

---
level: 2
---

# The Fundamental Building Block: `torch.Tensor`

<div grid="~ cols-2 gap-4" class="mt-4">
<div class="col-span-1">

```python {all|1|3-4|6-7|9-13|15-17|all}
import torch

# Create a 2D tensor
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Random weight tensor on GPU
w = torch.randn(2, 2).to("cuda")

# Enable autograd tracking
x = torch.tensor([[1.0, 2.0]], requires_grad=True)

# Forward computation (tracked)
y = x @ w.T

# Stop tracking: detach from graph
y_detached = y.detach()

# You can check:
print(y.requires_grad)         # True
print(y_detached.requires_grad)  # False
```

</div>

<div class="col-span-1">

<v-clicks at="-4">

- Like NumPy arrays, but with GPU support and autograd.
- **`.to("cuda")`**
  Moves tensor to the GPU for faster computation (if available).
- **`requires_grad=True`**
  Enables gradient tracking for optimization. Needed for model parameters.
- **`.detach()`**
  Returns a new tensor that shares data but is not part of the computation graph.
  Use when you want to stop gradients (e.g. logging or freezing layers).

</v-clicks>

</div>
</div>

---
level: 2
---

# Define a Model: Subclass `nn.Module`

<div grid="~ cols-2 gap-4" class="mt-4">
<div class="col-span-1">

````md magic-move {lines: true}
```python {all|1-7|9-12}
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)
```
```python
# Simpler: use nn.Sequential
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
```
````

</div>
<div class="col-span-1">

<v-clicks at="-2">

- `nn.Module` helps manage parameters and layers cleanly
- `forward()` defines the computation graph
- Use `nn.Sequential` for simple, linear stacks of layers
  - no need to define `forward()` explicitly

</v-clicks>

</div>
</div>

---
level: 2
---

# A Layer is a Module Too: `nn.Linear`

```python
layer = nn.Linear(4, 8)

x = torch.randn(1, 4)  # batch size = 1, features = 4
out = layer(x)

print(out.shape)  # torch.Size([1, 8])
```

> Automatically includes weights + biases, and supports gradients.

---
level: 2
---

# The Forward Pass + Autograd

```python
model = SimpleMLP()
x = torch.randn(5, 4)         # mini-batch of 5 samples
out = model(x)                # forward pass
loss = out.sum()              # dummy loss

loss.backward()               # computes dLoss/dParam
```

> Gradients are now stored in each parameter’s `.grad` field.

---
level: 2
---

## Inspecting Model Parameters

```python
# Get all parameters
for param in model.parameters():
    print(param.shape)

# Named parameters
for name, param in model.named_parameters():
    print(name, param.shape)

torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

> Use `.state_dict()` for fine control and reproducibility.

---
level: 2
---

# `torch.nn.functional` vs `torch.nn`

<br></br>

<div grid="~ cols-2 gap-4" class="mt-4">
<div class="col-span-1">

## `torch.nn`

### Layer objects with state

```python
self.relu = nn.ReLU()
self.linear = nn.Linear(10, 5)

x = self.linear(x)
x = self.relu(x)
```

- Registered as part of the model
- Has internal weights, tracked in `.parameters()`

</div>
<div class="col-span-1">

## `torch.nn.functional`

### Stateless ops

```python
import torch.nn.functional as F


x = F.linear(x, weight, bias)  # pass weights manually
x = F.relu(x)
```

- Not registered in the model
- Good for custom layers or one-off ops

</div>
</div>

<Admonition type="note" title="Best Practice">
Use nn for model structure, F.* for flexibility and lower-level control.
</Admonition>

---
level: 2
---

# Training Step: `optimizer.step()`

```python {all|1|3-5|7-11|all}
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Dummy input/output
x = torch.randn(5, 4)
y = torch.randn(5, 1)

# Training step
out = model(x)
loss = nn.functional.mse_loss(out, y)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

> `.step()` updates weights, `.zero_grad()` resets gradients.

---
level: 2
---

# Inference in PyTorch: Use `torch.no_grad()`

<div grid="~ cols-2 gap-4" class="mt-4">
<div class="col-span-1">


```python {all|3-5|3-5|3-5|1|all}
model.eval()  # important: set to inference mode

with torch.no_grad():
    x = torch.randn(1, 4)
    out = model(x)
```

</div>
<div class="col-span-1">

<v-clicks at="-4">

- Disables autograd inside the block
- No computation graph is created
- Reduces memory + improves speed
- `model.eval()` disables Dropout, BatchNorm updates

</v-clicks>

</div>
</div>

---
level: 2
---

# Exclude from Computational Graph: `.detach()`

```python
output = model(x).detach()
```

- Removes tensor from the computation graph
- Still useful outside of `no_grad()`
- Great for logging, visualizations, etc.

---
level: 2
---

# Datasets: `torch.utils.data.Dataset`

<div grid="~ cols-2 gap-4" class="mt-4">
<div class="col-span-1">

A `Dataset` is a read-only wrapper around your data

- Must implement:
  - `__len__()` — number of samples
  - `__getitem__(index)` — returns a single data point

</div>
<div class="col-span-1">

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
```

</div>
</div>

> This is the lowest-level API for loading datasets in PyTorch.

---
level: 2
---

## `DataLoader`: Batching, Shuffling, Parallel Loading

<div grid="~ cols-2 gap-4" class="mt-4">
<div class="col-span-1">

A `DataLoader` wraps a `Dataset` to enable:

- Mini-batches
- Shuffling
- Multi-process data loading
- Pinning memory (for fast GPU transfer)

</div>
<div class="col-span-1">

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    my_dataset,
    batch_size=32,
    shuffle=True,
)

for batch in loader:
    process(batch)
```

</div>
</div>

> Use `DataLoader` in every training loop — it handles batching and efficiency for you.

---
level: 2
---

# Recap

<div grid="~ cols-2 gap-4" class="mt-4">
<div class="col-span-1">

- `Tensor` → basic data container (with autograd!)
- `nn.Module` → base class for models
- `nn.Linear` → layers that manage weights
- `forward()` → defines computation
- `loss.backward()` + `optimizer.step()` → training
- `torch.no_grad()` → inference mode
- `Dataset` + `DataLoader` → data handling

</div>
<div class="col-span-1">

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
y.backward()
print(x.grad)  # tensor([2.])
```

> User code looks and feels like native Python, but is backed by a powerful engine underneath

</div>
</div>

---
level: 2
---

# Under the Hood: PyTorch Execution Model

| **Layer**            | **Description**                                  |
|------------------|----------------------------------------------|
| Python Frontend  | You write code using `torch.*` APIs          |
| Autograd Engine  | Tracks ops & builds computation graph        |
| ATen             | Core C++ tensor library                      |
| C++/CUDA Kernels | Low-level execution (CPU, GPU)               |
| XLA, Vulkan, etc | Alternate hardware targets                   |

- Operations are dispatched via the **dispatcher** to the appropriate backend (e.g., CPU/GPU).
- PyTorch uses a dynamic computation graph (define-by-run) with native code under the hood.

---
level: 2
---

# `torch.compile()` — Ahead-of-Time (AOT) Mode

Introduced in **PyTorch 2.0+**

<div grid="~ cols-2 gap-4" class="mt-4">
<div class="col-span-1">

- Compiles model to a graph (speed)
- Wraps model in an optimizing runtime
- Uses TorchDynamo + backend compiler (e.g. TorchInductor)
  - TorchInductor uses Triton to generate fused GPU kernels
- Works with most models & training loops

</div>
<div class="col-span-1">

```python
compiled_model = torch.compile(model)
out = compiled_model(input)
```

</div>
</div>


---
level: 2
---

# Profiling in PyTorch

Use PyTorch’s built-in profiler to analyze bottlenecks in training and inference.

<div grid="~ cols-2 gap-4" class="mt-4">
<div class="col-span-1">

## Basic Usage

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True
) as prof:
    model(input)

print(prof.key_averages().table(
    sort_by="cuda_time_total"))
```

</div>
<div class="col-span-1">

## Output Options

- Terminal summary
- Chrome trace viewer (`prof.export_chrome_trace("trace.json")`)
- TensorBoard plugin (`torch.profiler.tensorboard_trace_handler`)

</div>
</div>

---
level: 2
---

# NVIDIA Profiling Tools

| Tool              | Purpose                        |
|-------------------|--------------------------------|
| **Nsight Systems** | Timeline of CPU/GPU interaction |
| **Nsight Compute** | Kernel-level GPU performance   |
| **nvprof**         | Lightweight CLI profiler (legacy) |
| **ncu** / **nsys** | Modern CLI tools               |


---
level: 2
---

# Run PyTorch with `nsys` (Nsight Systems)

```bash
nsys profile -o output_report python train.py
```

- Collects timeline, kernel launches, CPU–GPU syncs
- Outputs .qdrep or .nsys-rep for use in GUI

Optional flags:
```bash
--capture-range=cudaProfilerApi \
--trace=cuda,osrt,nvtx
```

<Admonition type="note" title="Tip">
Recommended: add NVTX annotations for custom markers
</Admonition>


---
level: 2
---

# Add NVTX Markers in Your Code

PyTorch supports NVTX via torch.cuda.nvtx

```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("forward pass")
out = model(x)
nvtx.range_pop()

nvtx.range_push("loss + backward")
loss = criterion(out, y)
loss.backward()
nvtx.range_pop()
```

- Tags your code with named regions
- Visible in Nsight Systems timeline
- Helps isolate bottlenecks: forward, backward, data loading, etc.

---
level: 2
---

# `torchvision`: Computer Vision

## What it provides:
- Pretrained models (ResNet, ViT, Faster R-CNN, etc.)
- Common datasets (CIFAR, ImageNet, COCO...)
- Data transforms and augmentations
- Image utilities (`utils.draw_bounding_boxes` etc.)

```
from torchvision import models, transforms

model = models.resnet18(pretrained=True)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
```

---
level: 2
---

# `torchaudio`: Audio & Speech

## Tools for:
- Loading/writing audio (`torchaudio.load`)
- Preprocessing (spectrograms, MFCCs)
- Pretrained models (Wav2Vec2, HuBERT, etc.)
- ASR, speech commands, audio classification

```python
import torchaudio

waveform, sr = torchaudio.load("clip.wav")
spec = torchaudio.transforms.MelSpectrogram()(waveform)
```

---
level: 2
---

# `torchtext`: NLP Datasets & Preprocessing

## 🧰 Includes:
- Text tokenizers and pipelines
- Ready-to-use NLP datasets (AGNews, IMDB, etc.)
- Pretrained embeddings (GloVe, fastText)

```python
from torchtext.datasets import AG_NEWS

train_iter = AG_NEWS(split='train')
for label, text in train_iter:
    print(label, text)
    break
```

> Best for traditional NLP workflows (before Transformers).

---
level: 2
---


# Hugging Face Transformers + PyTorch

Go-to library for **NLP**, **LLMs**, and increasingly **multimodal** deep learning.

## What it provides:

- Pretrained models: BERT, GPT-2/3/4, LLaMA, Mistral, T5, etc.
- Tokenizers and model loading
- Pipelines for easy use
- Tools for fine-tuning and training



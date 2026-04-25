---
name: "safetensors-pytorch"
displayName: "Safetensors & PyTorch Tensors"
description: "Expert guidance for working with HuggingFace safetensors format and PyTorch tensor serialization. Covers safe model saving/loading, tensor manipulation, memory-mapped access, and distributed shard loading."
keywords: ["safetensors", "pytorch", "tensor", "huggingface", "model-weights"]
author: "MeshRun Team"
---

# Safetensors & PyTorch Tensors

## Overview

This power provides expert-level guidance for working with the HuggingFace safetensors format and PyTorch tensor operations. Safetensors is a secure, fast, zero-copy tensor serialization format that replaces Python's pickle-based approach, eliminating arbitrary code execution risks while providing superior performance.

Key capabilities covered:
- Saving and loading model weights with safetensors
- PyTorch tensor creation, manipulation, and memory layout
- Partial/selective tensor loading for multi-GPU and distributed inference
- Converting between safetensors and PyTorch native formats
- Tensor contiguity, dtype casting, and device management
- Sharded model loading patterns for large language models

## Onboarding

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (`torch`)
- safetensors library

### Installation

```bash
# With pip
pip install safetensors torch

# With uv
uv pip install safetensors torch

# With conda
conda install -c conda-forge safetensors pytorch
```

### Verification

```python
import torch
from safetensors.torch import save_file, load_file

# Quick smoke test
tensors = {"test": torch.randn(2, 3)}
save_file(tensors, "test.safetensors")
loaded = load_file("test.safetensors")
assert torch.equal(tensors["test"], loaded["test"])
print("safetensors + PyTorch working correctly")
```

## Safetensors Format Specification

The safetensors file format is structured as:

```
┌──────────────────────────────────────────────┐
│  8 bytes: header_size (little-endian uint64)  │
├──────────────────────────────────────────────┤
│  N bytes: JSON header (UTF-8)                 │
│    - Maps tensor names → {dtype, shape,       │
│      data_offsets: [start, end]}              │
│    - Optional "__metadata__" key for          │
│      user-defined string→string metadata      │
├──────────────────────────────────────────────┤
│  Remaining bytes: raw tensor data             │
│    - Contiguous, row-major (C order)          │
│    - Tensors laid out sequentially per        │
│      data_offsets                              │
└──────────────────────────────────────────────┘
```

The JSON header must begin with `{` (0x7B) and may be padded with spaces (0x20). No executable code can be embedded — only raw numeric data and a JSON metadata header.

### Supported dtypes

| safetensors dtype | PyTorch dtype    | Bytes/element |
| ----------------- | ---------------- | ------------- |
| `F16`             | `torch.float16`  | 2             |
| `BF16`            | `torch.bfloat16` | 2             |
| `F32`             | `torch.float32`  | 4             |
| `F64`             | `torch.float64`  | 8             |
| `I8`              | `torch.int8`     | 1             |
| `I16`             | `torch.int16`    | 2             |
| `I32`             | `torch.int32`    | 4             |
| `I64`             | `torch.int64`    | 8             |
| `U8`              | `torch.uint8`    | 1             |
| `BOOL`            | `torch.bool`     | 1             |

## Common Workflows

### Workflow 1: Save and Load Model Weights

The most common pattern — saving a PyTorch model's state dict to safetensors.

```python
import torch
from safetensors.torch import save_file, load_file

# Save model weights
model = MyModel()
state_dict = model.state_dict()
save_file(state_dict, "model.safetensors")

# Load model weights
loaded_state = load_file("model.safetensors", device="cpu")
model.load_state_dict(loaded_state)
```

**With metadata:**

```python
from safetensors.torch import save_file

save_file(
    tensors=state_dict,
    filename="model.safetensors",
    metadata={
        "format": "pt",
        "model_name": "my-3b-llm",
        "num_parameters": "3000000000",
    }
)
```

**Important:** Metadata values must be strings. The metadata dict is `Dict[str, str]` only.

### Workflow 2: Selective / Partial Tensor Loading

Load only specific tensors — critical for distributed inference where each node loads a subset of layers.

```python
from safetensors import safe_open

# Load only specific tensors by name
with safe_open("model.safetensors", framework="pt", device="cuda:0") as f:
    # List all tensor names
    all_keys = f.keys()

    # Load only the tensors you need
    embedding = f.get_tensor("embedding")
    layer_0_attn = f.get_tensor("layers.0.attention.weight")
```

**Slice loading (partial tensor reads):**

```python
from safetensors import safe_open

with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    # Get a slice object without loading the full tensor
    tensor_slice = f.get_slice("embedding")

    # Inspect shape without loading data
    vocab_size, hidden_dim = tensor_slice.get_shape()

    # Load only a sub-range (e.g., first 1000 rows)
    partial = tensor_slice[:1000, :]
```

This is memory-efficient because safetensors supports zero-copy memory-mapped reads — only the requested bytes are loaded from disk.

### Workflow 3: Convert PyTorch .pt/.bin to Safetensors

```python
import torch
from safetensors.torch import save_file

# Load legacy pickle-based checkpoint
state_dict = torch.load("model.bin", map_location="cpu", weights_only=True)

# Save as safetensors
save_file(state_dict, "model.safetensors")
```

**Important:** Always use `weights_only=True` when loading pickle-based files (PyTorch 2.6+ makes this the default). This prevents arbitrary code execution from untrusted checkpoint files.

### Workflow 4: In-Memory Serialization (Bytes)

Useful for sending tensors over the network or storing in databases.

```python
from safetensors.torch import save, load
import torch

tensors = {
    "hidden": torch.randn(32, 768, dtype=torch.float16),
    "logits": torch.randn(32, 50257, dtype=torch.float16),
}

# Serialize to bytes (no file I/O)
raw_bytes: bytes = save(tensors)

# Deserialize from bytes
restored: dict[str, torch.Tensor] = load(raw_bytes)
```

### Workflow 5: Sharded Model Loading for Distributed Inference

For large models split across multiple safetensors files (common with HuggingFace models):

```python
import json
from pathlib import Path
from safetensors import safe_open

model_dir = Path("my-3b-model")

# Read the shard index (generated by HuggingFace save_pretrained)
with open(model_dir / "model.safetensors.index.json") as f:
    index = json.load(f)

# index["weight_map"] maps tensor_name → shard_filename
weight_map = index["weight_map"]

# Load only the layers assigned to this node
my_layers = ["layers.0.attention.weight", "layers.0.ffn.weight"]

tensors = {}
for tensor_name in my_layers:
    shard_file = model_dir / weight_map[tensor_name]
    with safe_open(str(shard_file), framework="pt", device="cuda:0") as f:
        tensors[tensor_name] = f.get_tensor(tensor_name)
```

## PyTorch Tensor Essentials

### Tensor Contiguity

Safetensors requires tensors to be **contiguous and dense**. Non-contiguous tensors (from slicing, transposing, etc.) must be made contiguous before saving.

```python
import torch

t = torch.randn(4, 4)
sliced = t[:, ::2]          # Non-contiguous view
print(sliced.is_contiguous())  # False

# Fix: make contiguous before saving
contiguous = sliced.contiguous()
print(contiguous.is_contiguous())  # True
```

**Common operations that produce non-contiguous tensors:**
- `tensor.T` or `tensor.transpose()`
- Slicing with step: `tensor[::2]`
- `tensor.permute()`
- `tensor.expand()`

### Dtype Casting

```python
import torch

# Cast to half-precision for inference
t = torch.randn(1024, 1024, dtype=torch.float32)
t_fp16 = t.to(torch.float16)       # or t.half()
t_bf16 = t.to(torch.bfloat16)

# Cast to int8 for quantized models
t_int8 = t.to(torch.int8)
```

### Device Management

```python
import torch

# Move tensors between devices
t = torch.randn(1024, 1024)
t_gpu = t.to("cuda:0")             # CPU → GPU
t_cpu = t_gpu.to("cpu")            # GPU → CPU
t_gpu1 = t_gpu.to("cuda:1")       # GPU → different GPU

# Load safetensors directly to device
from safetensors.torch import load_file
tensors = load_file("model.safetensors", device="cuda:0")
```

### Memory Layout: Row-Major (C-Contiguous)

PyTorch tensors default to row-major (C-contiguous) memory layout, which matches safetensors' expected format. The last dimension varies fastest in memory.

```python
import torch

t = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])
# Memory layout: [1, 2, 3, 4, 5, 6]  (row-major)
# Stride: (3, 1) — moving along dim 0 skips 3 elements

print(t.stride())  # (3, 1)
print(t.is_contiguous())  # True
```

## Troubleshooting

### Error: "Tensors need to be contiguous and dense"

**Cause:** Attempting to save a non-contiguous tensor (from transpose, slice, etc.)
**Solution:**
```python
# Before saving, ensure contiguity
tensor = tensor.contiguous()
```

### Error: "Expected Dict[str, torch.Tensor]"

**Cause:** Passing nested dicts or non-tensor values to `save_file`.
**Solution:** Flatten the state dict so every value is a `torch.Tensor`:
```python
# Bad: nested dict
{"layer": {"weight": tensor}}

# Good: flat dict with dotted keys
{"layer.weight": tensor}
```

### Error: "weights_only" / UnpicklingError with torch.load

**Cause:** PyTorch 2.6+ defaults to `weights_only=True`, blocking pickle-based code execution.
**Solution:** This is a security feature. If you trust the source:
```python
# Only for trusted files:
state = torch.load("model.pt", weights_only=False)

# Better: convert to safetensors and never worry again
from safetensors.torch import save_file
save_file(state, "model.safetensors")
```

### Large Model OOM During Loading

**Cause:** Loading all tensors into memory at once.
**Solution:** Use selective loading with `safe_open`:
```python
from safetensors import safe_open

with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    # Load one tensor at a time
    for key in f.keys():
        tensor = f.get_tensor(key)
        process_and_free(tensor)
```

## Best Practices

- Always use safetensors over pickle-based formats (`torch.save`) for security — safetensors cannot execute arbitrary code
- Use `safe_open` with selective loading for large models to minimize memory usage
- Ensure tensors are contiguous before saving: call `.contiguous()` on transposed/sliced tensors
- Store metadata as string key-value pairs in the safetensors header for model provenance
- Use `weights_only=True` when loading any pickle-based PyTorch checkpoints
- For distributed inference, use the shard index pattern (`model.safetensors.index.json`) to load only the layers each node needs
- Prefer `load_file(path, device="cuda:0")` over loading to CPU then moving — it avoids an extra copy
- Keep tensor names flat with dotted notation (`"layers.0.weight"`) rather than nested dicts

## API Quick Reference

| Function                                      | Purpose                               | Returns             |
| --------------------------------------------- | ------------------------------------- | ------------------- |
| `save_file(tensors, filename, metadata=None)` | Save dict of tensors to file          | `None`              |
| `load_file(filename, device="cpu")`           | Load all tensors from file            | `Dict[str, Tensor]` |
| `save(tensors, metadata=None)`                | Serialize tensors to bytes            | `bytes`             |
| `load(data)`                                  | Deserialize tensors from bytes        | `Dict[str, Tensor]` |
| `safe_open(filename, framework, device)`      | Context manager for selective loading | `SafeOpen`          |
| `SafeOpen.keys()`                             | List tensor names                     | `List[str]`         |
| `SafeOpen.get_tensor(name)`                   | Load a single tensor                  | `Tensor`            |
| `SafeOpen.get_slice(name)`                    | Get a slice object (lazy)             | `TensorSlice`       |
| `TensorSlice.get_shape()`                     | Get tensor shape without loading      | `List[int]`         |
| `TensorSlice[...]`                            | Load a sub-range of the tensor        | `Tensor`            |

---

**Package:** `safetensors`
**PyTorch Integration:** `safetensors.torch`
**Official Docs:** [huggingface.co/docs/safetensors](https://huggingface.co/docs/safetensors)
**GitHub:** [huggingface/safetensors](https://github.com/huggingface/safetensors)

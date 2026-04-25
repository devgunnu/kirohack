# Installation Guide

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Git
- (Optional) NVIDIA GPU with CUDA for model inference

## Setup

### 1. Clone the Repository

```powershell
git clone <repository-url>
cd meshrun
```

### 2. Create Virtual Environment

```powershell
uv venv .venv
```

### 3. Activate Virtual Environment

```powershell
# PowerShell (Windows)
.venv\Scripts\activate

# Bash (Linux/macOS)
source .venv/bin/activate
```

### 4. Install the Package

```powershell
uv pip install -e .
```

### 5. Install Development Dependencies

```powershell
uv pip install pytest hypothesis pytest-asyncio ruff mypy
```

### 6. (Optional) Install PyTorch for GPU Inference

PyTorch is required for model loading and forward pass execution (Shard Manager, Layer Engine, Resource Monitor GPU polling). The TCP protocol and connection pool work without it.

```powershell
# CPU-only
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.x
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Verify Installation

```powershell
python -c "import meshrun; print('MeshRun installed successfully')"
```

## Project Structure

```
meshrun/
  app/           # Application entry points (not yet implemented)
  coordinator/   # Control plane — gRPC registry, scheduling, routing (not yet implemented)
  worker/        # Data plane — TCP protocol, shard manager, layer engine, etc.
    protocol.py           # TCP binary protocol (header, framing, tensor serialization)
    connection_pool.py    # Persistent TCP connection management
    shard_manager.py      # Safetensors selective download, GPU loading, caching
    layer_engine.py       # Forward pass execution through transformer layers
    resource_monitor.py   # GPU memory/utilization tracking, heartbeat snapshots
    test_protocol.py      # Tests for protocol.py
  security/      # Security utilities (out of scope for POC)
  __init__.py
```

## Dependencies Summary

| Category        | Packages                                  | Required For                             |
| --------------- | ----------------------------------------- | ---------------------------------------- |
| Core            | Python stdlib (socket, struct, threading) | TCP protocol, connection pool            |
| Model Execution | PyTorch                                   | Shard loading, forward pass, GPU metrics |
| Control Plane   | grpcio, protobuf                          | Coordinator communication (not yet used) |
| Testing         | pytest, hypothesis                        | Unit and property-based tests            |
| Dev Tools       | ruff, mypy                                | Linting, type checking                   |

## Troubleshooting

**`python` command not found on Windows**: Use `python` (not `python3`). Ensure Python 3.13+ is on your PATH.

**uv not found**: Install uv following the [official guide](https://docs.astral.sh/uv/getting-started/installation/).

**Import errors after install**: Make sure the virtual environment is activated (`.venv\Scripts\activate`) before running any Python commands.

**PyTorch import errors in Shard Manager / Layer Engine / Resource Monitor**: These components require PyTorch. Install it per step 6 above. The TCP protocol and connection pool work without PyTorch.

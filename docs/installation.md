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

### 6. Install Security Dependencies

The secure protocol layer (AES-256-GCM encryption) requires the `cryptography` package:

```powershell
uv pip install cryptography
```

### 7. (Optional) Install PyTorch for GPU Inference

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
  app/               # Application entry points (not yet implemented)
  coordinator/       # Control plane — gRPC registry, scheduling, routing (not yet implemented)
  worker/            # Data plane — all worker node components
    protocol.py              # TCP binary protocol (header, framing, tensor serialization, encryption)
    connection_pool.py       # Persistent TCP connection management
    shard_manager.py         # Safetensors selective download, GPU loading, caching
    layer_engine.py          # Forward pass execution through transformer layers
    resource_monitor.py      # GPU memory/utilization tracking, heartbeat snapshots
    layer_registry.py        # Layer assignment storage and pipeline topology queries
    coordinator_client.py    # gRPC client for Coordinator communication
    node.py                  # Worker node lifecycle (startup, registration, serving)
    serving.py               # Serving loop — request processing pipeline
    test_protocol.py         # Tests for protocol.py
    test_secure_protocol.py  # Tests for secure protocol (AES-GCM)
  security/          # Standalone security utilities
    crypto.py                # AES-256-GCM encryption/decryption helpers
  __init__.py
```

## Dependencies Summary

| Category        | Packages                                  | Required For                               |
| --------------- | ----------------------------------------- | ------------------------------------------ |
| Core            | Python stdlib (socket, struct, threading) | TCP protocol, connection pool              |
| Security        | cryptography                              | AES-256-GCM encryption for secure protocol |
| Model Execution | PyTorch                                   | Shard loading, forward pass, GPU metrics   |
| Control Plane   | grpcio, protobuf                          | Coordinator communication (planned)        |
| Testing         | pytest, hypothesis                        | Unit and property-based tests              |
| Dev Tools       | ruff, mypy                                | Linting, type checking                     |

## Troubleshooting

**`python` command not found on Windows**: Use `python` (not `python3`). Ensure Python 3.13+ is on your PATH.

**uv not found**: Install uv following the [official guide](https://docs.astral.sh/uv/getting-started/installation/).

**Import errors after install**: Make sure the virtual environment is activated (`.venv\Scripts\activate`) before running any Python commands.

**PyTorch import errors in Shard Manager / Layer Engine / Resource Monitor**: These components require PyTorch. Install it per step 7 above. The TCP protocol and connection pool work without PyTorch.

**cryptography import errors**: Install via `uv pip install cryptography`. Required for `write_message_secure` / `read_message_secure` and the `meshrun/security/crypto.py` module.

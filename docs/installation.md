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

This installs all core dependencies defined in `pyproject.toml`:

- `grpcio` and `grpcio-tools` — gRPC for control plane communication
- `cryptography` — AES-256-GCM encryption for secure data plane
- `torch` — PyTorch for model loading and forward pass execution
- `transformers` — HuggingFace tokenizer loading

### 5. Install Development Dependencies

```powershell
uv pip install pytest hypothesis pytest-asyncio ruff mypy
```

### 6. (Optional) Install PyTorch for GPU Inference

The default `torch` install from `pyproject.toml` may default to CPU. For GPU support:

```powershell
# CPU-only (already included via pyproject.toml)
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
  app/               # Application entry points and CLI
  client/            # Inference Client
    client.py                # End-to-end inference orchestration
    tokenizer.py             # HuggingFace tokenizer + embedding engine
    transport.py             # Encrypted TCP transport (AES-256-GCM)
  coordinator/       # Coordinator Server (control plane)
    server.py                # gRPC server and servicer implementation
    registry.py              # Node registry and health tracker
    scheduler.py             # Layer assignment, route building, priority queue
    key_manager.py           # AES-256 session key lifecycle
    proto/                   # Protobuf definitions and generated stubs
      coordinator.proto      # gRPC service definition
      coordinator_pb2.py     # Generated protobuf messages
      coordinator_pb2_grpc.py # Generated gRPC stubs
  worker/            # Worker Node (data plane)
    protocol.py              # TCP binary protocol + AES-256-GCM encryption
    connection_pool.py       # Persistent TCP connection management
    shard_manager.py         # Safetensors selective download, GPU loading
    layer_engine.py          # Forward pass execution
    resource_monitor.py      # GPU memory/utilization tracking
    layer_registry.py        # Layer assignment storage
    coordinator_client.py    # gRPC client for Coordinator communication
    node.py                  # Worker node lifecycle orchestration
    serving.py               # Serving loop — encrypted request processing
    test_protocol.py         # Tests for protocol.py
    test_secure_protocol.py  # Tests for secure protocol (AES-GCM)
  security/          # Standalone security utilities
    crypto.py                # AES-256-GCM encryption/decryption helpers
  __init__.py
```

## Dependencies Summary

| Category        | Packages                                  | Required For                                   |
| --------------- | ----------------------------------------- | ---------------------------------------------- |
| Core            | Python stdlib (socket, struct, threading) | TCP protocol, connection pool                  |
| Security        | cryptography                              | AES-256-GCM encryption for secure protocol     |
| Model Execution | torch                                     | Shard loading, forward pass, GPU metrics       |
| Tokenization    | transformers                              | HuggingFace AutoTokenizer for inference client |
| Control Plane   | grpcio, grpcio-tools, protobuf            | Coordinator gRPC server and worker gRPC client |
| Testing         | pytest, hypothesis                        | Unit and property-based tests                  |
| Dev Tools       | ruff, mypy                                | Linting, type checking                         |

## Regenerating gRPC Stubs

If you modify `meshrun/coordinator/proto/coordinator.proto`, regenerate the Python stubs:

```powershell
python -m grpc_tools.protoc `
  -I meshrun/coordinator/proto `
  --python_out=meshrun/coordinator/proto `
  --grpc_python_out=meshrun/coordinator/proto `
  meshrun/coordinator/proto/coordinator.proto
```

## Troubleshooting

**`python` command not found on Windows**: Use `python` (not `python3`). Ensure Python 3.13+ is on your PATH.

**uv not found**: Install uv following the [official guide](https://docs.astral.sh/uv/getting-started/installation/).

**Import errors after install**: Make sure the virtual environment is activated (`.venv\Scripts\activate`) before running any Python commands.

**PyTorch import errors**: Required for Shard Manager, Layer Engine, Resource Monitor, and Inference Client. The TCP protocol and connection pool work without PyTorch.

**transformers import errors**: Required for the Inference Client's tokenizer. Install via `uv pip install transformers`.

**gRPC import errors**: Required for Coordinator Server and Worker Node's Coordinator Client. Install via `uv pip install grpcio grpcio-tools`.

**cryptography import errors**: Required for `write_message_secure` / `read_message_secure` and the `meshrun/security/crypto.py` module.

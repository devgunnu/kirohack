# Usage Guide

This guide covers running MeshRun worker nodes, configuration options, and operational patterns.

## Prerequisites

Complete the [Installation Guide](installation.md) first. You need:

- Python 3.13+ with virtual environment activated
- PyTorch installed (for GPU inference)
- `cryptography` package installed (for secure protocol)

## Running a Worker Node

### Step-by-Step Startup

```python
from meshrun.worker.node import WorkerNode, NodeConfig

# Configure the node
config = NodeConfig(
    address="0.0.0.0:9000",              # TCP data plane listen address
    grpc_address="0.0.0.0:50051",        # gRPC address for Coordinator
    coordinator_address="10.0.0.1:50051", # Coordinator's gRPC address
    gpu_memory_limit_mb=6000,             # Max GPU memory this node may use
    poll_interval_s=1.0,                  # Resource monitor polling interval
)

node = WorkerNode(config)

# 1. Initialize — queries GPU, generates node_id
capacity = node.startup()
print(f"Node {node.node_id}: {capacity.gpu_memory_free_mb} MB free")

# 2. Register with Coordinator
node.register_with_coordinator()

# 3. Accept layer assignment (normally received from Coordinator)
node.accept_layer_assignment(
    model_id="llama-3b",
    model_url="https://huggingface.co/model/resolve/main/model.safetensors",
    layer_start=0,
    layer_end=9,
    dtype=1,  # FP16
    is_final_node=False,
    downstream_node="192.168.1.11:9000",
    upstream_nodes=[],
)

# 4. Build engine and start serving
node.build_engine_and_serve(hidden_dim=4096)
```

### Automated Lifecycle

For production-like usage, `run_lifecycle()` handles the full startup sequence:

```python
node = WorkerNode(config)
node.run_lifecycle(
    model_id="llama-3b",
    model_url="https://huggingface.co/model/resolve/main/model.safetensors",
    layer_start=0,
    layer_end=9,
    dtype=1,
    is_final_node=False,
    downstream_node="192.168.1.11:9000",
    upstream_nodes=[],
    hidden_dim=4096,
)
```

## Configuration Reference

### NodeConfig

| Parameter             | Type  | Default | Description                                 |
| --------------------- | ----- | ------- | ------------------------------------------- |
| `address`             | str   | —       | TCP data plane listen address (host:port)   |
| `grpc_address`        | str   | —       | gRPC address for Coordinator communication  |
| `coordinator_address` | str   | —       | Coordinator's gRPC endpoint                 |
| `gpu_memory_limit_mb` | int   | —       | Maximum GPU memory this node may use (MB)   |
| `poll_interval_s`     | float | 1.0     | Resource monitor polling interval (seconds) |

### Layer Assignment Parameters

| Parameter         | Type      | Description                                   |
| ----------------- | --------- | --------------------------------------------- |
| `model_id`        | str       | Model identifier (e.g., "llama-3b")           |
| `model_url`       | str       | HTTP URL to safetensors model file            |
| `layer_start`     | int       | First assigned layer index (inclusive)        |
| `layer_end`       | int       | Last assigned layer index (inclusive)         |
| `dtype`           | int       | 1 = fp16, 2 = int8                            |
| `is_final_node`   | bool      | True if this node hosts the last model layers |
| `downstream_node` | str/None  | TCP host:port of next node (None if final)    |
| `upstream_nodes`  | list[str] | TCP addresses that may send data to this node |

## Multi-Node Pipeline Example

A 3-node pipeline serving a 30-layer model:

```
Node A (layers 0-9)  →  Node B (layers 10-19)  →  Node C (layers 20-29)
```

### Node A (First Node)

```python
NodeConfig(address="192.168.1.10:9000", ...)
# layer_start=0, layer_end=9, is_final_node=False
# downstream_node="192.168.1.11:9000"
```

### Node B (Middle Node)

```python
NodeConfig(address="192.168.1.11:9000", ...)
# layer_start=10, layer_end=19, is_final_node=False
# downstream_node="192.168.1.12:9000"
# upstream_nodes=["192.168.1.10:9000"]
```

### Node C (Final Node)

```python
NodeConfig(address="192.168.1.12:9000", ...)
# layer_start=20, layer_end=29, is_final_node=True
# downstream_node=None
# upstream_nodes=["192.168.1.11:9000"]
```

## Memory Budget Planning

Each node operates within a user-configured memory budget. Plan your allocation:

| Budget Category     | Typical Size  | Notes                                     |
| ------------------- | ------------- | ----------------------------------------- |
| Framework Overhead  | ~500-800 MB   | PyTorch/CUDA context (fixed)              |
| Model Shard (fp16)  | ~200 MB/layer | 10 layers ≈ 2 GB                          |
| Model Shard (int8)  | ~100 MB/layer | 10 layers ≈ 1 GB                          |
| Activation Memory   | ~50-200 MB    | Per in-flight request, depends on seq_len |
| KV-Cache (optional) | Variable      | Grows with sequence length                |

**Capacity check**: `gpu_memory_limit_mb - framework_overhead >= shard_memory + min_activation_memory`

## Monitoring

The Resource Monitor tracks GPU metrics and provides heartbeat snapshots:

```python
# Check current GPU usage
metrics = node.resource_monitor.get_latest_metrics()
print(f"GPU memory: {metrics.gpu_memory_used_mb}/{metrics.gpu_memory_total_mb} MB")
print(f"GPU utilization: {metrics.gpu_utilization:.1%}")

# Check if over memory limit
if node.resource_monitor.is_over_limit():
    print("Warning: exceeding configured memory limit")

# Get serving stats
if node.serving_loop:
    stats = node.serving_loop.stats
    print(f"Requests: {stats.total_success} success, {stats.total_failures} failed")
```

## Development / Testing with StubCoordinatorClient

For local development without a running Coordinator, use the stub client:

```python
from meshrun.worker.coordinator_client import (
    StubCoordinatorClient, RegisterResponse, ConfirmReadyResponse,
    HeartbeatResponse, ReportFailureResponse,
)

client = StubCoordinatorClient(
    register_response=RegisterResponse(success=True, message="OK"),
    confirm_ready_response=ConfirmReadyResponse(success=True, message="OK"),
    heartbeat_response=HeartbeatResponse(success=True),
)

# Pass to WorkerNode for testing
config = NodeConfig(
    address="0.0.0.0:9000",
    grpc_address="0.0.0.0:50051",
    coordinator_address="stub",  # Not used with stub client
    gpu_memory_limit_mb=6000,
)
node = WorkerNode(config, coordinator_client=client)
```

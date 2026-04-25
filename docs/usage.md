# Usage Guide

This guide covers running the MeshRun distributed inference pipeline — Coordinator, Worker Nodes, and Inference Client.

## Prerequisites

Complete the [Installation Guide](installation.md) first. You need:

- Python 3.13+ with virtual environment activated
- All dependencies installed (`uv pip install -e .`)

## Running the Coordinator

The Coordinator is the central control plane. Start it before any worker nodes or clients.

```python
from meshrun.coordinator.server import CoordinatorServer

server = CoordinatorServer(
    host="0.0.0.0",
    port=50051,
    heartbeat_interval_s=5.0,
    missed_threshold=3,
    dead_threshold=5,
)

server.start()
print(f"Coordinator running on 0.0.0.0:50051")

# Access internal state
print(f"Registered nodes: {len(server.registry.get_all_nodes())}")

# Shutdown
server.stop(grace=5.0)
```

### Triggering Layer Assignment

After all worker nodes have registered, trigger layer assignment via gRPC:

```python
import grpc
from meshrun.coordinator.proto import coordinator_pb2 as pb2
from meshrun.coordinator.proto import coordinator_pb2_grpc as pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = pb2_grpc.CoordinatorServiceStub(channel)

response = stub.TriggerAssignment(pb2.TriggerAssignmentRequest(
    model_id="llama-3b",
    total_layers=30,
    dtype=pb2.DTYPE_FP16,
    model_url="https://huggingface.co/model/resolve/main/model.safetensors",
))
print(f"Assignment: {response.message}")
```

## Running a Worker Node

### Step-by-Step Startup

```python
from meshrun.worker.node import WorkerNode, NodeConfig

config = NodeConfig(
    address="0.0.0.0:9000",              # TCP data plane listen address
    grpc_address="0.0.0.0:50052",        # gRPC address for Coordinator callbacks
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

# 3. Accept layer assignment (normally pushed by Coordinator after TriggerAssignment)
node.accept_layer_assignment(
    model_id="llama-3b",
    model_url="https://huggingface.co/model/resolve/main/model.safetensors",
    layer_start=0,
    layer_end=9,
    dtype=1,  # FP16
    is_final_node=False,
    downstream_node="192.168.1.11:9000",
    upstream_nodes=[],
    session_key=b'\x00' * 32,  # Distributed by Coordinator
)

# 4. Build engine and start serving
node.build_engine_and_serve(hidden_dim=4096)
```

### Automated Lifecycle

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

## Running the Inference Client

```python
from meshrun.client.client import InferenceClient

client = InferenceClient(
    coordinator_address="10.0.0.1:50051",
    model_name="meta-llama/Llama-3.2-3B",
    model_url="https://huggingface.co/model/resolve/main/model.safetensors",
    cache_dir="./cache",
    device="cpu",
)

# One-time setup: load tokenizer and embedding weights
client.initialize()

# Run inference
output = client.submit_inference("What is artificial intelligence?")
print(output)

# Cleanup
client.close()
```

See [Inference Client](client.md) for detailed API documentation.

## Configuration Reference

### CoordinatorServer

| Parameter              | Type  | Default | Description                        |
| ---------------------- | ----- | ------- | ---------------------------------- |
| `host`                 | str   | —       | gRPC listen address                |
| `port`                 | int   | —       | gRPC listen port                   |
| `heartbeat_interval_s` | float | 5.0     | Health check interval (seconds)    |
| `missed_threshold`     | int   | 3       | Missed heartbeats before UNHEALTHY |
| `dead_threshold`       | int   | 5       | Missed heartbeats before DEAD      |

### NodeConfig

| Parameter             | Type  | Default | Description                                 |
| --------------------- | ----- | ------- | ------------------------------------------- |
| `address`             | str   | —       | TCP data plane listen address (host:port)   |
| `grpc_address`        | str   | —       | gRPC address for Coordinator callbacks      |
| `coordinator_address` | str   | —       | Coordinator's gRPC endpoint                 |
| `gpu_memory_limit_mb` | int   | —       | Maximum GPU memory this node may use (MB)   |
| `poll_interval_s`     | float | 1.0     | Resource monitor polling interval (seconds) |

### InferenceClient

| Parameter             | Type | Default | Description                               |
| --------------------- | ---- | ------- | ----------------------------------------- |
| `coordinator_address` | str  | —       | Coordinator's gRPC endpoint               |
| `model_name`          | str  | —       | HuggingFace model ID for tokenizer        |
| `model_url`           | str  | —       | HTTP URL to safetensors model file        |
| `cache_dir`           | str  | —       | Local directory for caching weights       |
| `device`              | str  | "cpu"   | Target device for embedding (cpu or cuda) |

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
| `session_key`     | bytes     | 32-byte AES-256 key from Coordinator          |

## Multi-Node Pipeline Example

A 3-node pipeline serving a 30-layer model:

```
Coordinator (gRPC :50051)
    ↓ Register / Heartbeat / Assignment
Node A (layers 0-9, TCP :9000)  →  Node B (layers 10-19, TCP :9001)  →  Node C (layers 20-29, TCP :9002)
    ↑                                                                          ↓
Client ─── gRPC route request ──→ Coordinator                          Client ←── encrypted logits
       ─── encrypted hidden states ──→ Node A ──→ ... ──→ Node C ──→ Client
```

### Node A (First Node)

```python
NodeConfig(address="192.168.1.10:9000", ...)
# layer_start=0, layer_end=9, is_final_node=False
# downstream_node="192.168.1.11:9001"
```

### Node B (Middle Node)

```python
NodeConfig(address="192.168.1.11:9001", ...)
# layer_start=10, layer_end=19, is_final_node=False
# downstream_node="192.168.1.12:9002"
```

### Node C (Final Node)

```python
NodeConfig(address="192.168.1.12:9002", ...)
# layer_start=20, layer_end=29, is_final_node=True
# downstream_node=None
```

## Memory Budget Planning

| Budget Category     | Typical Size  | Notes                                     |
| ------------------- | ------------- | ----------------------------------------- |
| Framework Overhead  | ~500-800 MB   | PyTorch/CUDA context (fixed)              |
| Model Shard (fp16)  | ~200 MB/layer | 10 layers ≈ 2 GB                          |
| Model Shard (int8)  | ~100 MB/layer | 10 layers ≈ 1 GB                          |
| Activation Memory   | ~50-200 MB    | Per in-flight request, depends on seq_len |
| KV-Cache (optional) | Variable      | Grows with sequence length                |

**Capacity check**: `gpu_memory_limit_mb - framework_overhead >= shard_memory + min_activation_memory`

## Monitoring

```python
# Worker node GPU metrics
metrics = node.resource_monitor.get_latest_metrics()
print(f"GPU memory: {metrics.gpu_memory_used_mb}/{metrics.gpu_memory_total_mb} MB")
print(f"GPU utilization: {metrics.gpu_utilization:.1%}")

# Serving stats
if node.serving_loop:
    stats = node.serving_loop.stats
    print(f"Requests: {stats.total_success} success, {stats.total_failures} failed")

# Coordinator registry
nodes = server.registry.get_all_healthy_nodes()
print(f"Healthy nodes: {len(nodes)}")
```

## Development / Testing with StubCoordinatorClient

For local development without a running Coordinator:

```python
from meshrun.worker.coordinator_client import (
    StubCoordinatorClient, RegisterResponse, ConfirmReadyResponse,
    HeartbeatResponse,
)

client = StubCoordinatorClient(
    register_response=RegisterResponse(success=True, message="OK"),
    confirm_ready_response=ConfirmReadyResponse(success=True, message="OK"),
    heartbeat_response=HeartbeatResponse(success=True),
)

node = WorkerNode(config, coordinator_client=client)
```

# Worker Node Components

All worker node sub-components live under `meshrun/worker/`. For the node lifecycle orchestration (startup, registration, serving), see [Worker Node Lifecycle](worker-lifecycle.md).

## Connection Pool

Source: `meshrun/worker/connection_pool.py`

Manages persistent TCP connections to downstream nodes. Connections are established once and reused across requests.

### Key Classes

- `ConnectionPool` — Main pool managing outbound and inbound TCP connections
- `ConnectionInfo` — Immutable metadata for a single connection (state, socket, timestamps)
- `ConnectionState` — Lifecycle enum: DISCONNECTED → CONNECTING → CONNECTED → FAILED

### Configuration Constants

| Constant                           | Default | Description                       |
| ---------------------------------- | ------- | --------------------------------- |
| `CONNECTION_ESTABLISHMENT_TIMEOUT` | 5.0s    | TCP connect timeout               |
| `IDLE_CONNECTION_TIMEOUT`          | 30.0s   | Idle connection cleanup threshold |
| `MAX_RETRY_ATTEMPTS`               | 1       | Retries before reporting failure  |

### Usage

```python
from meshrun.worker.connection_pool import ConnectionPool

pool = ConnectionPool()

# Get or establish a connection
sock = pool.get_connection(("192.168.1.10", 9000))

# Check if connected
if pool.is_connected(("192.168.1.10", 9000)):
    pass

# Start accepting incoming connections
def handle_connection(sock, addr):
    # Process incoming data
    pass

pool.accept_incoming(("0.0.0.0", 9000), on_connection=handle_connection)

# Cleanup
pool.close_connection(("192.168.1.10", 9000))
pool.close_all()  # Shutdown
```

### Connection Lifecycle

```
Disconnected → Connecting (get_connection called)
Connecting → Connected (TCP handshake success)
Connecting → Failed (timeout / refused)
Connected → Failed (connection lost)
Failed → Connecting (single retry)
Failed → Disconnected (retry exhausted)
Connected → Disconnected (close_connection called)
```

---

## Shard Manager

Source: `meshrun/worker/shard_manager.py`

Manages the lifecycle of quantized model shards — selectively downloading only assigned layer weights from a safetensors model file via HTTP Range requests, caching locally, loading to GPU, validating, and unloading.

### Key Classes

- `ShardMetadata` — Mutable state tracking model_id, layer range, dtype, load status, loaded tensors
- `ShardInfo` — Immutable snapshot for external consumers
- `TensorInfo` — Metadata for a single tensor from the safetensors header
- `LayerRange` — Immutable value type for contiguous layer range (start, end inclusive)
- `LoadStatus` — Lifecycle enum: UNLOADED → DOWNLOADING → LOADING → READY (or ERROR)
- `ShardDType` — Quantization format: FP16 or INT8

### Core Functions

| Function                                              | Description                                                |
| ----------------------------------------------------- | ---------------------------------------------------------- |
| `fetch_safetensors_header(url)`                       | Fetch and parse safetensors header via HTTP Range requests |
| `filter_tensors_for_assignment(tensors, layer_range)` | Filter tensors to only those needed by this node           |
| `download_selected_tensors_cached(url, ...)`          | Download tensors with local caching (cache-first)          |
| `load_shard(metadata)`                                | Full orchestration: download → deserialize → GPU load      |
| `validate_shard(metadata, assigned_tensors)`          | Verify layer count, dtype, hidden dimension consistency    |
| `unload_shard(metadata)`                              | Free GPU memory, reset state to UNLOADED                   |
| `get_shard_info(metadata)`                            | Return immutable snapshot of shard state                   |

### Weight Download Strategy

1. Fetch safetensors header via HTTP Range request (first 8 bytes → header size, then header JSON)
2. Parse tensor metadata to identify tensors for assigned layers (e.g., `model.layers.5.*` through `model.layers.9.*`)
3. Download only those tensor byte ranges using HTTP Range requests
4. Cache downloaded weights locally (`cache_dir/{model_id}/{tensor_name}.bin`)
5. Load cached weights into GPU memory via PyTorch

This approach is provider-agnostic: works with HuggingFace Hub, S3, GCS, or any HTTP server supporting Range requests.

### Usage

```python
from meshrun.worker.shard_manager import (
    ShardMetadata, LayerRange, ShardDType, LoadStatus,
    load_shard, validate_shard, unload_shard, get_shard_info,
)
from pathlib import Path

metadata = ShardMetadata(
    model_id="llama-3b",
    model_url="https://huggingface.co/model/resolve/main/model.safetensors",
    layer_range=LayerRange(start=5, end=9),
    dtype=ShardDType.FP16,
    cache_dir=Path("./cache"),
)

# Load shard (download → deserialize → GPU)
metadata = load_shard(metadata, device="cuda")
assert metadata.load_status == LoadStatus.READY

# Get info snapshot
info = get_shard_info(metadata)
print(f"Loaded {info.layer_start}-{info.layer_end}, {info.memory_footprint_mb:.1f} MB")

# Unload
metadata = unload_shard(metadata)
assert metadata.load_status == LoadStatus.UNLOADED
```

---

## Layer Engine

Source: `meshrun/worker/layer_engine.py`

Executes sequential forward passes through hosted transformer layers. Supports both intermediate hidden state output and final logits output (when the node hosts the last layers).

### Key Classes

- `TransformerLayer` — Holds weight tensors for a single transformer layer (attention Q/K/V/O projections, MLP gate/up/down projections, RMSNorm weights)
- `LayerEngine` — Stateful engine holding an ordered list of transformer layers, optional LM head, and configuration

### Core Functions

| Function                                     | Description                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------- |
| `build_layer_engine(loaded_tensors, ...)`    | Construct a LayerEngine from loaded shard tensors (groups by layer index) |
| `forward(engine, hidden_states, step_id)`    | Sequential forward pass through all hosted layers, returns output tensor  |
| `warm_up(engine, hidden_dim, device, dtype)` | Run dummy forward pass to pre-allocate GPU kernels and activation memory  |

### Forward Pass Details

- Layers are executed sequentially in order (layer_start through layer_end)
- Each layer applies: RMSNorm → Attention → Residual → RMSNorm → MLP → Residual
- If `is_final_node` is true, applies a final RMSNorm + LM head linear projection to produce logits
- Output is validated for NaN/Inf values

### Usage

```python
from meshrun.worker.layer_engine import build_layer_engine, forward, warm_up

# Build engine from loaded shard tensors
engine = build_layer_engine(
    loaded_tensors=shard_metadata.loaded_tensors,
    layer_start=5,
    layer_end=9,
    is_final_node=False,
    device="cuda",
)

# Warm up GPU kernels
warm_up(engine, hidden_dim=4096, device="cuda", dtype="float16")

# Run forward pass
output = forward(engine, hidden_states_tensor, step_id=0)
```

---

## Resource Monitor

Source: `meshrun/worker/resource_monitor.py`

Tracks GPU memory, compute utilization, and active request count. Provides heartbeat snapshots and memory limit alerting. Memory limits are user-configured — the monitor observes only, it does not adjust allocations.

### Key Classes

- `GpuMetrics` — Immutable snapshot of GPU state (total/used/free memory, utilization)
- `HeartbeatSnapshot` — Subset of metrics for inclusion in heartbeat messages
- `ResourceMonitor` — Stateful monitor with background polling thread

### Metrics Tracked

| Metric                 | Type  | Description                                |
| ---------------------- | ----- | ------------------------------------------ |
| `gpu_memory_total_mb`  | int   | Total GPU memory available                 |
| `gpu_memory_used_mb`   | int   | Current GPU memory in use                  |
| `gpu_memory_free_mb`   | int   | Remaining GPU memory                       |
| `gpu_memory_limit_mb`  | int   | User-configured memory limit               |
| `gpu_utilization`      | float | GPU compute utilization (0.0-1.0)          |
| `active_requests`      | int   | Number of in-flight forward passes         |
| `shard_memory_mb`      | int   | Memory consumed by loaded model shard      |
| `activation_memory_mb` | int   | Estimated memory for in-flight activations |

### Usage

```python
from meshrun.worker.resource_monitor import ResourceMonitor

monitor = ResourceMonitor(
    gpu_memory_limit_mb=6000,
    poll_interval_s=1.0,
    device_index=0,
)

# Start background polling
monitor.start()

# Get current metrics
metrics = monitor.get_latest_metrics()
print(f"GPU used: {metrics.gpu_memory_used_mb} MB")

# Get heartbeat snapshot
snapshot = monitor.get_heartbeat_snapshot()

# Track active requests
monitor.increment_active_requests()
monitor.decrement_active_requests()

# Check memory limit
if monitor.is_over_limit():
    print("Warning: GPU memory exceeds configured limit")

# Stop polling
monitor.stop()
```

---

## Layer Assignment Registry

Source: `meshrun/worker/layer_registry.py`

Thread-safe registry that stores the current layer assignment for a worker node. Provides query methods for other sub-components to look up pipeline topology, layer ranges, and downstream/upstream addresses.

### Key Classes

- `AssignmentDType` — Quantization format enum: FP16 or INT8
- `LayerAssignment` — Immutable snapshot of a layer assignment (node_id, model_id, model_url, layer range, dtype, topology)
- `LayerAssignmentRegistry` — Thread-safe registry storing exactly one assignment at a time

### LayerAssignment Fields

| Field           | Type            | Description                               |
| --------------- | --------------- | ----------------------------------------- |
| node_id         | str (UUID)      | This node's identifier                    |
| model_id        | str             | Model being served                        |
| model_url       | str (URL)       | HTTP URL to safetensors file              |
| layer_start     | int             | First assigned layer (inclusive)          |
| layer_end       | int             | Last assigned layer (inclusive)           |
| dtype           | AssignmentDType | fp16 or int8                              |
| is_final_node   | bool            | Whether this node hosts the last layers   |
| downstream_node | Optional[str]   | Next node `host:port` (None if final)     |
| upstream_nodes  | tuple[str, ...] | Addresses that may send data to this node |

### Validation Rules

- `node_id`, `model_id`, `model_url` must be non-empty
- `layer_start` >= 0, `layer_end` >= `layer_start`
- `dtype` must be a valid `AssignmentDType`
- If `is_final_node` is True, `downstream_node` must be None
- If `is_final_node` is False, `downstream_node` is required

### Usage

```python
from meshrun.worker.layer_registry import (
    LayerAssignmentRegistry, LayerAssignment, AssignmentDType,
)

registry = LayerAssignmentRegistry()

# Store assignment from Coordinator
assignment = LayerAssignment(
    node_id="node-abc-123",
    model_id="llama-3b",
    model_url="https://example.com/model.safetensors",
    layer_start=5,
    layer_end=9,
    dtype=AssignmentDType.FP16,
    is_final_node=False,
    downstream_node="192.168.1.11:9000",
    upstream_nodes=("192.168.1.9:9000",),
)
registry.accept_layer_assignment(assignment)

# Query from other components
downstream = registry.get_downstream_address()  # "192.168.1.11:9000"
layer_range = registry.get_layer_range()         # (5, 9)
is_final = registry.is_final_node()              # False

# Clear on shutdown
registry.clear()
```

# Worker Node Components

All worker node sub-components live under `meshrun/worker/`.

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
    # Connection is live
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

### Key Classes and Functions

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

## Layer Engine (Not Yet Implemented)

Will be located at: `meshrun/worker/layer_engine.py`

Executes sequential forward passes through hosted transformer layers.

### Planned Interface

| Method    | Input                  | Output                    | Description                                       |
| --------- | ---------------------- | ------------------------- | ------------------------------------------------- |
| `Forward` | hidden_states, step_id | hidden_states (or logits) | Sequential forward pass through all hosted layers |
| `WarmUp`  | dummy tensor           | —                         | Pre-allocate GPU kernels and activation memory    |

---

## Resource Monitor (Not Yet Implemented)

Will be located at: `meshrun/worker/resource_monitor.py`

Tracks GPU memory, compute utilization, and active request count.

### Planned Metrics

| Metric              | Type    | Description                       |
| ------------------- | ------- | --------------------------------- |
| gpu_memory_total_mb | uint32  | Total GPU memory                  |
| gpu_memory_used_mb  | uint32  | Current GPU memory in use         |
| gpu_memory_free_mb  | uint32  | Remaining GPU memory              |
| gpu_memory_limit_mb | uint32  | User-configured memory limit      |
| gpu_utilization     | float32 | GPU compute utilization (0.0-1.0) |
| active_requests     | uint16  | In-flight forward passes          |

---

## Layer Assignment Registry (Not Yet Implemented)

Will be located at: `meshrun/worker/layer_registry.py`

Stores the current layer assignment and provides pipeline topology information.

### Planned Fields

| Field           | Type               | Description                               |
| --------------- | ------------------ | ----------------------------------------- |
| node_id         | string (UUID)      | This node's identifier                    |
| model_id        | string             | Model being served                        |
| model_url       | string (URL)       | HTTP URL to safetensors file              |
| layer_start     | uint16             | First assigned layer (inclusive)          |
| layer_end       | uint16             | Last assigned layer (inclusive)           |
| dtype           | enum               | fp16 or int8                              |
| is_final_node   | bool               | Whether this node hosts the last layers   |
| downstream_node | string (host:port) | Next node in pipeline (null if final)     |
| upstream_nodes  | list of string     | Addresses that may send data to this node |

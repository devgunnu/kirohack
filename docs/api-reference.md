# API Reference

## meshrun.worker.protocol

TCP binary protocol implementation: header serialization, reliable framing, and tensor serialization.

### Constants

| Name                   | Value                | Description                                 |
| ---------------------- | -------------------- | ------------------------------------------- |
| `HEADER_SIZE`          | 32                   | Every serialized header is exactly 32 bytes |
| `HEADER_STRUCT_FORMAT` | `<BIIIBB4IB`         | Little-endian struct format for the header  |
| `MAX_DIMS`             | 4                    | Maximum number of tensor dimensions         |
| `DTYPE_SIZE`           | `{FP16: 2, INT8: 1}` | Byte size per element for each dtype        |

### Enums

**MessageType(IntEnum)**
- `FORWARD = 1` — Hidden states flowing through pipeline
- `RESULT = 2` — Final logits returned to client
- `ERROR = 3` — Error response
- `HEARTBEAT_DATA = 4` — Data-plane heartbeat

**DType(IntEnum)**
- `FP16 = 1` — IEEE 754 half-precision (2 bytes/element)
- `INT8 = 2` — Signed 8-bit integer (1 byte/element)

### Header

```python
@dataclass(frozen=True, slots=True)
class Header:
    message_type: int    # uint8, must be valid MessageType
    request_id: int      # uint32
    step_id: int         # uint32
    payload_size: int    # uint32
    dtype: int           # uint8, must be valid DType
    num_dims: int        # uint8, range [1, 4]
    dims: tuple[int, int, int, int] = (0, 0, 0, 0)  # uint32[4]
    reserved: int = 0    # uint8
```

**Methods:**
- `validate() -> None` — Validate all fields per protocol spec. Raises `ValueError`.
- `pack() -> bytes` — Serialize to exactly 32 bytes (little-endian).
- `Header.unpack(data: bytes) -> Header` — Deserialize 32 bytes, auto-validates. Raises `ValueError`.

### Functions

**`read_exact(sock, n) -> bytes`**
Read exactly `n` bytes from a TCP socket, looping on `recv()`. Raises `ConnectionError` on EOF, propagates `socket.timeout`.

**`write_all(sock, data) -> None`**
Write all bytes to a TCP socket, looping on `send()`. Raises `ConnectionError` if `send()` returns 0, propagates `socket.timeout`.

**`tensor_to_bytes(elements, dtype) -> bytes`**
Serialize a flat list of numeric values to raw contiguous bytes. FP16 uses little-endian IEEE 754 half-precision. INT8 uses signed 8-bit. Raises `ValueError` for invalid dtype or out-of-range values.

**`bytes_to_tensor(data, dtype, dims, num_dims) -> list`**
Deserialize raw bytes to a flat list of numeric values. Inverse of `tensor_to_bytes`. Raises `ValueError` for mismatched sizes or invalid dtype.

**`read_message(sock) -> tuple[Header, list]`**
Read a complete message: header (32 bytes) + payload. Returns `(header, tensor_data)`.

**`write_message(sock, header, tensor_data) -> None`**
Write a complete message: validates tensor length, serializes, writes header + payload.

---

## meshrun.worker.connection_pool

Persistent TCP connection management for the data plane.

### Enums

**ConnectionState(IntEnum)**
- `DISCONNECTED = 0`
- `CONNECTING = 1`
- `CONNECTED = 2`
- `FAILED = 3`

### ConnectionInfo

```python
@dataclass(frozen=True, slots=True)
class ConnectionInfo:
    target_addr: tuple[str, int]
    state: ConnectionState
    socket: Optional[socket.socket] = None
    last_activity: float = <current time>
    retry_count: int = 0
    error_message: Optional[str] = None
```

### ConnectionPool

**`__init__() -> None`** — Initialize empty pool.

**`get_connection(target_addr) -> Optional[socket.socket]`** — Get or establish TCP connection. Returns existing connection if available, establishes new one with 5s timeout otherwise. Single retry on failure. Returns `None` if connection fails.

**`close_connection(target_addr) -> None`** — Close and remove a specific connection. Raises `KeyError` if not found.

**`close_all() -> None`** — Close all connections, stop listener, clean up resources.

**`is_connected(target_addr) -> bool`** — Check if a live connection exists. Performs socket liveness probe.

**`accept_incoming(listen_addr, on_connection=None, backlog=5) -> None`** — Start TCP listener on a background thread. Calls `on_connection(sock, addr)` for each accepted connection. Raises `RuntimeError` if already listening.

**`get_incoming_connections() -> list[tuple[socket.socket, tuple[str, int]]]`** — Snapshot of all accepted incoming connections.

---

## meshrun.worker.shard_manager

Selective download and lifecycle management of model shards using safetensors format.

### Enums

**ShardDType(IntEnum)** — `FP16 = 1`, `INT8 = 2`

**LoadStatus(IntEnum)** — `UNLOADED = 0`, `DOWNLOADING = 1`, `LOADING = 2`, `READY = 3`, `ERROR = 4`

### Data Classes

**`LayerRange(start: int, end: int)`** — Immutable, both inclusive. Property: `count -> int`.

**`ShardMetadata`** — Mutable shard state: model_id, model_url, layer_range, dtype, cache_dir, memory_footprint_mb, load_status, bytes_downloaded, bytes_total, loaded_tensors, error_message. Property: `download_progress -> float`.

**`ShardInfo`** — Immutable snapshot: model_id, model_url, layer_start, layer_end, dtype, memory_footprint_mb, load_status, bytes_downloaded, bytes_total, download_progress.

**`TensorInfo`** — Immutable tensor metadata: name, dtype (string), shape, data_offset_start, data_offset_end. Property: `byte_size -> int`.

### Functions

**`fetch_safetensors_header(url) -> tuple[int, dict[str, TensorInfo]]`** — Two HTTP Range requests to fetch header size + JSON. Returns `(header_size, tensors_dict)`. Raises `SafetensorsHeaderError`.

**`filter_tensors_for_assignment(tensors, layer_range, is_first_node=False, is_final_node=False) -> dict[str, TensorInfo]`** — Filter to tensors matching `model.layers.{i}.*` for assigned range. Includes embedding tensors for first node, LM head for final node.

**`download_selected_tensors_cached(url, header_size, tensors, cache_dir, model_id) -> dict[str, bytes]`** — Cache-first download: check local cache, download on miss, save to cache.

**`load_shard(metadata, is_first_node=False, is_final_node=False, device="cuda") -> ShardMetadata`** — Full lifecycle: UNLOADED → DOWNLOADING → LOADING → READY (or ERROR).

**`validate_shard(metadata, assigned_tensors) -> ShardMetadata`** — Check layer count, dtype match, hidden dimension consistency. Sets ERROR on failure.

**`unload_shard(metadata, memory_freed_callback=None) -> ShardMetadata`** — Free GPU memory, reset to UNLOADED.

**`get_shard_info(metadata) -> ShardInfo`** — Immutable snapshot of current state.

### Exceptions

- `SafetensorsHeaderError` — Header fetch/parse failures
- `ShardValidationError` — Shard validation failures

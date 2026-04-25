# API Reference

## meshrun.worker.protocol

TCP binary protocol implementation: header serialization, reliable framing, tensor serialization, and AES-256-GCM encryption.

### Constants

| Name                   | Value                | Description                                 |
| ---------------------- | -------------------- | ------------------------------------------- |
| `HEADER_SIZE`          | 32                   | Every serialized header is exactly 32 bytes |
| `HEADER_STRUCT_FORMAT` | `<BIIIBB4IB`         | Little-endian struct format for the header  |
| `MAX_DIMS`             | 4                    | Maximum number of tensor dimensions         |
| `DTYPE_SIZE`           | `{FP16: 2, INT8: 1}` | Byte size per element for each dtype        |
| `NONCE_SIZE`           | 12                   | AES-GCM nonce size (96-bit)                 |
| `TAG_SIZE`             | 16                   | AES-GCM authentication tag size (128-bit)   |
| `WIRE_LEN_SIZE`        | 4                    | Length prefix size for encrypted messages   |

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

### Plaintext Functions

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

### Encryption Functions

**`generate_session_key() -> bytes`**
Generate a random 256-bit (32-byte) AES session key via `os.urandom(32)`.

**`encrypt_message(plaintext, key) -> bytes`**
Encrypt plaintext with AES-256-GCM. Returns `[12-byte nonce][ciphertext + 16-byte tag]`.

**`decrypt_message(encrypted_blob, key) -> bytes`**
Decrypt an AES-256-GCM encrypted blob. Raises `InvalidTag` on wrong key or tampered data.

**`write_message_secure(sock, header, tensor_data, key) -> None`**
Write an encrypted message. Validates, serializes, encrypts with fresh nonce, prepends 4-byte length prefix, writes via `write_all`. Wire format: `[4-byte len][12-byte nonce][ciphertext][16-byte tag]`.

**`read_message_secure(sock, key) -> tuple[Header, list]`**
Read and decrypt a message. Reads 4-byte length prefix, reads encrypted blob, decrypts, parses header + payload. Raises `InvalidTag` on decryption failure.

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

**`LayerRange(start: int, end: int)`** — Immutable, both inclusive. Property: `count -> int`. Validates `start >= 0` and `end >= start`.

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

---

## meshrun.worker.layer_engine

Forward pass execution through hosted transformer layers.

### Data Classes

**`TransformerLayer`** — Holds weight tensors for a single transformer layer: attention Q/K/V/O projections, MLP gate/up/down projections, input/post-attention RMSNorm weights.

**`LayerEngine`** — Stateful engine: ordered list of `TransformerLayer` instances, optional LM head weight, layer range, configuration (num_heads, head_dim, is_final_node).

### Functions

**`build_layer_engine(loaded_tensors, layer_start, layer_end, is_final_node, device) -> LayerEngine`** — Construct engine from loaded shard tensors. Groups tensors by layer index, infers head configuration, validates completeness.

**`forward(engine, hidden_states, step_id) -> Tensor`** — Sequential forward pass through all hosted layers. Applies RMSNorm → Attention → Residual → RMSNorm → MLP → Residual per layer. If final node, applies LM head to produce logits. Validates output for NaN/Inf.

**`warm_up(engine, hidden_dim, device, dtype) -> None`** — Run dummy forward pass to pre-allocate GPU kernels and activation memory.

---

## meshrun.worker.resource_monitor

GPU memory and utilization tracking with heartbeat snapshots.

### Data Classes

**`GpuMetrics`** — Immutable GPU state snapshot: gpu_memory_total_mb, gpu_memory_used_mb, gpu_memory_free_mb, gpu_utilization (0.0-1.0). Validates all values are non-negative and free + used <= total.

**`HeartbeatSnapshot`** — Subset for heartbeat messages: gpu_utilization, memory_used_mb, active_requests.

### ResourceMonitor

**`__init__(gpu_memory_limit_mb, poll_interval_s=1.0, device_index=0)`** — Initialize monitor with user-configured memory limit.

**`start() -> None`** — Start background polling thread.

**`stop() -> None`** — Stop background polling thread.

**`poll_once() -> GpuMetrics`** — Poll GPU metrics once (synchronous).

**`get_latest_metrics() -> Optional[GpuMetrics]`** — Return most recent polled metrics.

**`get_heartbeat_snapshot() -> HeartbeatSnapshot`** — Return current metrics for heartbeat.

**`increment_active_requests() -> int`** / **`decrement_active_requests() -> int`** — Thread-safe request counting.

**`set_shard_memory_mb(mb) -> None`** / **`set_activation_memory_mb(mb) -> None`** — Update memory tracking.

**`is_over_limit() -> bool`** — Check if GPU usage exceeds configured limit.

**Properties:** `gpu_memory_limit_mb`, `poll_interval_s`, `is_polling`, `active_requests`, `shard_memory_mb`, `activation_memory_mb`.

---

## meshrun.worker.layer_registry

Layer assignment storage and pipeline topology queries.

### Enums

**AssignmentDType(IntEnum)** — `FP16 = 1`, `INT8 = 2`

### Data Classes

**`LayerAssignment`** — Immutable assignment: node_id, model_id, model_url, layer_start, layer_end, dtype, is_final_node, downstream_node, upstream_nodes. Property: `layer_count -> int`. Validates all invariants on construction.

### LayerAssignmentRegistry

Thread-safe registry storing exactly one assignment at a time.

**`accept_layer_assignment(assignment) -> None`** — Store a new assignment (replaces previous).

**`clear() -> None`** — Remove current assignment.

**Properties:** `has_assignment -> bool`, `assignment -> Optional[LayerAssignment]`.

**Query methods:** `get_downstream_address()`, `get_upstream_addresses()`, `get_layer_range()`, `get_dtype()`, `is_final_node()`, `get_model_id()`, `get_model_url()`, `get_node_id()`.

---

## meshrun.worker.coordinator_client

gRPC client abstraction for Coordinator communication.

### Enums

**RegistrationStatus(IntEnum)** — `OK = 0`, `ALREADY_REGISTERED = 1`, `REJECTED = 2`

### Data Classes

- `CapacityInfo` — gpu_memory_total_mb, gpu_memory_free_mb, gpu_memory_limit_mb
- `RegisterRequest` — node_id, address, grpc_address, capacity
- `RegisterResponse` — success, message
- `ConfirmReadyRequest` — node_id, layers_loaded (list of ints)
- `ConfirmReadyResponse` — success, message
- `HeartbeatRequest` — node_id, gpu_utilization, memory_used_mb, active_requests
- `HeartbeatResponse` — success
- `ReportFailureRequest` — request_id, failed_node_id, node_id
- `RerouteInfo` — backup_node_id, backup_address
- `ReportFailureResponse` — success, reroute_info

### CoordinatorClient (Abstract Base)

**`register(request) -> RegisterResponse`**

**`confirm_ready(request) -> ConfirmReadyResponse`**

**`heartbeat(request) -> HeartbeatResponse`**

**`report_failure(request) -> ReportFailureResponse`**

**`close() -> None`**

### Implementations

- `GrpcCoordinatorClient(coordinator_address)` — Real gRPC client (pending proto generation)
- `StubCoordinatorClient(...)` — In-memory stub for testing, accepts pre-configured responses

---

## meshrun.worker.node

Worker node lifecycle orchestration.

### Enums

**NodeState(IntEnum)** — `INITIALIZING = 0`, `REGISTERING = 1`, `WAITING_ASSIGNMENT = 2`, `LOADING_SHARD = 3`, `VALIDATING = 4`, `READY = 5`, `SERVING = 6`, `DRAINING = 7`, `ERROR = 8`

### Data Classes

- `NodeCapacity` — gpu_memory_total_mb, gpu_memory_free_mb, gpu_memory_limit_mb
- `NodeConfig` — address, grpc_address, coordinator_address, gpu_memory_limit_mb, poll_interval_s

### WorkerNode

**`__init__(config, coordinator_client=None)`** — Initialize with config and optional client override.

**`startup() -> NodeCapacity`** — Initialize Resource Monitor, query GPU, generate node_id.

**`register_with_coordinator() -> RegisterResponse`** — Send Register RPC.

**`accept_layer_assignment(...) -> None`** — Store assignment, load shard, validate.

**`confirm_ready() -> ConfirmReadyResponse`** — Send ConfirmReady RPC.

**`build_engine_and_serve(hidden_dim) -> None`** — Build Layer Engine, start serving + heartbeat.

**`start_serving(on_connection=None) -> ServingLoop`** — Start the serving loop.

**`start_heartbeat() -> HeartbeatSender`** — Start periodic heartbeat sender.

**`run_lifecycle(...) -> None`** — Run full lifecycle end-to-end.

**Properties:** `node_id`, `state`, `config`, `capacity`, `shard_metadata`, `resource_monitor`, `connection_pool`, `layer_registry`, `coordinator_client`, `serving_loop`, `heartbeat_sender`, `layer_engine`.

### HeartbeatSender

**`__init__(coordinator_client, resource_monitor, node_id, interval_s=5.0)`**

**`start() -> None`** / **`stop() -> None`** — Lifecycle management.

**`send_once() -> HeartbeatResponse`** — Send a single heartbeat.

**Properties:** `is_running`, `consecutive_failures`.

---

## meshrun.worker.serving

Request processing pipeline for the serving state.

### Data Classes

- `ServingConfig` — listen_addr, recv_timeout_s
- `ServingStats` — total_success, total_failures. Methods: `record_success()`, `record_failure()`.

### ServingLoop

**`__init__(config, connection_pool, layer_engine, layer_registry, resource_monitor, coordinator_client)`**

**`start() -> None`** — Start accepting connections and processing requests.

**`stop() -> None`** — Stop the serving loop.

**Properties:** `stats -> ServingStats`, `is_running -> bool`.

### Internal Functions

- `_handle_connection(...)` — Process a single TCP connection: read message → forward pass → send downstream
- `_send_downstream(...)` — Send output to downstream node with failure handling and reroute
- `_send_error_upstream(...)` — Send ERROR message back to upstream on unrecoverable failure
- `_build_response_header(...)` — Build FORWARD or RESULT header based on pipeline position
- `_build_error_header(...)` — Build ERROR header for failure responses

---

## meshrun.security.crypto

Standalone AES-256-GCM encryption helpers (separate from protocol.py's integrated encryption).

**`generate_session_key() -> bytes`** — 32 random bytes.

**`derive_key_from_password(password, salt=None) -> tuple[bytes, bytes]`** — PBKDF2-HMAC-SHA256 derivation. Returns `(key, salt)`.

**`encrypt(plaintext, key, aad=None) -> bytes`** — AES-256-GCM encrypt. Returns `[nonce][ciphertext+tag]`.

**`decrypt(encrypted_blob, key, aad=None) -> bytes`** — AES-256-GCM decrypt. Raises `InvalidTag`.

**`pack_for_wire(plaintext, key, aad=None) -> bytes`** — Encrypt + frame: `[4-byte len][encrypted_blob]`.

**`unpack_from_wire(wire_data, key, aad=None) -> bytes`** — Unframe + decrypt.

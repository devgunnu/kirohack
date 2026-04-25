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

- `read_exact(sock, n) -> bytes` — Read exactly `n` bytes, looping on `recv()`. Raises `ConnectionError` on EOF.
- `write_all(sock, data) -> None` — Write all bytes, looping on `send()`. Raises `ConnectionError` if broken.
- `tensor_to_bytes(elements, dtype) -> bytes` — Serialize flat list to raw contiguous bytes.
- `bytes_to_tensor(data, dtype, dims, num_dims) -> list` — Deserialize raw bytes to flat list.
- `read_message(sock) -> tuple[Header, list]` — Read complete message: header + payload.
- `write_message(sock, header, tensor_data) -> None` — Write complete message: header + payload.

### Encryption Functions

- `generate_session_key() -> bytes` — 32-byte AES-256 key via `os.urandom(32)`.
- `encrypt_message(plaintext, key) -> bytes` — AES-256-GCM encrypt. Returns `[nonce][ciphertext+tag]`.
- `decrypt_message(encrypted_blob, key) -> bytes` — AES-256-GCM decrypt. Raises `InvalidTag`.
- `write_message_secure(sock, header, tensor_data, key) -> None` — Encrypted write. Wire: `[4-byte len][nonce][ciphertext][tag]`.
- `read_message_secure(sock, key) -> tuple[Header, list]` — Encrypted read + decrypt.

---

## meshrun.worker.connection_pool

Persistent TCP connection management for the data plane.

### ConnectionPool

- `get_connection(target_addr) -> Optional[socket.socket]` — Get or establish TCP connection (5s timeout, single retry).
- `close_connection(target_addr) -> None` — Close specific connection.
- `close_all() -> None` — Close all connections and stop listener.
- `is_connected(target_addr) -> bool` — Check if live connection exists.
- `accept_incoming(listen_addr, on_connection, backlog) -> None` — Start TCP listener on background thread.
- `get_incoming_connections() -> list` — Snapshot of accepted incoming connections.

---

## meshrun.worker.shard_manager

Selective download and lifecycle management of model shards using safetensors format.

### Key Functions

- `fetch_safetensors_header(url) -> tuple[int, dict[str, TensorInfo]]` — Fetch header via HTTP Range requests.
- `filter_tensors_for_assignment(tensors, layer_range, ...) -> dict` — Filter to assigned layer tensors.
- `download_selected_tensors_cached(url, header_size, tensors, cache_dir, model_id) -> dict` — Cache-first download.
- `load_shard(metadata, ...) -> ShardMetadata` — Full lifecycle: download → deserialize → GPU load.
- `validate_shard(metadata, assigned_tensors) -> ShardMetadata` — Check layer count, dtype, hidden dims.
- `unload_shard(metadata) -> ShardMetadata` — Free GPU memory, reset to UNLOADED.
- `get_shard_info(metadata) -> ShardInfo` — Immutable snapshot.

### Data Classes

- `LayerRange(start, end)` — Immutable, both inclusive.
- `ShardMetadata` — Mutable shard state (model_id, model_url, layer_range, dtype, cache_dir, load_status, loaded_tensors).
- `ShardInfo` — Immutable snapshot for external consumers.
- `TensorInfo` — Tensor metadata from safetensors header (name, dtype, shape, data_offsets).

---

## meshrun.worker.layer_engine

Forward pass execution through hosted transformer layers.

- `build_layer_engine(loaded_tensors, layer_start, layer_end, is_final_node, device) -> LayerEngine` — Construct from loaded shard tensors.
- `forward(engine, hidden_states, step_id) -> Tensor` — Sequential forward pass. Applies LM head if final node.
- `warm_up(engine, hidden_dim, device, dtype) -> None` — Dummy forward pass to pre-allocate GPU resources.

---

## meshrun.worker.resource_monitor

GPU memory and utilization tracking with heartbeat snapshots.

### ResourceMonitor

- `__init__(gpu_memory_limit_mb, poll_interval_s, device_index)` — Initialize with user-configured limit.
- `start() / stop()` — Background polling thread lifecycle.
- `poll_once() -> GpuMetrics` — Synchronous GPU poll.
- `get_latest_metrics() -> Optional[GpuMetrics]` — Most recent metrics.
- `get_heartbeat_snapshot() -> HeartbeatSnapshot` — Metrics for heartbeat messages.
- `increment_active_requests() / decrement_active_requests()` — Thread-safe request counting.
- `is_over_limit() -> bool` — Check if GPU usage exceeds configured limit.

---

## meshrun.worker.layer_registry

Layer assignment storage and pipeline topology queries.

### LayerAssignmentRegistry

Thread-safe registry storing exactly one assignment at a time.

- `accept_layer_assignment(assignment) -> None` — Store new assignment.
- `clear() -> None` — Remove current assignment.
- `get_downstream_address()`, `get_upstream_addresses()`, `get_layer_range()`, `get_dtype()`, `is_final_node()`, `get_model_id()`, `get_model_url()`, `get_node_id()` — Query methods.

---

## meshrun.worker.coordinator_client

gRPC client abstraction for Coordinator communication.

### CoordinatorClient (Abstract Base)

- `register(request) -> RegisterResponse`
- `confirm_ready(request) -> ConfirmReadyResponse`
- `heartbeat(request) -> HeartbeatResponse`
- `report_failure(request) -> ReportFailureResponse`
- `close() -> None`

### Implementations

- `GrpcCoordinatorClient(coordinator_address)` — Real gRPC client using generated proto stubs.
- `StubCoordinatorClient(...)` — In-memory stub for testing.

---

## meshrun.worker.node

Worker node lifecycle orchestration.

### WorkerNode

- `__init__(config, coordinator_client=None)` — Initialize with config and optional client override.
- `startup() -> NodeCapacity` — Initialize Resource Monitor, query GPU, generate node_id.
- `register_with_coordinator() -> RegisterResponse` — Send Register RPC.
- `accept_layer_assignment(...) -> None` — Store assignment, load shard, validate. Accepts `session_key`.
- `confirm_ready() -> ConfirmReadyResponse` — Send ConfirmReady RPC.
- `build_engine_and_serve(hidden_dim) -> None` — Build Layer Engine, start serving + heartbeat.
- `start_serving(on_connection) -> ServingLoop` — Start the serving loop.
- `start_heartbeat() -> HeartbeatSender` — Start periodic heartbeat sender.
- `run_lifecycle(...) -> None` — Run full lifecycle end-to-end.

### HeartbeatSender

- `start() / stop()` — Lifecycle management.
- `send_once() -> HeartbeatResponse` — Send a single heartbeat.

---

## meshrun.worker.serving

Encrypted request processing pipeline for the serving state.

### ServingLoop

- `__init__(config, connection_pool, layer_engine, layer_registry, resource_monitor, coordinator_client, session_key)` — Initialize with session key for encrypted I/O.
- `start() -> None` — Start accepting connections and processing requests.
- `stop() -> None` — Stop the serving loop.
- `stats -> ServingStats` — Success/failure counts.
- `is_running -> bool` — Whether the loop is active.

All data plane reads/writes use `read_message_secure` / `write_message_secure` with the session key.

---

## meshrun.coordinator.server

gRPC Coordinator server.

### CoordinatorServer

- `__init__(host, port, heartbeat_interval_s, missed_threshold, dead_threshold)` — Initialize all components.
- `start() -> None` — Start gRPC server and health tracker.
- `stop(grace) -> None` — Graceful shutdown.
- `registry -> NodeRegistry` — Access node registry.
- `key_manager -> KeyManager` — Access key manager.
- `layer_map -> LayerMap` — Access layer map.
- `priority_queue -> PriorityQueue` — Access priority queue.

### CoordinatorServicer

Implements all gRPC RPCs: Register, Heartbeat, ConfirmReady, Deregister, RequestRoute, ReportFailure, TriggerAssignment, AcceptLayerAssignment.

---

## meshrun.coordinator.registry

Node registry and health tracking.

### NodeRegistry

Thread-safe registry. All mutations protected by `threading.Lock`.

- `register_node(registration) -> RegistrationResult` — Add node, reject duplicates.
- `deregister_node(node_id) -> bool` — Remove node.
- `update_heartbeat(node_id, metrics) -> HeartbeatResult` — Update last_seen and metrics.
- `mark_node_healthy(node_id) -> None` — REGISTERED → HEALTHY transition.
- `update_node_assignment(node_id, layer_start, layer_end) -> None` — Store assignment.
- `get_node(node_id) -> NodeEntry | None` — Lookup single node.
- `get_all_healthy_nodes() -> list[NodeEntry]` — Only HEALTHY nodes.
- `get_all_nodes() -> list[NodeEntry]` — All nodes.

### HealthTracker

- `__init__(registry, heartbeat_interval_s, missed_threshold, dead_threshold)` — Configure thresholds.
- `start() / stop()` — Background thread lifecycle.

---

## meshrun.coordinator.scheduler

Layer assignment, route building, and priority queue.

### Functions

- `compute_assignments(model_id, total_layers, dtype, nodes, key_manager) -> AssignmentPlan` — Greedy contiguous assignment.
- `build_route(model_id, layer_map, registry, key_manager) -> ExecutionPath` — Ordered node list + session key.
- `handle_failure(request_id, failed_node_id, layer_map, registry) -> RerouteInfo` — Backup lookup.

### LayerMap

- `get_primary_node_for_layer(layer_index) -> LayerMapEntry | None`
- `get_backup_for_range(layer_start, layer_end) -> LayerMapEntry | None`
- `get_all_entries() -> list[LayerMapEntry]`
- `set_entries(entries) -> None`

### PriorityQueue

- `__init__(max_depth, alpha, beta)` — Configure capacity and scoring weights.
- `enqueue(request_id, client_id, model_id, compute_contributed) -> QueueEntry`
- `dequeue() -> QueueEntry | None` — Re-score and return highest priority.
- `is_full -> bool`

---

## meshrun.coordinator.key_manager

AES-256 session key lifecycle.

### KeyManager

Thread-safe `model_id → session_key` store.

- `generate_pipeline_key(model_id) -> bytes` — Create 32-byte key via `os.urandom(32)`.
- `get_pipeline_key(model_id) -> bytes | None` — Retrieve stored key.
- `rotate_key(model_id) -> bytes` — Replace with fresh key.
- `delete_key(model_id) -> bool` — Remove key.

---

## meshrun.client.client

End-to-end inference orchestration.

### InferenceClient

- `__init__(coordinator_address, model_name, model_url, cache_dir, device)` — Initialize client.
- `initialize() -> None` — Load tokenizer and embedding weights.
- `request_route(model_id) -> ExecutionPath` — gRPC call to Coordinator.
- `submit_inference(prompt_text) -> str` — Full flow: tokenize → embed → route → encrypt → send → decode.
- `close() -> None` — Release gRPC channel and transport sockets.

### Data Classes

- `RouteNode(node_id, address, layer_start, layer_end)` — Single node in execution path.
- `ExecutionPath(request_id, session_key, nodes, backup_map)` — Complete execution path.

---

## meshrun.client.tokenizer

Tokenizer and embedding engine.

### ModelTokenizer

- `load_tokenizer(model_name_or_path) -> None` — Load HuggingFace AutoTokenizer.
- `tokenize(text) -> list[int]` — Text to token IDs.
- `detokenize(token_ids) -> str` — Token IDs to text.
- `load_embedding(model_url, cache_dir, device) -> None` — Selective download of embed_tokens weights.
- `embed(token_ids) -> torch.Tensor` — Embedding lookup → `[1, seq_len, hidden_dim]` fp16.
- `decode_logits(logits_tensor) -> list[int]` — Greedy argmax on last position.

---

## meshrun.client.transport

Encrypted TCP transport.

### SecureTransport

- `__init__(connect_timeout)` — Initialize with connection timeout.
- `connect(node_addr) -> socket.socket` — Establish TCP connection to `host:port`.
- `send_forward(sock, hidden_states, session_key, request_id, step_id) -> None` — Encrypt and send FORWARD.
- `receive_result(sock, session_key) -> tuple[Header, list]` — Receive and decrypt RESULT.
- `close() -> None` — Close all open sockets.

---

## meshrun.security.crypto

Standalone AES-256-GCM encryption helpers.

- `generate_session_key() -> bytes` — 32 random bytes.
- `derive_key_from_password(password, salt) -> tuple[bytes, bytes]` — PBKDF2-HMAC-SHA256 derivation.
- `encrypt(plaintext, key, aad) -> bytes` — AES-256-GCM encrypt.
- `decrypt(encrypted_blob, key, aad) -> bytes` — AES-256-GCM decrypt.
- `pack_for_wire(plaintext, key, aad) -> bytes` — Encrypt + frame: `[4-byte len][encrypted_blob]`.
- `unpack_from_wire(wire_data, key, aad) -> bytes` — Unframe + decrypt.

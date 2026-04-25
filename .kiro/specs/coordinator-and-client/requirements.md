# Requirements: Coordinator Server & Inference Client

## Requirement 1: gRPC Service Definition & Server

### 1.1 Proto Service Definition
The Coordinator MUST define a `.proto` file specifying all gRPC service methods and message types for the MeshRun control plane.

**Acceptance Criteria:**
- The proto file defines a `CoordinatorService` with RPCs: Register, Heartbeat, ConfirmReady, Deregister, RequestRoute, ReportFailure
- All request/response message types are defined with correct field types matching the existing Python dataclasses in `coordinator_client.py`
- The proto file compiles successfully with `grpcio-tools` to produce `coordinator_pb2.py` and `coordinator_pb2_grpc.py`
- The `RequestRouteResponse` includes a `session_key` field (bytes) for AES-256 key distribution
- The `AcceptLayerAssignmentRequest` includes a `session_key` field (bytes) for key distribution to workers

### 1.2 gRPC Server Lifecycle
The Coordinator MUST run a gRPC server that accepts concurrent RPCs from worker nodes and clients.

**Acceptance Criteria:**
- The server starts on a configurable host and port (default `0.0.0.0:50051`)
- The server handles concurrent RPCs from multiple worker nodes and clients
- The server can be started and stopped gracefully (with a configurable grace period for in-flight RPCs)
- The servicer class delegates all business logic to the registry, scheduler, and key manager components

### 1.3 Proto Stub Integration with Worker Node
The existing `GrpcCoordinatorClient` in `coordinator_client.py` MUST be updated to use the real generated proto stubs instead of returning synthetic responses.

**Acceptance Criteria:**
- `GrpcCoordinatorClient.register()` translates `RegisterRequest` dataclass to the protobuf `RegisterRequest` message and calls the real stub
- `GrpcCoordinatorClient.confirm_ready()` translates `ConfirmReadyRequest` to protobuf and calls the real stub
- `GrpcCoordinatorClient.heartbeat()` translates `HeartbeatRequest` to protobuf and calls the real stub
- `GrpcCoordinatorClient.report_failure()` translates `ReportFailureRequest` to protobuf and calls the real stub
- All synthetic/placeholder responses are removed
- The client correctly handles gRPC errors (unavailable, deadline exceeded, etc.)

## Requirement 2: Node Registry & Health Tracking

### 2.1 Node Registration
The Coordinator MUST maintain a live registry of all worker nodes with their identity, network addresses, GPU capacity, and current status.

**Acceptance Criteria:**
- `register_node` adds a node to the registry with status REGISTERED, storing node_id, address, grpc_address, capacity (gpu_memory_total_mb, gpu_memory_free_mb, memory_limit_mb, gpu_utilization), and registration timestamp
- Duplicate node_id registrations are rejected with an appropriate error
- `deregister_node` removes the node from the registry and returns success
- `get_node` returns the full node entry or None if not found
- `get_all_healthy_nodes` returns only nodes with status HEALTHY
- All registry operations are thread-safe (protected by a lock)

### 2.2 Health Monitoring via Heartbeats
The Coordinator MUST track node health by processing periodic heartbeat RPCs and detecting missed heartbeats.

**Acceptance Criteria:**
- `update_heartbeat` updates the node's `last_seen` timestamp, `gpu_utilization`, `memory_used_mb`, and `active_requests`
- A background health check thread runs every `heartbeat_interval_s` (default 5 seconds)
- Nodes that miss heartbeats beyond `missed_threshold` consecutive intervals (default 3, i.e., 15 seconds) are marked UNHEALTHY
- Nodes that miss heartbeats beyond `dead_threshold` consecutive intervals (default 5, i.e., 25 seconds) are marked DEAD
- UNHEALTHY nodes are excluded from new route computations but remain in the registry
- Nodes that resume heartbeats transition from UNHEALTHY back to HEALTHY
- `mark_node_healthy` transitions a REGISTERED node to HEALTHY (called after ConfirmReady)

### 2.3 Node Status Lifecycle
The Node Registry MUST enforce valid status transitions.

**Acceptance Criteria:**
- Valid transitions: REGISTERED → HEALTHY, HEALTHY → UNHEALTHY, UNHEALTHY → HEALTHY, UNHEALTHY → DEAD, HEALTHY → DEREGISTERED
- Invalid transitions are rejected (e.g., REGISTERED → DEAD directly)
- Status changes are logged for observability

## Requirement 3: Layer Assignment

### 3.1 Static Layer Assignment Algorithm
The Coordinator MUST compute contiguous layer assignments for registered nodes based on their GPU capacity.

**Acceptance Criteria:**
- `compute_assignments` accepts `model_id`, `total_layers`, and `dtype` (fp16 or int8)
- Per-layer memory estimation: ~200MB for fp16, ~100MB for int8 (for a ~3B parameter model)
- Framework overhead of 800MB is subtracted from each node's `memory_limit_mb` to compute usable memory
- Nodes are sorted by usable memory descending (largest capacity first)
- Layers are assigned greedily in contiguous blocks: each node gets `min(max_layers_it_can_hold, remaining_unassigned_layers)`
- Every layer index from 0 to `total_layers - 1` is assigned to exactly one primary node
- No layer range overlaps between primary nodes
- If total node capacity is insufficient to cover all layers, the assignment fails with a clear error

### 3.2 Backup Node Assignment
The Coordinator MUST assign backup nodes for each primary layer range when spare capacity exists.

**Acceptance Criteria:**
- After primary assignment, for each primary node's layer range, the algorithm searches for another node with enough spare capacity to host the same range
- Backup assignments are stored in the Layer Map alongside primary assignments
- If no node has spare capacity for a given range, that range has no backup (acceptable for POC)
- Backup nodes receive the same session key as primary nodes during assignment distribution

### 3.3 Session Key Generation During Assignment
The Coordinator MUST generate an AES-256 session key during layer assignment and distribute it to all nodes in the pipeline.

**Acceptance Criteria:**
- A single 32-byte AES-256 key is generated per model pipeline using `os.urandom(32)`
- The session key is included in every `AcceptLayerAssignment` message sent to worker nodes
- The session key is stored by the Key Manager and retrievable by model_id
- The same session key is included in `RequestRoute` responses so clients can encrypt data plane traffic

## Requirement 4: Route Building & Priority Queue

### 4.1 Execution Path Construction
The Coordinator MUST build ordered execution paths mapping model layers to healthy nodes.

**Acceptance Criteria:**
- `build_route` accepts `model_id` and returns an `ExecutionPath` with an ordered list of nodes, a unique `request_id`, and the pipeline `session_key`
- For each layer range, the primary node is selected if HEALTHY; otherwise the backup node is used
- If any layer range has no healthy primary or backup node, the route request fails with a descriptive error
- The node list is ordered by layer range (ascending layer indices)
- Each node entry includes `node_id`, `address` (TCP host:port), `layer_start`, and `layer_end`

### 4.2 Priority Queue Scheduling
The Coordinator MUST schedule inference requests using a priority queue with a scoring function.

**Acceptance Criteria:**
- Scoring function: `priority = α * compute_contributed + β * wait_time` with defaults α=0.7, β=0.3
- `enqueue_request` adds a request with its `client_id`, `compute_contributed`, and enqueue timestamp
- `dequeue_request` returns the highest-priority entry (re-scored at dequeue time to account for updated wait_time)
- Maximum queue depth is configurable (default 100); requests beyond this are rejected with QUEUE_FULL
- Wait time is computed as `current_time - enqueued_at` in seconds

### 4.3 Failure Handling & Reroute
The Coordinator MUST handle failure reports from worker nodes and provide backup routing information.

**Acceptance Criteria:**
- `handle_failure` accepts `request_id` and `failed_node_id`
- The Coordinator looks up the failed node's layer range in the Layer Map
- If a backup node exists for that range and is HEALTHY, returns `RerouteInfo` with the backup node's TCP address
- If no backup is available, returns `RerouteInfo` with `backup_addr=None`
- The failed node's status is updated (optionally marked UNHEALTHY if not already)

## Requirement 5: Key Manager

### 5.1 Session Key Lifecycle
The Key Manager MUST generate, store, retrieve, and delete AES-256 session keys for model pipelines.

**Acceptance Criteria:**
- `generate_pipeline_key(model_id)` creates a 32-byte key using `os.urandom(32)` and stores it keyed by `model_id`
- `get_pipeline_key(model_id)` returns the stored key or `None` if no key exists
- `rotate_key(model_id)` generates a new key, replaces the old one, and returns the new key
- `delete_key(model_id)` removes the key and returns `True` if it existed
- All operations are thread-safe
- Keys are stored in-memory only (no persistence for POC)

## Requirement 6: Inference Client — Tokenization & Embedding

### 6.1 Model-Specific Tokenizer Loading
The Client MUST load a HuggingFace AutoTokenizer matching the served model.

**Acceptance Criteria:**
- `load_tokenizer(model_name_or_path)` loads the tokenizer using `transformers.AutoTokenizer.from_pretrained()`
- The tokenizer is loaded once and reused across inference requests
- `tokenize(text)` converts raw text to a list of token IDs
- `detokenize(token_ids)` converts token IDs back to text
- The tokenizer matches the model being served (e.g., if serving LLaMA-3B, the LLaMA tokenizer is used)

### 6.2 Embedding Weight Loading via Selective Download
The Client MUST load the embedding layer weights by selectively downloading only the `embed_tokens` tensors from the safetensors model file.

**Acceptance Criteria:**
- Reuses the existing `shard_manager.py` infrastructure (`fetch_safetensors_header`, `filter_tensors_for_assignment`, `download_selected_tensors_cached`)
- Filters the safetensors header for tensors containing `embed_tokens` in their name
- Downloads only the embedding tensor byte ranges via HTTP Range requests
- Caches downloaded embedding weights locally to avoid re-downloading
- Reconstructs the embedding weight as a `torch.Tensor` of shape `[vocab_size, hidden_dim]`
- The embedding weight is moved to the target device (CPU for POC, optionally GPU)

### 6.3 Local Embedding Execution
The Client MUST run the embedding layer locally to convert token IDs to hidden states.

**Acceptance Criteria:**
- `embed(token_ids)` performs embedding lookup: `hidden_states = embedding_weight[token_ids]`
- Output shape is `[1, seq_len, hidden_dim]` (batch=1)
- The embedding is computed using PyTorch (`torch.nn.functional.embedding` or direct indexing)
- The output tensor dtype matches the model's expected dtype (fp16)

### 6.4 Logits Decoding
The Client MUST decode logits tensors into output token IDs.

**Acceptance Criteria:**
- `decode_logits(logits_tensor)` applies greedy argmax: `argmax(logits[:, -1, :])` for the last token position
- Returns a list of output token IDs
- For POC: single-pass inference (one forward pass, decode the last position)

## Requirement 7: Inference Client — Route Acquisition & Transport

### 7.1 Route Acquisition via gRPC
The Client MUST connect to the Coordinator via gRPC to obtain execution paths and session keys.

**Acceptance Criteria:**
- The client creates a gRPC channel to the Coordinator address
- `request_route(model_id)` calls the `RequestRoute` RPC and returns an `ExecutionPath` with nodes list and session_key
- The client handles gRPC errors gracefully (unavailable, deadline exceeded)
- The gRPC channel is reusable across multiple inference requests

### 7.2 Encrypted Tensor Transmission
The Client MUST encrypt hidden states using AES-256-GCM before sending to the pipeline.

**Acceptance Criteria:**
- Uses the `session_key` received from the Coordinator's `RequestRoute` response
- Builds a `Header` with `message_type=FORWARD`, correct tensor shape, dtype, and payload_size
- Calls `write_message_secure(sock, header, tensor_data, session_key)` from `protocol.py`
- The wire format matches the existing protocol: `[4-byte len][12-byte nonce][ciphertext][16-byte GCM tag]`

### 7.3 Encrypted Result Reception
The Client MUST receive and decrypt the final logits from the pipeline.

**Acceptance Criteria:**
- Calls `read_message_secure(sock, session_key)` from `protocol.py` to receive the encrypted RESULT
- Validates that `header.message_type == RESULT`
- Reconstructs the logits tensor from the decrypted flat data using header dims and dtype
- Handles `InvalidTag` exceptions (decryption failure) gracefully with a clear error message

### 7.4 End-to-End Inference Flow
The Client MUST orchestrate the complete inference pipeline from text input to text output.

**Acceptance Criteria:**
- `submit_inference(prompt_text)` executes the full flow: tokenize → embed → request route → encrypt & send → receive & decrypt → decode → detokenize
- Returns the generated text as a string
- Handles errors at each stage with appropriate error messages
- Cleans up TCP connections after each inference request

## Requirement 8: Worker Node Updates — Secure Serving

### 8.1 Encrypted Serving Loop
The Worker Node's serving loop MUST use encrypted message read/write for all data plane traffic.

**Acceptance Criteria:**
- `_handle_connection` uses `read_message_secure(sock, session_key)` instead of `read_message(sock)`
- Downstream forwarding uses `write_message_secure(sock, header, data, session_key)` instead of `write_message`
- RESULT messages sent back to the client use `write_message_secure`
- ERROR messages sent upstream use `write_message_secure`
- The session key is passed through from the `ServingLoop` to each connection handler

### 8.2 Session Key Storage in Worker Node
The Worker Node MUST accept and store the session key received during layer assignment.

**Acceptance Criteria:**
- `accept_layer_assignment` accepts a `session_key: bytes` parameter
- The session key is stored on the `WorkerNode` instance
- The session key is passed to the `ServingLoop` when `start_serving` is called
- The `ServingLoop` makes the session key available to all connection handler threads

## Requirement 9: Coordinator Server Lifecycle

### 9.1 Server Startup
The Coordinator Server MUST initialize all internal components and start the gRPC server.

**Acceptance Criteria:**
- On startup: initialize Node Registry, Scheduler (with empty Layer Map and Priority Queue), Key Manager, and Health Tracker background thread
- Start the gRPC server on the configured address
- Log the server address and readiness status

### 9.2 Server Shutdown
The Coordinator Server MUST shut down gracefully.

**Acceptance Criteria:**
- Stop the health check background thread
- Stop the gRPC server with a grace period for in-flight RPCs
- Clear the Node Registry and Key Manager state
- Log shutdown completion

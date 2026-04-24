# Tasks: Distributed AI Inference

## Task 1: TCP Binary Protocol — Header Serialization & Validation
> Requirements: 1.1, 1.2

- [ ] 1.1 Define the header data structure with all fields (message_type, request_id, step_id, payload_size, dtype, num_dims, dims[4], reserved) and their byte sizes/offsets
- [ ] 1.2 Implement header serialization (pack) that produces exactly 32 bytes in the specified byte layout
- [ ] 1.3 Implement header deserialization (unpack) that reads 32 bytes and extracts all fields
- [ ] 1.4 Implement header validation: check message_type ∈ {1,2,3,4}, dtype ∈ {1,2}, num_dims ∈ {1,2,3,4}, dims consistency, and payload_size = product(dims) * dtype_size
- [ ] 1.5 Write unit tests for header round-trip (serialize → deserialize produces identical fields), 32-byte size invariant, and validation rejection of invalid headers

## Task 2: TCP Binary Protocol — Reliable Framing & Tensor Serialization
> Requirements: 1.3, 1.4, 1.5

- [ ] 2.1 Implement `read_exact(n)` function that loops on recv() until exactly n bytes are accumulated, with EOF and timeout error handling
- [ ] 2.2 Implement `write_all(data)` function that writes all bytes to a TCP socket, handling partial writes
- [ ] 2.3 Implement tensor-to-bytes serialization (raw contiguous, row-major, matching dtype: fp16 little-endian IEEE 754, int8 signed)
- [ ] 2.4 Implement bytes-to-tensor deserialization using dtype and dims from the header
- [ ] 2.5 Implement full message read: read_exact(32) for header → validate → read_exact(payload_size) for payload → reconstruct tensor
- [ ] 2.6 Implement full message write: serialize tensor → compute header fields → pack header → write_all(header + payload)
- [ ] 2.7 Write unit tests for read_exact with simulated partial reads, tensor round-trip (serialize → deserialize bitwise identical), and full message read/write cycle

## Task 3: Connection Pool
> Requirements: 2.1, 2.2, 2.3

- [ ] 3.1 Implement Connection Pool data structure with a map of target_addr → TCP connection
- [ ] 3.2 Implement GetConnection(target_addr): return existing connection or establish new TCP connection with 5-second timeout
- [ ] 3.3 Implement connection reuse: subsequent GetConnection calls for the same address return the same connection
- [ ] 3.4 Implement CloseConnection(target_addr) and CloseAll for graceful cleanup
- [ ] 3.5 Implement IsConnected(target_addr) to check if a live connection exists
- [ ] 3.6 Implement TCP listener (AcceptIncoming) that accepts inbound connections on the configured data plane port
- [ ] 3.7 Implement failure detection: detect broken connections on send/receive and trigger single retry before reporting failure
- [ ] 3.8 Write unit tests for connection reuse, timeout behavior, and close operations

## Task 4: Shard Manager
> Requirements: 3.1, 3.2, 3.3, 3.4

- [ ] 4.1 Define shard metadata structure: model_id, layer_range (start, end), dtype, memory_footprint_mb, load_status (UNLOADED, LOADING, READY, ERROR)
- [ ] 4.2 Implement LoadShard(model_id, layer_start, layer_end, dtype): load quantized weights from disk, move to GPU, update status UNLOADED → LOADING → READY
- [ ] 4.3 Implement ValidateShard: verify layer count matches (layer_end - layer_start + 1), dtype matches, hidden dimensions are consistent
- [ ] 4.4 Implement UnloadShard: free GPU memory, update status to UNLOADED, report freed memory to Resource Monitor
- [ ] 4.5 Implement GetShardInfo: return current shard metadata
- [ ] 4.6 Write unit tests for load/validate/unload lifecycle and status transitions

## Task 5: Layer Engine
> Requirements: 4.1, 4.2

- [ ] 5.1 Implement Forward(hidden_states, step_id): run sequential forward pass through all hosted layers, return output hidden states
- [ ] 5.2 Implement final-node logic: if is_final_node, apply LM head to produce logits instead of hidden states
- [ ] 5.3 Implement WarmUp: run dummy forward pass to compile GPU kernels and pre-allocate activation memory
- [ ] 5.4 Add output validation: check output tensor has correct shape and contains no NaN/Inf values
- [ ] 5.5 Write unit tests for forward pass output shape, final-node logits output, and NaN/Inf detection

## Task 6: Resource Monitor
> Requirements: 5.1, 5.2, 5.3

- [ ] 6.1 Implement GPU metrics polling: query gpu_memory_total, gpu_memory_used, gpu_memory_free, gpu_utilization at configurable interval
- [ ] 6.2 Implement tracking of active_requests, shard_memory_mb, and activation_memory_mb
- [ ] 6.3 Store user-configured gpu_memory_limit_mb and implement limit comparison on each poll
- [ ] 6.4 Implement heartbeat snapshot: return current gpu_utilization, memory_used_mb, active_requests
- [ ] 6.5 Implement memory limit alert: generate warning when gpu_memory_used_mb exceeds gpu_memory_limit_mb
- [ ] 6.6 Write unit tests for metric tracking, snapshot accuracy, and limit alerting

## Task 7: Layer Assignment Registry
> Requirements: 6.1, 6.2

- [ ] 7.1 Define registry data structure: node_id, model_id, layer_start, layer_end, dtype, is_final_node, downstream_node, upstream_nodes
- [ ] 7.2 Implement assignment storage: set assignment on AcceptLayerAssignment from Coordinator
- [ ] 7.3 Implement query methods for other sub-components to read assignment details (downstream address, layer range, dtype, is_final)
- [ ] 7.4 Write unit tests for assignment storage and query

## Task 8: Worker Node Lifecycle — Startup & Registration
> Requirements: 7.1, 7.2

- [ ] 8.1 Implement node startup: initialize Resource Monitor, query local GPU resources, generate node_id
- [ ] 8.2 Implement Coordinator registration: send Register RPC with node_id, address, grpc_address, capacity (including user memory_limit)
- [ ] 8.3 Implement AcceptLayerAssignment handler: receive assignment, store in registry, trigger Shard Manager LoadShard
- [ ] 8.4 Implement ConfirmReady: after shard load + validation, send ConfirmReady RPC to Coordinator
- [ ] 8.5 Implement state transitions: Initializing → Registering → WaitingAssignment → LoadingShard → Validating → Ready

## Task 9: Worker Node Lifecycle — Serving & Request Processing
> Requirements: 7.3, 7.5

- [ ] 9.1 Implement the main serving loop: accept TCP connections, read Forward messages, process, and forward results
- [ ] 9.2 Wire together Message Handler → Layer Engine → Connection Pool for the forward pipeline
- [ ] 9.3 Implement periodic heartbeat sending to Coordinator with Resource Monitor snapshots
- [ ] 9.4 Implement downstream failure handling: detect TCP send failure, call ReportFailure on Coordinator, receive RerouteInfo, retry to backup node
- [ ] 9.5 Implement error response: if backup also fails, send ERROR message back upstream
- [ ] 9.6 Write integration test: two-node pipeline with Forward message flowing through

## Task 10: Worker Node Lifecycle — Graceful Shutdown
> Requirements: 7.4

- [ ] 10.1 Implement shutdown signal handler: stop accepting new Forward requests
- [ ] 10.2 Implement drain: wait for all in-flight forward passes to complete
- [ ] 10.3 Trigger Shard Manager UnloadShard and Connection Pool CloseAll on shutdown
- [ ] 10.4 Write test for graceful shutdown with in-flight request completing before termination

# Requirements: Distributed AI Inference

## Requirement 1: TCP Binary Protocol (Message Handler)

### 1.1 Fixed-Size Header Serialization
The Message Handler MUST serialize and deserialize a fixed 32-byte header containing: message_type (uint8), request_id (uint32), step_id (uint32), payload_size (uint32), dtype (uint8), num_dims (uint8), dims (uint32[4]), and reserved (uint8).

**Acceptance Criteria:**
- Every serialized header is exactly 32 bytes regardless of field values
- Header fields are packed at the correct byte offsets as specified in the protocol (offset 0: message_type, offset 1: request_id, offset 5: step_id, etc.)
- Round-trip serialization/deserialization produces identical field values

### 1.2 Header Validation
The Message Handler MUST validate all header fields before processing the payload.

**Acceptance Criteria:**
- message_type must be in {1, 2, 3, 4} (FORWARD, RESULT, ERROR, HEARTBEAT_DATA)
- dtype must be in {1, 2} (fp16, int8)
- num_dims must be in {1, 2, 3, 4}
- dims[i] > 0 for i < num_dims, dims[i] = 0 for i >= num_dims
- payload_size must equal product(dims[0:num_dims]) * dtype_size(dtype) where dtype_size(fp16)=2, dtype_size(int8)=1
- Invalid headers are rejected with an appropriate error before any payload read

### 1.3 Reliable TCP Framing (read_exact)
The Message Handler MUST implement a `read_exact(n)` function that reads exactly n bytes from a TCP stream, handling partial reads.

**Acceptance Criteria:**
- The function loops until exactly n bytes have been accumulated, even if individual recv() calls return fewer bytes
- On EOF before n bytes are read, the function returns an error (not partial data)
- On timeout, the function returns a timeout error
- The header is read using read_exact(32), then the payload is read using read_exact(payload_size)

### 1.4 Tensor Serialization
The Message Handler MUST serialize tensors to raw contiguous bytes (row-major / C-contiguous order) and deserialize them back.

**Acceptance Criteria:**
- fp16 tensors are serialized as little-endian IEEE 754 half-precision bytes
- int8 tensors are serialized as signed 8-bit integer bytes
- Round-trip serialize/deserialize produces a bitwise-identical tensor for any valid dtype and dimensions
- No serialization libraries (JSON, Protobuf, MessagePack) are used in the data path

### 1.5 Message Write
The Message Handler MUST write complete messages (header + payload) to TCP connections.

**Acceptance Criteria:**
- Header and payload are written as a single contiguous write operation (write_all)
- The payload_size field in the header matches the actual payload byte count
- The dims fields in the header match the tensor's actual shape

## Requirement 2: Connection Pool

### 2.1 Persistent TCP Connections
The Connection Pool MUST maintain persistent TCP connections to downstream nodes, reusing them across requests.

**Acceptance Criteria:**
- Calling GetConnection(addr) for the same address returns the same connection object (no new connection created)
- Connections are established once and reused for all subsequent Forward operations to that target
- One TCP connection per node pair (not per request)

### 2.2 Connection Lifecycle Management
The Connection Pool MUST handle connection establishment, failure detection, and cleanup.

**Acceptance Criteria:**
- Connection establishment times out after 5 seconds
- Idle connections time out after 30 seconds
- When a connection is lost, the pool detects the failure on the next send/receive attempt
- Failed connections trigger a single retry attempt before reporting failure to the Coordinator
- CloseAll gracefully closes all connections during shutdown
- CloseConnection removes a specific connection from the pool

### 2.3 Incoming Connection Acceptance
The Connection Pool MUST accept inbound TCP connections from upstream nodes and clients.

**Acceptance Criteria:**
- A TCP listener accepts incoming connections on the node's configured data plane port
- Multiple upstream nodes can connect simultaneously
- Accepted connections are handed to the Message Handler for processing

## Requirement 3: Shard Manager

### 3.1 Shard Loading
The Shard Manager MUST load quantized model shards (fp16 or int8) for the assigned layer range.

**Acceptance Criteria:**
- LoadShard accepts model_id, layer_start, layer_end, and dtype parameters
- Weights are loaded from disk or network and moved to GPU memory
- After loading, the shard's memory footprint is reported to the Resource Monitor
- Load status transitions: UNLOADED → LOADING → READY (or ERROR on failure)

### 3.2 Shard Validation
The Shard Manager MUST validate that a loaded shard matches the expected configuration.

**Acceptance Criteria:**
- ValidateShard checks that the number of loaded layers matches (layer_end - layer_start + 1)
- ValidateShard checks that the dtype of loaded weights matches the assigned dtype
- ValidateShard checks that hidden dimensions are consistent across all loaded layers
- Validation failure transitions the load status to ERROR

### 3.3 Shard Unloading
The Shard Manager MUST free GPU memory when unloading a shard.

**Acceptance Criteria:**
- UnloadShard releases all GPU memory held by the model shard
- After unloading, the Resource Monitor reflects the freed memory
- Load status transitions to UNLOADED after successful unload

### 3.4 Shard Info Reporting
The Shard Manager MUST report metadata about the currently loaded shard.

**Acceptance Criteria:**
- GetShardInfo returns: model_id, layer_range (start, end), dtype, memory_footprint_mb, and load status
- Returns accurate information reflecting the current state of the shard

## Requirement 4: Layer Engine

### 4.1 Forward Pass Execution
The Layer Engine MUST execute a sequential forward pass through all hosted transformer layers.

**Acceptance Criteria:**
- Forward accepts hidden_states tensor and step_id as input
- Layers are executed sequentially in order (layer_start through layer_end)
- Output tensor has the correct shape for the next node's input (or logits shape if final node)
- Output tensor contains no NaN or Inf values for valid input
- If is_final_node is true, the LM head is applied to produce logits

### 4.2 GPU Warm-Up
The Layer Engine MUST support a warm-up operation to pre-allocate GPU resources.

**Acceptance Criteria:**
- WarmUp runs a dummy forward pass through all hosted layers
- After warm-up, GPU kernels are compiled/cached and activation memory is allocated
- Warm-up completes without errors for valid shard configurations

## Requirement 5: Resource Monitor

### 5.1 GPU Metrics Tracking
The Resource Monitor MUST track GPU memory and compute utilization metrics.

**Acceptance Criteria:**
- Tracks: gpu_memory_total_mb, gpu_memory_used_mb, gpu_memory_free_mb, gpu_utilization, active_requests, shard_memory_mb, activation_memory_mb
- Metrics are polled at a configurable interval (default: 1 second)
- gpu_memory_limit_mb reflects the user-configured memory limit

### 5.2 Heartbeat Resource Snapshot
The Resource Monitor MUST provide a resource snapshot for inclusion in heartbeat messages.

**Acceptance Criteria:**
- Snapshot includes gpu_utilization, memory_used_mb, and active_requests
- Snapshot reflects current state at time of request (not stale data beyond one polling interval)

### 5.3 Memory Limit Enforcement
The Resource Monitor MUST alert when actual GPU usage exceeds the user-configured memory limit.

**Acceptance Criteria:**
- Compares gpu_memory_used_mb against gpu_memory_limit_mb on each poll
- Generates an alert/warning when usage exceeds the configured limit
- Does NOT autonomously adjust allocations (monitoring only — allocation is user-controlled)

## Requirement 6: Layer Assignment Registry

### 6.1 Assignment Storage
The Layer Assignment Registry MUST store the current layer assignment for the node.

**Acceptance Criteria:**
- Stores: node_id, model_id, layer_start, layer_end, dtype, is_final_node, downstream_node address, upstream_nodes addresses
- Assignment is set when AcceptLayerAssignment is received from the Coordinator
- Assignment can be queried by other sub-components (Message Handler, Layer Engine)

### 6.2 Pipeline Topology Awareness
The Layer Assignment Registry MUST provide downstream and upstream node information for routing.

**Acceptance Criteria:**
- downstream_node is null if is_final_node is true
- downstream_node contains the TCP host:port of the next node in the pipeline
- upstream_nodes contains the list of TCP addresses that may send data to this node

## Requirement 7: Worker Node Lifecycle

### 7.1 Node Startup and Registration
The Worker Node MUST register with the Coordinator on startup, reporting its capacity and user-configured memory limit.

**Acceptance Criteria:**
- On startup, the node queries local GPU resources via the Resource Monitor
- The node sends a Register RPC to the Coordinator with: node_id, address, grpc_address, capacity (including user-configured memory_limit), and empty layers_hosted
- The node transitions to WaitingAssignment state after successful registration

### 7.2 Layer Assignment Acceptance
The Worker Node MUST accept layer assignments from the Coordinator and load the corresponding shard.

**Acceptance Criteria:**
- On receiving AcceptLayerAssignment, the node stores the assignment in the Layer Assignment Registry
- The node triggers the Shard Manager to load the assigned layers with the specified dtype
- After successful load and validation, the node sends ConfirmReady to the Coordinator
- The node transitions to Ready state

### 7.3 Serving State
The Worker Node MUST process Forward requests while in the Serving state.

**Acceptance Criteria:**
- The node accepts incoming TCP connections and processes Forward messages
- For each Forward message: read header, validate, read payload, reconstruct tensor, run forward pass, serialize output, send to downstream node (or return to client if final)
- The node sends periodic heartbeats to the Coordinator with current resource metrics

### 7.4 Graceful Shutdown
The Worker Node MUST support graceful shutdown by draining in-flight requests.

**Acceptance Criteria:**
- On shutdown signal, the node stops accepting new Forward requests
- The node waits for all in-flight forward passes to complete
- The Shard Manager unloads the shard and frees GPU memory
- The Connection Pool closes all connections
- The node transitions to terminated state

### 7.5 Failure Reporting
The Worker Node MUST report downstream node failures to the Coordinator.

**Acceptance Criteria:**
- When a TCP send to the downstream node fails, the node calls ReportFailure on the Coordinator via gRPC
- The node receives RerouteInfo with the backup node's address
- The node retries the send to the backup node (single retry)
- If the backup also fails, the request is marked as FAILED and an ERROR message is sent back upstream

## Requirement 8: Coordinator Integration (Context — Other Team)

### 8.1 Node Registration Handling
The Coordinator MUST accept node registrations and maintain a live registry.

**Acceptance Criteria:**
- Register RPC adds the node to the registry with its capacity and address information
- The layer→node mapping is updated when layer assignments are made
- Deregister RPC removes the node and triggers backup reassignment for affected layers

### 8.2 Health Monitoring
The Coordinator MUST monitor node health via heartbeats.

**Acceptance Criteria:**
- Heartbeat RPC updates the node's last_seen timestamp and load metrics
- Nodes that miss heartbeats beyond the threshold (e.g., 3 consecutive at 5-second intervals) are marked UNHEALTHY
- UNHEALTHY nodes are excluded from new execution paths

### 8.3 Execution Path Building
The Coordinator MUST build ordered execution paths for inference requests.

**Acceptance Criteria:**
- RequestRoute returns an ordered list of healthy nodes covering all model layers
- Each node in the path has a contiguous, non-overlapping layer range
- The path includes backup node information for each layer range
- The path is assigned a unique request_id

### 8.4 Priority Queue Scheduling
The Coordinator MUST schedule requests using a priority queue.

**Acceptance Criteria:**
- Requests are scored using: priority = α * compute_contributed + β * wait_time
- Dequeue always returns the highest-priority request
- Wait time increases over time, ensuring fairness for long-waiting requests
- Queue rejects new requests when at maximum capacity (e.g., 100 for POC)

## Requirement 9: Client Integration (Context — Other Team)

### 9.1 Inference Submission
The Client MUST handle the end-to-end inference flow.

**Acceptance Criteria:**
- Client tokenizes raw text input into token IDs
- Client runs the embedding layer locally to produce initial hidden states
- Client requests an execution path from the Coordinator via gRPC
- Client sends initial hidden states to the first node in the path via TCP (using the binary protocol)
- Client receives final logits from the last node via TCP
- Client decodes logits into output text

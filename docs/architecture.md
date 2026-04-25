# Architecture Overview

## System Design

MeshRun uses a centrally scheduled, distributed pipeline architecture with strict separation between control plane and data plane.

- **Coordinator (Control Plane)**: Handles node registration, health monitoring, layer assignment, request scheduling, route building, and session key distribution via gRPC. Never touches inference data.
- **Worker Nodes (Data Plane)**: Host contiguous blocks of transformer layers and execute forward passes. Tensor data flows directly between workers over persistent TCP connections using a custom binary protocol with AES-256-GCM encryption.
- **Client**: Tokenizes input, runs the embedding layer locally, acquires a route and session key from the Coordinator, sends encrypted hidden states to the first pipeline node, and decodes encrypted logits from the last node.

The Coordinator is only in the path during initial routing and key distribution. Once a request is dispatched, tensor data flows directly between Worker Nodes.

## Data Flow

```
Client → Coordinator (gRPC: get execution path + session key)
Client → Node A (TCP/AES-256-GCM: encrypted hidden states)
Node A → Node B (TCP/AES-256-GCM: encrypted hidden states after layers 0-9)
Node B → Node C (TCP/AES-256-GCM: encrypted hidden states after layers 10-19)
Node C → Client (TCP/AES-256-GCM: encrypted logits after layers 20-29)
```

## Component Map

### Coordinator Server (`meshrun/coordinator/`)

| Component      | File             | Purpose                                                              |
| -------------- | ---------------- | -------------------------------------------------------------------- |
| gRPC Servicer  | `server.py`      | Exposes all Coordinator RPCs, wires to internal components           |
| Node Registry  | `registry.py`    | Live registry of worker nodes, status tracking, thread-safe          |
| Health Tracker | `registry.py`    | Background thread monitoring heartbeats, marking unhealthy/dead      |
| Scheduler      | `scheduler.py`   | Layer assignment algorithm, route building, priority queue           |
| Key Manager    | `key_manager.py` | AES-256 session key generation, storage, rotation per model pipeline |
| Proto Stubs    | `proto/`         | gRPC service definition and generated Python stubs                   |

### Worker Node (`meshrun/worker/`)

| Component                 | File                    | Purpose                                                             |
| ------------------------- | ----------------------- | ------------------------------------------------------------------- |
| Message Handler           | `protocol.py`           | TCP binary protocol: header serialization, framing, tensor I/O      |
| Secure Protocol           | `protocol.py`           | AES-256-GCM authenticated encryption for data plane messages        |
| Connection Pool           | `connection_pool.py`    | Persistent TCP connections to downstream nodes, incoming acceptance |
| Shard Manager             | `shard_manager.py`      | Selective download of model weights via HTTP Range, GPU loading     |
| Layer Engine              | `layer_engine.py`       | Forward pass execution through hosted transformer layers            |
| Resource Monitor          | `resource_monitor.py`   | GPU memory/utilization tracking, heartbeat snapshots                |
| Layer Assignment Registry | `layer_registry.py`     | Stores current layer assignment and pipeline topology               |
| Coordinator Client        | `coordinator_client.py` | gRPC client for Coordinator RPCs (Register, Heartbeat, etc.)        |
| Worker Node               | `node.py`               | Full lifecycle orchestration: startup → registration → serving      |
| Serving Loop              | `serving.py`            | Encrypted request processing: read → forward → send downstream      |

### Inference Client (`meshrun/client/`)

| Component        | File           | Purpose                                                            |
| ---------------- | -------------- | ------------------------------------------------------------------ |
| Tokenizer        | `tokenizer.py` | HuggingFace AutoTokenizer loading, tokenize/detokenize             |
| Embedding Engine | `tokenizer.py` | Selective download of embed_tokens weights, local embedding        |
| Secure Transport | `transport.py` | Encrypted TCP send/receive using protocol.py's secure functions    |
| Client           | `client.py`    | End-to-end orchestration: tokenize → embed → route → send → decode |

## Worker Node Lifecycle

```
Initializing → Registering → WaitingAssignment → LoadingShard → Validating → Ready → Serving → Draining
```

1. **Initializing**: Query local GPU resources via Resource Monitor, generate node_id
2. **Registering**: Send Register RPC to Coordinator with capacity info
3. **WaitingAssignment**: Wait for layer assignment from Coordinator
4. **LoadingShard**: Shard Manager selectively downloads assigned layer weights via HTTP Range requests and loads to GPU
5. **Validating**: Verify loaded shard matches expected configuration (layer count, dtype, hidden dims)
6. **Ready**: Send ConfirmReady to Coordinator, build Layer Engine, run warm-up
7. **Serving**: Process encrypted Forward requests via Message Handler → Layer Engine → Connection Pool pipeline, send heartbeats
8. **Draining**: Graceful shutdown — drain in-flight requests, unload shard, close connections

See [Worker Node Lifecycle](worker-lifecycle.md) for implementation details.

## Coordinator Internals

### Layer Assignment Algorithm

Static assignment for POC. The Coordinator computes contiguous layer blocks based on GPU capacity:

1. Collect all HEALTHY nodes with their `memory_limit_mb`
2. Sort by usable memory descending (subtract 800 MB framework overhead)
3. Estimate per-layer memory: ~200 MB (fp16) or ~100 MB (int8)
4. Greedily assign contiguous blocks until all layers are covered
5. Assign backup nodes for each primary range (if spare capacity exists)
6. Generate a pipeline session key via Key Manager

### Route Building

1. Look up the Layer Map for the requested model
2. For each layer range, select primary node if HEALTHY, otherwise backup
3. Build ordered list of `(node_address, layer_start, layer_end)`
4. Include the pipeline session key
5. Assign unique request_id

### Priority Queue

- Scoring: `priority = 0.7 * compute_contributed + 0.3 * wait_time`
- Max queue depth: 100 (configurable)
- Wait time increases linearly with time since enqueue

See [Coordinator Server](coordinator.md) for full details.

## Security Model

Data plane messages are encrypted with AES-256-GCM authenticated encryption:

- Session keys are generated per-pipeline by the Coordinator's Key Manager
- Keys are distributed to workers during layer assignment via gRPC
- Keys are included in `RequestRoute` responses so clients can encrypt traffic
- Every TCP message (header + payload) is encrypted with a fresh 12-byte nonce
- Wire format: `[4-byte length][12-byte nonce][ciphertext][16-byte GCM tag]`
- Tampered or replayed messages are rejected via GCM authentication

See [TCP Binary Protocol — Security Layer](protocol.md#security-layer--aes-256-gcm-encryption) for wire format details.

## Fault Tolerance

- Single retry to backup node on downstream failure
- Coordinator tracks backup nodes for each layer range
- Fail-fast after backup exhausted — no cascading retries
- Health monitoring via periodic heartbeats (missed threshold → UNHEALTHY → DEAD)
- Worker nodes report downstream failures to Coordinator via `ReportFailure` RPC
- Coordinator returns `RerouteInfo` with backup node address

## Layer Assignment Memory Budget

| Dtype | Per-Layer Memory (approx) | 5-Layer Block | 10-Layer Block |
| ----- | ------------------------- | ------------- | -------------- |
| fp16  | ~200 MB                   | ~1.0 GB       | ~2.0 GB        |
| int8  | ~100 MB                   | ~0.5 GB       | ~1.0 GB        |

Framework overhead: ~500-800 MB (PyTorch/CUDA context, subtracted from usable memory).

## Scope Boundaries

- Target: 3-5 nodes, 3-6 layers each, static partitioning
- Blocking I/O acceptable for POC scope
- gRPC control plane is unencrypted (`insecure_channel`) for POC
- Key rotation requires re-assignment (out of scope for POC)

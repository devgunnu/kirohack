# Architecture Overview

## System Design

MeshRun uses a centrally scheduled, distributed pipeline architecture with strict separation between control plane and data plane.

- **Coordinator (Control Plane)**: Handles node registration, health monitoring, request scheduling, and route building via gRPC. Never touches inference data.
- **Worker Nodes (Data Plane)**: Host contiguous blocks of transformer layers and execute forward passes. Tensor data flows directly between workers over persistent TCP connections using a custom binary protocol.
- **Client**: Tokenizes input, runs the embedding layer locally, sends hidden states to the first pipeline node, and decodes logits from the last node.

The Coordinator is only in the path during initial routing. Once a request is dispatched, tensor data flows directly between Worker Nodes.

## Data Flow

```
Client → Coordinator (gRPC: get execution path + session key)
Client → Node A (TCP/encrypted: hidden states)
Node A → Node B (TCP/encrypted: hidden states after layers 0-9)
Node B → Node C (TCP/encrypted: hidden states after layers 10-19)
Node C → Client (TCP/encrypted: logits after layers 20-29)
```

## Worker Node Internal Architecture

Each worker node contains these sub-components:

| Component                 | File                           | Status   | Purpose                                                                           |
| ------------------------- | ------------------------------ | -------- | --------------------------------------------------------------------------------- |
| Message Handler           | `worker/protocol.py`           | Complete | TCP binary protocol: header serialization, framing, tensor serialization          |
| Secure Protocol           | `worker/protocol.py`           | Complete | AES-256-GCM authenticated encryption for data plane messages                      |
| Connection Pool           | `worker/connection_pool.py`    | Complete | Persistent TCP connections to downstream nodes, incoming connection acceptance    |
| Shard Manager             | `worker/shard_manager.py`      | Complete | Selective download of model weights via HTTP Range requests, GPU loading, caching |
| Layer Engine              | `worker/layer_engine.py`       | Complete | Forward pass execution through hosted transformer layers                          |
| Resource Monitor          | `worker/resource_monitor.py`   | Complete | GPU memory/utilization tracking, heartbeat snapshots, memory limit alerting       |
| Layer Assignment Registry | `worker/layer_registry.py`     | Complete | Stores current layer assignment and pipeline topology                             |
| Coordinator Client        | `worker/coordinator_client.py` | Complete | gRPC client abstraction for Coordinator RPCs (Register, Heartbeat, etc.)          |
| Worker Node               | `worker/node.py`               | Complete | Full lifecycle orchestration: startup, registration, assignment, serving          |
| Serving Loop              | `worker/serving.py`            | Complete | Request processing pipeline: read → forward → send downstream                     |

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
7. **Serving**: Process Forward requests via Message Handler → Layer Engine → Connection Pool pipeline, send heartbeats with Resource Monitor snapshots
8. **Draining**: Graceful shutdown — drain in-flight requests, unload shard, close connections

See [Worker Node Lifecycle](worker-lifecycle.md) for implementation details.

## Layer Assignment Strategy

Static assignment for POC. The Coordinator assigns contiguous layer blocks to nodes based on GPU memory capacity:

- Each node gets a contiguous block of layers (no gaps, no interleaving)
- Layer blocks do not overlap between primary nodes
- Target: 3-5 nodes, 3-6 layers each, covering the full model

| Dtype | Per-Layer Memory (approx) | 5-Layer Block | 10-Layer Block |
| ----- | ------------------------- | ------------- | -------------- |
| fp16  | ~200 MB                   | ~1.0 GB       | ~2.0 GB        |
| int8  | ~100 MB                   | ~0.5 GB       | ~1.0 GB        |

## Security Model

Data plane messages are encrypted with AES-256-GCM authenticated encryption:

- Session keys are generated per-pipeline by the Coordinator
- Keys are distributed to workers during layer assignment via gRPC
- Every TCP message (header + payload) is encrypted with a fresh 12-byte nonce
- Wire format: `[4-byte length][12-byte nonce][ciphertext][16-byte GCM tag]`
- Tampered or replayed messages are rejected via GCM authentication

See [TCP Binary Protocol — Security Layer](protocol.md#security-layer--aes-256-gcm-encryption) for wire format details.

## Fault Tolerance

- Single retry to backup node on downstream failure
- Coordinator tracks backup nodes for each layer range
- Fail-fast after backup exhausted — no cascading retries
- Health monitoring via periodic heartbeats (missed threshold → UNHEALTHY)
- Worker nodes report downstream failures to Coordinator via `ReportFailure` RPC

## Scope Boundaries

- Coordinator server and Inference Client: planned (see `coordinator-and-client` spec)
- Primary focus: Worker Node data plane (complete)
- Target: 3-5 nodes, 3-6 layers each, static partitioning

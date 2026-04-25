# Architecture Overview

## System Design

MeshRun uses a centrally scheduled, distributed pipeline architecture with strict separation between control plane and data plane.

- **Coordinator (Control Plane)**: Handles node registration, health monitoring, request scheduling, and route building via gRPC. Never touches inference data.
- **Worker Nodes (Data Plane)**: Host contiguous blocks of transformer layers and execute forward passes. Tensor data flows directly between workers over persistent TCP connections using a custom binary protocol.
- **Client**: Tokenizes input, runs the embedding layer locally, sends hidden states to the first pipeline node, and decodes logits from the last node.

The Coordinator is only in the path during initial routing. Once a request is dispatched, tensor data flows directly between Worker Nodes.

## Data Flow

```
Client → Coordinator (gRPC: get execution path)
Client → Node A (TCP: hidden states)
Node A → Node B (TCP: hidden states after layers 0-9)
Node B → Node C (TCP: hidden states after layers 10-19)
Node C → Client (TCP: logits after layers 20-29)
```

## Worker Node Internal Architecture

Each worker node contains these sub-components:

| Component                 | File                                | Status      | Purpose                                                                           |
| ------------------------- | ----------------------------------- | ----------- | --------------------------------------------------------------------------------- |
| Message Handler           | `worker/protocol.py`                | Complete    | TCP binary protocol: header serialization, framing, tensor serialization          |
| Connection Pool           | `worker/connection_pool.py`         | Complete    | Persistent TCP connections to downstream nodes, incoming connection acceptance    |
| Shard Manager             | `worker/shard_manager.py`           | Complete    | Selective download of model weights via HTTP Range requests, GPU loading, caching |
| Layer Engine              | `worker/layer_engine.py`            | Complete    | Forward pass execution through hosted transformer layers                          |
| Resource Monitor          | `worker/resource_monitor.py`        | Complete    | GPU memory/utilization tracking, heartbeat snapshots, memory limit alerting       |
| Layer Assignment Registry | Planned: `worker/layer_registry.py` | Not Started | Stores current layer assignment and pipeline topology                             |

## Worker Node Lifecycle

```
Initializing → Registering → WaitingAssignment → LoadingShard → Validating → Ready → Serving → Draining
```

1. **Initializing**: Query local GPU resources via Resource Monitor, generate node_id
2. **Registering**: Send Register RPC to Coordinator with capacity info
3. **WaitingAssignment**: Wait for layer assignment from Coordinator
4. **LoadingShard**: Shard Manager selectively downloads assigned layer weights via HTTP Range requests and loads to GPU
5. **Validating**: Verify loaded shard matches expected configuration (layer count, dtype, hidden dims)
6. **Ready**: Send ConfirmReady to Coordinator, run Layer Engine warm-up
7. **Serving**: Process Forward requests via Message Handler → Layer Engine → Connection Pool pipeline, send heartbeats with Resource Monitor snapshots
8. **Draining**: Graceful shutdown — drain in-flight requests, unload shard, close connections

## Layer Assignment Strategy

Static assignment for POC. The Coordinator assigns contiguous layer blocks to nodes based on GPU memory capacity:

- Each node gets a contiguous block of layers (no gaps, no interleaving)
- Layer blocks do not overlap between primary nodes
- Target: 3-5 nodes, 3-6 layers each, covering the full model

| Dtype | Per-Layer Memory (approx) | 5-Layer Block | 10-Layer Block |
| ----- | ------------------------- | ------------- | -------------- |
| fp16  | ~200 MB                   | ~1.0 GB       | ~2.0 GB        |
| int8  | ~100 MB                   | ~0.5 GB       | ~1.0 GB        |

## Fault Tolerance

- Single retry to backup node on downstream failure
- Coordinator tracks backup nodes for each layer range
- Fail-fast after backup exhausted — no cascading retries
- Health monitoring via periodic heartbeats (missed threshold → UNHEALTHY)

## Scope Boundaries

- Security (auth, encryption): out of scope for POC
- Coordinator and Client: owned by another team
- Primary focus: Worker Node data plane

# Coordinator Server

The Coordinator is the central control plane for MeshRun. It manages the lifecycle of all worker nodes — accepting registrations, tracking health via heartbeats, computing static layer assignments, building execution routes, distributing AES-256-GCM session keys, and orchestrating fault tolerance through backup node assignment.

The Coordinator exposes a gRPC service and never touches inference tensor data.

Source: `meshrun/coordinator/`

## gRPC Service Definition

Defined in `meshrun/coordinator/proto/coordinator.proto`. The `CoordinatorService` exposes these RPCs:

| RPC Method              | Request → Response                                               | Description                                    |
| ----------------------- | ---------------------------------------------------------------- | ---------------------------------------------- |
| `Register`              | `RegisterRequest` → `RegisterResponse`                           | Worker node self-registers with capacity info  |
| `Heartbeat`             | `HeartbeatRequest` → `HeartbeatResponse`                         | Periodic health signal from worker node        |
| `ConfirmReady`          | `ConfirmReadyRequest` → `ConfirmReadyResponse`                   | Worker signals shard loaded and ready to serve |
| `Deregister`            | `DeregisterRequest` → `DeregisterResponse`                       | Worker node graceful removal                   |
| `RequestRoute`          | `RequestRouteRequest` → `RequestRouteResponse`                   | Client requests execution path + session key   |
| `ReportFailure`         | `ReportFailureRequest` → `ReportFailureResponse`                 | Worker reports downstream node failure         |
| `TriggerAssignment`     | `TriggerAssignmentRequest` → `TriggerAssignmentResponse`         | Admin triggers layer assignment computation    |
| `AcceptLayerAssignment` | `AcceptLayerAssignmentRequest` → `AcceptLayerAssignmentResponse` | Coordinator pushes assignment to worker        |

### Key Protobuf Messages

**Capacity**: `gpu_memory_total_mb`, `gpu_memory_free_mb`, `memory_limit_mb`, `gpu_utilization`

**RouteNode**: `node_id`, `address` (TCP host:port), `layer_start`, `layer_end`

**RerouteInfo**: `backup_addr` (TCP host:port of backup), `message`

**RequestRouteResponse**: `request_id`, `session_key` (32-byte AES-256), `nodes[]`, `backup_map[]`

**AcceptLayerAssignmentRequest**: `node_id`, `model_id`, `model_url`, `layer_start`, `layer_end`, `dtype`, `is_final_node`, `downstream_addr`, `upstream_addrs[]`, `session_key`

### Regenerating Stubs

```powershell
python -m grpc_tools.protoc `
  -I meshrun/coordinator/proto `
  --python_out=meshrun/coordinator/proto `
  --grpc_python_out=meshrun/coordinator/proto `
  meshrun/coordinator/proto/coordinator.proto
```

## Node Registry & Health Tracker

Source: `meshrun/coordinator/registry.py`

### Node Status Lifecycle

```
REGISTERED → HEALTHY → UNHEALTHY → DEAD
                ↑          |
                └──────────┘ (heartbeat resumed)
HEALTHY → DEREGISTERED (graceful removal)
```

- **REGISTERED**: Node has called `Register` but hasn't confirmed shard loaded
- **HEALTHY**: Node confirmed ready and is sending heartbeats
- **UNHEALTHY**: Missed heartbeats beyond threshold (default: 3 × 5s = 15s)
- **DEAD**: Extended absence (default: 5 × 5s = 25s), candidate for removal
- **DEREGISTERED**: Graceful removal via `Deregister` RPC

### NodeRegistry

Thread-safe registry of all worker nodes. All mutations protected by `threading.Lock`.

| Method                   | Description                                                |
| ------------------------ | ---------------------------------------------------------- |
| `register_node`          | Add node with status REGISTERED, reject duplicate node_ids |
| `deregister_node`        | Remove node, return success                                |
| `update_heartbeat`       | Update last_seen, gpu_utilization, memory_used, requests   |
| `mark_node_healthy`      | Transition REGISTERED → HEALTHY (after ConfirmReady)       |
| `update_node_assignment` | Store layer_start/layer_end on the node entry              |
| `get_node`               | Lookup single node by ID                                   |
| `get_all_healthy_nodes`  | Return only HEALTHY nodes                                  |
| `get_all_nodes`          | Return all nodes regardless of status                      |

### HealthTracker

Background daemon thread that runs every `heartbeat_interval_s` (default 5s):

- Checks `now - last_seen` against `missed_threshold` (default 3) and `dead_threshold` (default 5)
- Transitions: HEALTHY → UNHEALTHY → DEAD
- Recovers: UNHEALTHY → HEALTHY when heartbeats resume

```python
tracker = HealthTracker(registry, heartbeat_interval_s=5.0, missed_threshold=3, dead_threshold=5)
tracker.start()
# ... running ...
tracker.stop()
```

## Scheduler

Source: `meshrun/coordinator/scheduler.py`

### Layer Assignment

`compute_assignments(model_id, total_layers, dtype, nodes, key_manager)` implements the greedy contiguous assignment algorithm:

1. Sort nodes by usable memory descending (`memory_limit_mb - 800 MB` framework overhead)
2. Estimate per-layer memory: 200 MB (fp16) or 100 MB (int8)
3. Greedily assign contiguous blocks to each node
4. Validate full layer coverage (every layer 0 to total_layers-1 assigned)
5. Assign backup nodes for each primary range (spare capacity check)
6. Generate pipeline session key via Key Manager

Returns an `AssignmentPlan` containing `list[LayerMapEntry]` and the `session_key`.

Raises `InsufficientCapacityError` if nodes can't cover all layers.

### Layer Map

Stores `layer_range → (primary_node, backup_node)` mappings.

| Method                       | Description                                |
| ---------------------------- | ------------------------------------------ |
| `get_primary_node_for_layer` | Which node hosts a specific layer index    |
| `get_backup_for_range`       | Backup node for a given layer range        |
| `get_all_entries`            | All layer map entries                      |
| `set_entries`                | Replace all entries (after new assignment) |

### Route Building

`build_route(model_id, layer_map, registry, key_manager)`:

1. Iterate layer map entries in order
2. Select primary node if HEALTHY, else backup
3. Include session key from Key Manager
4. Assign unique request_id
5. Return `ExecutionPath` with nodes list, session_key, and backup_map

Raises `RouteError` if any layer range has no healthy primary or backup.

### Priority Queue

`PriorityQueue(max_depth=100, alpha=0.7, beta=0.3)`:

- `enqueue(request_id, client_id, model_id, compute_contributed)` — add with timestamp
- `dequeue()` — re-score all entries, return highest priority
- Scoring: `priority = alpha * compute_contributed + beta * wait_time`
- Raises `QueueFullError` when at capacity

### Failure Handling

`handle_failure(request_id, failed_node_id, layer_map, registry)`:

- Looks up the failed node's layer range
- Returns `RerouteInfo` with backup node address (or None if no backup)

## Key Manager

Source: `meshrun/coordinator/key_manager.py`

Thread-safe store of `model_id → session_key` mappings. Keys are 32-byte AES-256, generated via `os.urandom(32)`.

| Method                  | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `generate_pipeline_key` | Create and store a new 32-byte key for a model |
| `get_pipeline_key`      | Retrieve stored key (or None)                  |
| `rotate_key`            | Replace with a fresh key, return new key       |
| `delete_key`            | Remove key, return True if existed             |

Keys are stored in-memory only (no persistence for POC). The gRPC channel is unencrypted for POC.

## Coordinator Server

Source: `meshrun/coordinator/server.py`

### CoordinatorServicer

Implements the generated `CoordinatorServiceServicer`. Each RPC method translates protobuf messages to/from internal Python objects and delegates to the registry, scheduler, and key manager.

### CoordinatorServer

Manages the full server lifecycle:

```python
from meshrun.coordinator.server import CoordinatorServer

server = CoordinatorServer(
    host="0.0.0.0",
    port=50051,
    heartbeat_interval_s=5.0,
    missed_threshold=3,
    dead_threshold=5,
)

server.start()
# ... running ...
server.stop(grace=5.0)
```

**Properties**: `registry`, `key_manager`, `layer_map`, `priority_queue`, `servicer`

### Startup Sequence

1. Initialize NodeRegistry, KeyManager, LayerMap, PriorityQueue
2. Create CoordinatorServicer wiring all components
3. Start HealthTracker background thread
4. Start gRPC server on configured host:port

### Shutdown Sequence

1. Stop HealthTracker background thread
2. Stop gRPC server with grace period for in-flight RPCs
3. Clear registry and key manager state

## End-to-End Flow

1. Worker nodes start and call `Register` RPC
2. Admin calls `TriggerAssignment` with model_id, total_layers, dtype
3. Coordinator computes assignments, generates session key
4. Coordinator pushes `AcceptLayerAssignment` to each worker (with session key)
5. Workers load shards, call `ConfirmReady`
6. Workers begin sending `Heartbeat` RPCs periodically
7. Client calls `RequestRoute` → receives execution path + session key
8. Client sends encrypted FORWARD to first node
9. Tensors flow through pipeline (all encrypted with shared session key)
10. Final node returns encrypted RESULT to client

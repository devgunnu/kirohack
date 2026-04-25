# Worker Node Lifecycle

The worker node lifecycle is orchestrated by `WorkerNode` in `meshrun/worker/node.py`, with request processing handled by `ServingLoop` in `meshrun/worker/serving.py` and Coordinator communication via `CoordinatorClient` in `meshrun/worker/coordinator_client.py`.

## State Machine

```
Initializing → Registering → WaitingAssignment → LoadingShard → Validating → Ready → Serving → Draining
```

Any state can transition to `Error` on unrecoverable failures. From `Error`, the node can retry registration.

### Valid State Transitions

| From              | To                | Trigger                                    |
| ----------------- | ----------------- | ------------------------------------------ |
| Initializing      | Registering       | GPU resources queried, node_id generated   |
| Registering       | WaitingAssignment | Register RPC succeeds                      |
| WaitingAssignment | LoadingShard      | Layer assignment received from Coordinator |
| LoadingShard      | Validating        | Shard Manager finishes loading to GPU      |
| Validating        | Ready             | Shard validation passes, ConfirmReady sent |
| Ready             | Serving           | Serving loop and heartbeat sender started  |
| Serving           | Draining          | Shutdown signal received                   |
| Any               | Error             | Unrecoverable failure                      |
| Error             | Registering       | Retry registration                         |

## WorkerNode

Source: `meshrun/worker/node.py`

The `WorkerNode` class orchestrates the full lifecycle. It wires together all sub-components: Resource Monitor, Connection Pool, Layer Registry, Shard Manager, Layer Engine, Coordinator Client, Heartbeat Sender, and Serving Loop.

### Configuration

```python
from meshrun.worker.node import WorkerNode, NodeConfig

config = NodeConfig(
    address="192.168.1.10:9000",        # TCP data plane address
    grpc_address="192.168.1.10:50051",  # gRPC address for Coordinator
    coordinator_address="10.0.0.1:50051",
    gpu_memory_limit_mb=6000,
    poll_interval_s=1.0,
)

node = WorkerNode(config)
```

### Lifecycle Methods

| Method                         | Description                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------- |
| `startup()`                    | Initialize Resource Monitor, query GPU, generate node_id → Initializing      |
| `register_with_coordinator()`  | Send Register RPC → Registering → WaitingAssignment                          |
| `accept_layer_assignment(...)` | Store assignment, load shard, validate → LoadingShard → Validating → Ready   |
| `confirm_ready()`              | Send ConfirmReady RPC to Coordinator                                         |
| `build_engine_and_serve(...)`  | Build Layer Engine, start serving loop and heartbeat → Serving               |
| `start_serving(...)`           | Start the serving loop (accepts TCP connections, processes Forward messages) |
| `start_heartbeat()`            | Start periodic heartbeat sender to Coordinator                               |
| `stop_heartbeat()`             | Stop the heartbeat sender                                                    |
| `run_lifecycle(...)`           | Run the full lifecycle end-to-end (startup → register → serve)               |

### Full Lifecycle Example

```python
from meshrun.worker.node import WorkerNode, NodeConfig

config = NodeConfig(
    address="0.0.0.0:9000",
    grpc_address="0.0.0.0:50051",
    coordinator_address="10.0.0.1:50051",
    gpu_memory_limit_mb=6000,
)

node = WorkerNode(config)

# Option 1: Step-by-step
capacity = node.startup()
node.register_with_coordinator()
# ... wait for assignment from Coordinator ...
node.accept_layer_assignment(
    model_id="llama-3b",
    model_url="https://example.com/model.safetensors",
    layer_start=5, layer_end=9,
    dtype=1,  # FP16
    is_final_node=False,
    downstream_node="192.168.1.11:9000",
    upstream_nodes=["192.168.1.9:9000"],
)
node.build_engine_and_serve(hidden_dim=4096)

# Option 2: Automated (blocks until shutdown)
# node.run_lifecycle(model_id="llama-3b", ...)
```

---

## Heartbeat Sender

Source: `meshrun/worker/node.py` (class `HeartbeatSender`)

Sends periodic heartbeat RPCs to the Coordinator with Resource Monitor snapshots. Runs on a background daemon thread.

### Configuration

| Parameter               | Default | Description                              |
| ----------------------- | ------- | ---------------------------------------- |
| `interval_s`            | 5.0     | Seconds between heartbeat sends          |
| `max_consecutive_fails` | 3       | Failures before logging critical warning |

### Behavior

- Sends `HeartbeatRequest` with node_id, gpu_utilization, memory_used_mb, active_requests
- Tracks consecutive failures; logs warnings on repeated failures
- Runs as a daemon thread (auto-terminates when main thread exits)

---

## Serving Loop

Source: `meshrun/worker/serving.py`

Handles the encrypted request processing pipeline: accepts TCP connections, reads encrypted Forward messages, runs the Layer Engine, and sends encrypted results downstream.

### Key Classes

- `ServingConfig` — Configuration for the serving loop (listen address, timeouts)
- `ServingStats` — Tracks success/failure counts for processed requests
- `ServingLoop` — Main serving loop that accepts connections and processes requests

### Request Processing Pipeline (Encrypted)

For each incoming TCP connection:

1. `read_message_secure(sock, session_key)` — Read and decrypt header + tensor payload from upstream
2. Validate message type is FORWARD
3. Convert flat tensor list to PyTorch tensor
4. `forward(engine, tensor, step_id)` — Run Layer Engine forward pass
5. Build response header (RESULT if final node, FORWARD otherwise)
6. Convert output tensor to flat list
7. `write_message_secure(downstream_sock, header, data, session_key)` — Encrypt and send to downstream node or back to client

All data plane reads/writes use the AES-256-GCM secure variants with the session key distributed by the Coordinator during layer assignment.

### Failure Handling

- If downstream send fails, reports failure to Coordinator via `ReportFailure` RPC
- Receives `RerouteInfo` with backup node address
- Retries encrypted send to backup node (single retry)
- If backup also fails, sends encrypted ERROR message back upstream

### Usage

```python
from meshrun.worker.serving import ServingLoop, ServingConfig

config = ServingConfig(
    listen_addr=("0.0.0.0", 9000),
    recv_timeout_s=30.0,
)

loop = ServingLoop(
    config=config,
    connection_pool=pool,
    layer_engine=engine,
    layer_registry=registry,
    resource_monitor=monitor,
    coordinator_client=client,
    session_key=session_key,  # 32-byte AES-256 key from Coordinator
)

loop.start()
# ... serving encrypted requests ...
loop.stop()
```

---

## Coordinator Client

Source: `meshrun/worker/coordinator_client.py`

Abstract client interface for communicating with the Coordinator via gRPC. Provides two implementations:

- `GrpcCoordinatorClient` — Real gRPC client (for production use, pending proto generation)
- `StubCoordinatorClient` — In-memory stub for testing and development

### RPC Methods

| Method             | Request Type           | Response Type           | Description                                |
| ------------------ | ---------------------- | ----------------------- | ------------------------------------------ |
| `register()`       | `RegisterRequest`      | `RegisterResponse`      | Register node with Coordinator             |
| `confirm_ready()`  | `ConfirmReadyRequest`  | `ConfirmReadyResponse`  | Signal that shard is loaded and validated  |
| `heartbeat()`      | `HeartbeatRequest`     | `HeartbeatResponse`     | Send periodic health/metrics update        |
| `report_failure()` | `ReportFailureRequest` | `ReportFailureResponse` | Report downstream node failure, get backup |

### Data Classes

- `CapacityInfo` — Node capacity (gpu_memory_total_mb, gpu_memory_free_mb, gpu_memory_limit_mb)
- `RegisterRequest` — Registration payload (node_id, address, grpc_address, capacity)
- `RegisterResponse` — Registration result (success, message)
- `HeartbeatRequest` — Heartbeat payload (node_id, gpu_utilization, memory_used_mb, active_requests)
- `RerouteInfo` — Backup node info (backup_node_id, backup_address)
- `ReportFailureResponse` — Failure report result (success, reroute_info)

### Usage

```python
from meshrun.worker.coordinator_client import (
    StubCoordinatorClient, RegisterRequest, CapacityInfo,
)

# For development/testing
client = StubCoordinatorClient(
    register_response=RegisterResponse(success=True, message="OK"),
)

# Register
response = client.register(RegisterRequest(
    node_id="node-abc",
    address="192.168.1.10:9000",
    grpc_address="192.168.1.10:50051",
    capacity=CapacityInfo(
        gpu_memory_total_mb=8000,
        gpu_memory_free_mb=7200,
        gpu_memory_limit_mb=6000,
    ),
))
```

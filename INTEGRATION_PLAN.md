# MeshRun Distributed Integration Plan

## Goal

Make MeshRun run end-to-end across multiple physical machines connected via Tailscale, with:
- A Coordinator process on one machine (gRPC control plane on port 50051)
- Worker Node processes on GPU machines (gRPC control plane + TCP data plane)
- A Client that submits inference and gets text back
- Auto-assignment: when a worker joins, the Coordinator recomputes layer assignments and pushes them to all workers
- CLI entry points for all three roles

---

## Current State of Each Component

### What Works (tested, implemented)
- `meshrun/worker/protocol.py` — TCP binary protocol, header pack/unpack, tensor serialization, AES-256-GCM encryption
- `meshrun/worker/connection_pool.py` — TCP connection management, accept_incoming, retry logic
- `meshrun/worker/shard_manager.py` — HTTP Range download of safetensors, caching, tensor reconstruction
- `meshrun/worker/layer_engine.py` — Transformer layer forward pass (RMS norm, attention, MLP)
- `meshrun/worker/layer_registry.py` — Thread-safe layer assignment storage
- `meshrun/worker/resource_monitor.py` — GPU metrics polling via PyTorch
- `meshrun/worker/serving.py` — TCP serving loop, message handling, downstream forwarding
- `meshrun/worker/node.py` — Full worker lifecycle state machine (INITIALIZING → SERVING)
- `meshrun/worker/coordinator_client.py` — gRPC client for Register, Heartbeat, ConfirmReady, ReportFailure
- `meshrun/coordinator/registry.py` — Node registry with health tracking
- `meshrun/coordinator/scheduler.py` — Layer assignment algorithm, route building, failure handling, priority queue
- `meshrun/coordinator/key_manager.py` — AES-256 session key generation per pipeline
- `meshrun/coordinator/server.py` — gRPC servicer wiring all coordinator components
- `meshrun/coordinator/proto/` — Protobuf definitions and generated stubs
- `meshrun/client/client.py` — Full inference client (tokenize → embed → route → send → receive → decode)
- `meshrun/client/transport.py` — Encrypted TCP transport for client
- `meshrun/client/tokenizer.py` — HuggingFace tokenizer + embedding weight download

### What's Broken / Missing
1. No CLI entry point to start the Coordinator
2. No CLI entry point to start a Worker Node
3. No CLI entry point for real inference submission (current `meshrun submit` calls mocks)
4. Coordinator `Register` handler does NOT trigger auto-assignment
5. Coordinator `TriggerAssignment` computes assignments but does NOT push them to workers
6. No worker-side gRPC server to receive `AcceptLayerAssignment` push from Coordinator
7. Workers advertise `0.0.0.0` as their address — unusable over a network
8. `meshrun/app/client/inference.py` is 100% mock data
9. `meshrun/app/display/` modules use hardcoded data instead of accepting parameters
10. Dashboard streams fake events
11. Kiro extension sidebar uses mock data
12. Two conflicting `pyproject.toml` files with incompatible entry points
13. CLI imports use `from app.` which only works when installed from `meshrun/` directory

---

## Change Plan

### CHANGE 1: Unified pyproject.toml at Root

**File:** `pyproject.toml` (root)

**Current state:** Declares `distributed-inference-server` with only core deps (grpcio, torch, etc.). No build system, no entry points.

**Required changes:**

Replace the entire file with a unified configuration that:
1. Merges dependencies from both existing pyproject.toml files
2. Adds a build system
3. Declares CLI entry points for coordinator, worker, and client
4. Discovers all packages under `meshrun/`

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "meshrun"
version = "0.1.0"
description = "Distributed AI inference on compute you already own."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    # Core dependencies (from root pyproject.toml)
    "grpcio>=1.70.0",
    "grpcio-tools>=1.70.0",
    "cryptography>=44.0.0",
    "torch>=2.6.0",
    "transformers>=4.48.0",
    # CLI dependencies (from meshrun/pyproject.toml)
    "typer[all]>=0.12.0",
    "rich>=13.0.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "websockets>=12.0",
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "toml>=0.10.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=9.0.0",
    "hypothesis>=6.150.0",
    "pytest-asyncio>=0.25.0",
    "ruff>=0.15.0",
    "mypy>=1.15.0",
]

[project.scripts]
meshrun = "meshrun.app.cli.main:app"

[tool.setuptools.packages.find]
where = ["."]
include = ["meshrun*"]
```

**Delete:** `meshrun/pyproject.toml` and `meshrun/app/pyproject.toml` (no longer needed)

---

### CHANGE 2: Fix CLI Import Paths

**Files:** All files under `meshrun/app/`

**Current state:** All imports use `from app.xxx import yyy`

**Required changes:** Change all imports to use `from meshrun.app.xxx import yyy`

Files to update:
- `meshrun/app/cli/main.py` — change `from app.cli.commands import ...` to `from meshrun.app.cli.commands import ...`
- `meshrun/app/cli/commands/submit.py` — change `from app.client.inference import ...` and `from app.display.panels import ...` and `from app.display.spinners import ...`
- `meshrun/app/cli/commands/status.py` — change `from app.client.inference import ...` and `from app.display.panels import ...` and `from app.display.spinners import ...` and `from app.display.tables import ...`
- `meshrun/app/cli/commands/join.py` — change `from app.client.inference import ...` and `from app.display.panels import ...` and `from app.display.spinners import ...`
- `meshrun/app/cli/commands/leave.py` — change `from app.client.inference import ...` and `from app.display.spinners import ...`
- `meshrun/app/cli/commands/nodes.py` — change `from app.client.inference import ...` and `from app.display.spinners import ...` and `from app.display.tables import ...`
- `meshrun/app/cli/commands/credits.py` — change `from app.client.inference import ...` and `from app.config import ...` and `from app.display.panels import ...` and `from app.display.spinners import ...` and `from app.display.tables import ...`
- `meshrun/app/cli/commands/dashboard.py` — change `from app.config import ...` and `from app.display.spinners import ...` and `from app.dashboard.server import ...`
- `meshrun/app/config.py` — no imports to change
- `meshrun/app/display/panels.py` — change `from app.display.spinners import ...`
- `meshrun/app/display/tables.py` — change `from app.display.spinners import ...`
- `meshrun/app/dashboard/server.py` — change `from app.dashboard.events import ...`
- `meshrun/app/dashboard/events.py` — no imports to change

---

### CHANGE 3: Add CLI Command for Coordinator

**New file:** `meshrun/app/cli/commands/coordinator.py`

```python
"""Start the MeshRun Coordinator server."""

import typer
from rich import print as rprint

app = typer.Typer(help="Start the MeshRun Coordinator server.")


@app.callback(invoke_without_command=True)
def coordinator(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address for gRPC server."),
    port: int = typer.Option(50051, "--port", "-p", help="Port for gRPC server."),
    model_id: str = typer.Option("llama-3b", "--model", "-m", help="Default model ID for this cluster."),
    total_layers: int = typer.Option(28, "--layers", "-l", help="Total transformer layers in the model."),
    dtype: str = typer.Option("int8", "--dtype", "-d", help="Weight data type: fp16 or int8."),
    model_url: str = typer.Option("", "--model-url", "-u", help="HTTP URL to the safetensors model file."),
):
    """Start the Coordinator gRPC server."""
    import signal
    import sys
    
    from meshrun.coordinator.server import CoordinatorServer
    
    rprint(f"[bold cyan]Starting MeshRun Coordinator[/bold cyan]")
    rprint(f"  gRPC: {host}:{port}")
    rprint(f"  Model: {model_id} ({total_layers} layers, {dtype})")
    rprint(f"  Model URL: {model_url or '(not specified)'}")
    
    server = CoordinatorServer(host=host, port=port)
    
    # Store model config for auto-assignment
    server._model_config = {
        "model_id": model_id,
        "total_layers": total_layers,
        "dtype": dtype,
        "model_url": model_url,
    }
    
    server.start()
    
    rprint("[green]Coordinator started. Press Ctrl+C to stop.[/green]")
    
    def handle_sigterm(signum, frame):
        rprint("\n[yellow]Shutting down...[/yellow]")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    # Block forever
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
```

**Update:** `meshrun/app/cli/main.py` to register the new command:

```python
from meshrun.app.cli.commands import submit, status, join, leave, nodes, credits, dashboard, coordinator

# ... existing code ...
app.add_typer(coordinator.app, name="coordinator")
```

---

### CHANGE 4: Add CLI Command for Worker

**New file:** `meshrun/app/cli/commands/worker.py`

```python
"""Start a MeshRun Worker Node."""

import typer
from rich import print as rprint

app = typer.Typer(help="Start a MeshRun Worker Node.")


@app.callback(invoke_without_command=True)
def worker(
    coordinator: str = typer.Option("localhost:50051", "--coordinator", "-c", help="Coordinator gRPC address (host:port)."),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address for TCP data plane."),
    data_port: int = typer.Option(9100, "--data-port", "-d", help="TCP port for data plane."),
    grpc_port: int = typer.Option(50052, "--grpc-port", "-g", help="TCP port for gRPC control plane callbacks."),
    advertise: str = typer.Option("", "--advertise", "-a", help="Advertised address (host:port) for other nodes. Auto-detected if not specified."),
    gpu_memory_limit_mb: int = typer.Option(4096, "--gpu-limit", help="Maximum GPU memory to use (MB)."),
    device: int = typer.Option(0, "--device", help="CUDA device index."),
):
    """Start a MeshRun Worker Node."""
    import signal
    import sys
    import subprocess
    
    from meshrun.worker.node import NodeConfig, WorkerNode
    
    # Auto-detect advertise address if not specified
    if advertise:
        advertise_host, advertise_port = advertise.rsplit(":", 1)
        advertise_port = int(advertise_port)
    else:
        # Try to detect Tailscale IP
        advertise_host = _detect_tailscale_ip()
        if not advertise_host:
            # Fall back to hostname
            import socket
            advertise_host = socket.gethostbyname(socket.gethostname())
        advertise_port = data_port
    
    rprint(f"[bold cyan]Starting MeshRun Worker Node[/bold cyan]")
    rprint(f"  Coordinator: {coordinator}")
    rprint(f"  Data plane: {host}:{data_port}")
    rprint(f"  gRPC callbacks: {host}:{grpc_port}")
    rprint(f"  Advertised address: {advertise_host}:{advertise_port}")
    rprint(f"  GPU limit: {gpu_memory_limit_mb} MB, device {device}")
    
    config = NodeConfig(
        host=host,
        data_port=data_port,
        grpc_port=grpc_port,
        coordinator_address=coordinator,
        gpu_memory_limit_mb=gpu_memory_limit_mb,
        device_index=device,
    )
    
    node = WorkerNode(config=config)
    
    # Override the address property to return the advertised address
    node._advertise_host = advertise_host
    node._advertise_port = advertise_port
    
    # Monkey-patch the address property
    original_address = type(node).address.fget
    def patched_address(self):
        return f"{self._advertise_host}:{self._advertise_port}"
    type(node).address = property(patched_address)
    
    # Start the worker lifecycle
    # Phase 1: Startup (detect GPU, create ResourceMonitor)
    capacity = node.startup()
    rprint(f"[green]GPU detected: {capacity.gpu_memory_total_mb} MB total, {capacity.gpu_memory_free_mb} MB free[/green]")
    
    # Phase 2: Register with Coordinator
    response = node.register_with_coordinator()
    if response.status.name != "OK":
        rprint(f"[red]Registration failed: {response.message}[/red]")
        raise typer.Exit(1)
    rprint(f"[green]Registered with coordinator[/green]")
    
    # Phase 3: Wait for layer assignment (blocking)
    rprint("[yellow]Waiting for layer assignment from coordinator...[/yellow]")
    
    # The worker now needs to receive the AcceptLayerAssignment push from the coordinator.
    # This requires a gRPC server on the worker side (see CHANGE 7).
    # For now, we'll poll the coordinator or wait for the push.
    
    # ... rest of the lifecycle will be handled by the gRPC push handler ...
    
    rprint("[green]Worker started. Press Ctrl+C to stop.[/green]")
    
    def handle_sigterm(signum, frame):
        rprint("\n[yellow]Shutting down worker...[/yellow]")
        node.stop_heartbeat()
        # Additional cleanup
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    # Block forever (the serving loop runs in background threads)
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def _detect_tailscale_ip() -> str | None:
    """Try to detect the Tailscale IP address of this machine.
    
    Uses the `tailscale ip --4` command if available.
    Returns None if Tailscale is not installed or not connected.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["tailscale", "ip", "--4"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode == 0:
            ip = result.stdout.strip()
            if ip.startswith("100."):
                return ip
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
```

**Update:** `meshrun/app/cli/main.py` to register the new command:

```python
from meshrun.app.cli.commands import submit, status, join, leave, nodes, credits, dashboard, coordinator, worker

# ... existing code ...
app.add_typer(worker.app, name="worker")
```

---

### CHANGE 5: Add Model Config Storage to CoordinatorServer

**File:** `meshrun/coordinator/server.py`

**Current state:** CoordinatorServer has no storage for model config (model_id, total_layers, dtype, model_url).

**Required changes:**

Add to `CoordinatorServer.__init__`:

```python
# Model configuration (set via TriggerAssignment or CLI)
self._model_config: dict | None = None
```

Add property and setter:

```python
@property
def model_config(self) -> dict | None:
    """The current model configuration for auto-assignment."""
    return self._model_config

def set_model_config(self, model_id: str, total_layers: int, dtype: str, model_url: str) -> None:
    """Set the model configuration for auto-assignment."""
    self._model_config = {
        "model_id": model_id,
        "total_layers": total_layers,
        "dtype": dtype,
        "model_url": model_url,
    }
```

---

### CHANGE 6: Auto-Assignment on Worker Registration

**File:** `meshrun/coordinator/server.py`

**Current state:** The `Register` RPC handler just adds the node to the registry. No assignment is triggered.

**Required changes:**

Modify `CoordinatorServicer.Register` to trigger auto-assignment after successful registration:

```python
def Register(self, request: pb2.RegisterRequest, context: grpc.ServicerContext) -> pb2.RegisterResponse:
    """Worker node self-registers with capacity info."""
    try:
        cap = request.capacity
        self._registry.register_node(
            node_id=request.node_id,
            address=request.address,
            grpc_address=request.grpc_address,
            gpu_memory_total_mb=cap.gpu_memory_total_mb,
            gpu_memory_free_mb=cap.gpu_memory_free_mb,
            memory_limit_mb=cap.memory_limit_mb,
            gpu_utilization=cap.gpu_utilization,
        )
        
        # Trigger auto-assignment if model config is set
        self._maybe_trigger_assignment()
        
        return pb2.RegisterResponse(
            status=pb2.REGISTRATION_STATUS_OK,
            message=f"Node '{request.node_id}' registered successfully",
        )
    except DuplicateNodeError as exc:
        return pb2.RegisterResponse(
            status=pb2.REGISTRATION_STATUS_REJECTED,
            message=str(exc),
        )
    except Exception as exc:
        logger.exception("Register RPC failed for node '%s'", request.node_id)
        return pb2.RegisterResponse(
            status=pb2.REGISTRATION_STATUS_ERROR,
            message=f"Internal error: {exc}",
        )
```

Add helper method to `CoordinatorServicer`:

```python
def _maybe_trigger_assignment(self) -> None:
    """Check if we have model config and enough nodes, then trigger assignment."""
    # This needs access to the server's model_config, which we'll pass via the servicer
    pass  # Implementation in CHANGE 8
```

---

### CHANGE 7: Worker-Side gRPC Server for AcceptLayerAssignment

**New file:** `meshrun/worker/assignment_server.py`

This file implements a gRPC server on the worker that listens for `AcceptLayerAssignment` pushes from the Coordinator.

```python
"""Worker-side gRPC server for receiving layer assignment pushes from the Coordinator."""

from __future__ import annotations

import logging
from concurrent import futures

import grpc

from meshrun.coordinator.proto import coordinator_pb2 as pb2
from meshrun.coordinator.proto import coordinator_pb2_grpc

logger = logging.getLogger(__name__)


class WorkerAssignmentServicer(coordinator_pb2_grpc.CoordinatorServiceServicer):
    """Receives AcceptLayerAssignment pushes from the Coordinator."""
    
    def __init__(self, worker_node: "WorkerNode") -> None:
        self._worker = worker_node
    
    def AcceptLayerAssignment(
        self, 
        request: pb2.AcceptLayerAssignmentRequest, 
        context: grpc.ServicerContext
    ) -> pb2.AcceptLayerAssignmentResponse:
        """Handle layer assignment push from Coordinator."""
        from meshrun.worker.layer_registry import AssignmentDType
        
        logger.info(
            "Received layer assignment: model=%s, layers %d-%d, is_final=%s",
            request.model_id,
            request.layer_start,
            request.layer_end,
            request.is_final_node,
        )
        
        # Map proto DType to AssignmentDType
        dtype_map = {
            pb2.DTYPE_FP16: AssignmentDType.FP16,
            pb2.DTYPE_INT8: AssignmentDType.INT8,
        }
        dtype = dtype_map.get(request.dtype)
        if dtype is None:
            return pb2.AcceptLayerAssignmentResponse(
                acknowledged=False,
                message=f"Unsupported dtype: {request.dtype}",
            )
        
        try:
            # Call the worker's accept_layer_assignment method
            metadata = self._worker.accept_layer_assignment(
                model_id=request.model_id,
                model_url=request.model_url,
                layer_start=request.layer_start,
                layer_end=request.layer_end,
                dtype=dtype,
                is_final_node=request.is_final_node,
                downstream_node=request.downstream_addr if request.downstream_addr else None,
                upstream_nodes=tuple(request.upstream_addrs),
                session_key=bytes(request.session_key) if request.session_key else None,
            )
            
            # Check if shard loaded successfully
            from meshrun.worker.shard_manager import LoadStatus
            if metadata.load_status == LoadStatus.READY:
                # Send ConfirmReady to Coordinator
                self._worker.confirm_ready()
                
                # Build engine and start serving
                self._worker.build_engine_and_serve()
                
                return pb2.AcceptLayerAssignmentResponse(
                    acknowledged=True,
                    message=f"Layers {request.layer_start}-{request.layer_end} loaded and serving",
                )
            else:
                return pb2.AcceptLayerAssignmentResponse(
                    acknowledged=False,
                    message=f"Shard load failed: {metadata.error_message}",
                )
        except Exception as exc:
            logger.exception("Failed to accept layer assignment")
            return pb2.AcceptLayerAssignmentResponse(
                acknowledged=False,
                message=f"Error: {exc}",
            )


class WorkerAssignmentServer:
    """Manages the worker's gRPC server for receiving Coordinator pushes."""
    
    def __init__(
        self,
        worker_node: "WorkerNode",
        host: str = "0.0.0.0",
        port: int = 50052,
    ) -> None:
        self._worker = worker_node
        self._host = host
        self._port = port
        self._server: grpc.Server | None = None
    
    def start(self) -> None:
        """Start the gRPC server."""
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
        servicer = WorkerAssignmentServicer(self._worker)
        coordinator_pb2_grpc.add_CoordinatorServiceServicer_to_server(
            servicer, self._server
        )
        address = f"{self._host}:{self._port}"
        self._server.add_insecure_port(address)
        self._server.start()
        logger.info("Worker assignment server started on %s", address)
    
    def stop(self, grace: float = 5.0) -> None:
        """Stop the gRPC server."""
        if self._server:
            self._server.stop(grace=grace)
            logger.info("Worker assignment server stopped")
```

---

### CHANGE 8: Coordinator Pushes Assignments to Workers

**File:** `meshrun/coordinator/server.py`

**Current state:** `TriggerAssignment` computes assignments but doesn't push them.

**Required changes:**

Add a method to push assignments to all workers:

```python
def _push_assignments_to_workers(self, plan: AssignmentPlan) -> None:
    """Push AcceptLayerAssignment RPCs to all workers in the plan."""
    import grpc
    from meshrun.coordinator.proto import coordinator_pb2_grpc
    
    # Build a lookup of node_id -> grpc_address from the registry
    nodes = self._registry.get_all_nodes()
    grpc_addresses: dict[str, str] = {n.node_id: n.grpc_address for n in nodes}
    
    # Map dtype string to proto enum
    dtype_to_proto = {
        "fp16": pb2.DTYPE_FP16,
        "int8": pb2.DTYPE_INT8,
    }
    
    # Get model config
    model_config = self._model_config  # Set via CLI or TriggerAssignment
    if not model_config:
        logger.warning("No model config set, cannot push assignments")
        return
    
    proto_dtype = dtype_to_proto.get(model_config["dtype"], pb2.DTYPE_INT8)
    
    for entry in plan.assignments:
        grpc_addr = grpc_addresses.get(entry.primary_node_id)
        if not grpc_addr:
            logger.warning(
                "No gRPC address for node %s, skipping assignment push",
                entry.primary_node_id,
            )
            continue
        
        # Determine if this is the final node
        is_final = (entry.layer_end == model_config["total_layers"] - 1)
        
        # Find downstream node (next entry in the plan)
        downstream_addr = ""
        entries = list(plan.assignments)
        idx = entries.index(entry)
        if idx + 1 < len(entries):
            downstream_addr = entries[idx + 1].primary_address
        
        # Find upstream nodes (previous entry in the plan)
        upstream_addrs = []
        if idx > 0:
            upstream_addrs.append(entries[idx - 1].primary_address)
        
        # Build the request
        request = pb2.AcceptLayerAssignmentRequest(
            node_id=entry.primary_node_id,
            model_id=model_config["model_id"],
            model_url=model_config["model_url"],
            layer_start=entry.layer_start,
            layer_end=entry.layer_end,
            dtype=proto_dtype,
            is_final_node=is_final,
            downstream_addr=downstream_addr,
            upstream_addrs=upstream_addrs,
            session_key=plan.session_key,
        )
        
        # Push to the worker
        try:
            channel = grpc.insecure_channel(grpc_addr)
            stub = coordinator_pb2_grpc.CoordinatorServiceStub(channel)
            response = stub.AcceptLayerAssignment(request, timeout=30.0)
            channel.close()
            
            if response.acknowledged:
                logger.info(
                    "Pushed assignment to %s: layers %d-%d",
                    entry.primary_node_id,
                    entry.layer_start,
                    entry.layer_end,
                )
            else:
                logger.error(
                    "Worker %s rejected assignment: %s",
                    entry.primary_node_id,
                    response.message,
                )
        except Exception as exc:
            logger.error(
                "Failed to push assignment to %s at %s: %s",
                entry.primary_node_id,
                grpc_addr,
                exc,
            )
```

Update `_maybe_trigger_assignment`:

```python
def _maybe_trigger_assignment(self) -> None:
    """Check if we have model config and enough nodes, then trigger assignment."""
    # Access the server's model config through the servicer's reference
    # We need to pass the server reference to the servicer
    pass
```

**Important:** The `CoordinatorServicer` needs a reference to the `CoordinatorServer` to access `_model_config`. Update the constructor:

```python
def __init__(
    self,
    registry: NodeRegistry,
    health_tracker: HealthTracker,
    key_manager: KeyManager,
    layer_map: LayerMap,
    priority_queue: PriorityQueue,
    server: CoordinatorServer,  # Add this
) -> None:
    self._registry = registry
    self._health_tracker = health_tracker
    self._key_manager = key_manager
    self._layer_map = layer_map
    self._priority_queue = priority_queue
    self._server = server  # Store reference
```

And update `CoordinatorServer.__init__` to pass `self`:

```python
self._servicer = CoordinatorServicer(
    registry=self._registry,
    health_tracker=self._health_tracker,
    key_manager=self._key_manager,
    layer_map=self._layer_map,
    priority_queue=self._priority_queue,
    server=self,  # Pass self
)
```

Now implement `_maybe_trigger_assignment`:

```python
def _maybe_trigger_assignment(self) -> None:
    """Check if we have model config and enough nodes, then trigger assignment."""
    model_config = self._server._model_config
    if not model_config:
        logger.debug("No model config set, skipping auto-assignment")
        return
    
    healthy_nodes = self._registry.get_all_healthy_nodes()
    if not healthy_nodes:
        logger.debug("No healthy nodes, skipping auto-assignment")
        return
    
    try:
        plan = compute_assignments(
            model_id=model_config["model_id"],
            total_layers=model_config["total_layers"],
            dtype=model_config["dtype"],
            nodes=healthy_nodes,
            key_manager=self._key_manager,
        )
        self._layer_map.set_entries(list(plan.assignments))
        
        # Update registry with layer assignments
        for entry in plan.assignments:
            self._registry.update_node_assignment(
                node_id=entry.primary_node_id,
                layer_start=entry.layer_start,
                layer_end=entry.layer_end,
            )
        
        # Push assignments to all workers
        self._push_assignments_to_workers(plan)
        
        logger.info(
            "Auto-assigned %d layers across %d nodes",
            model_config["total_layers"],
            len(plan.assignments),
        )
    except InsufficientCapacityError as exc:
        logger.warning("Auto-assignment failed: %s", exc)
    except Exception as exc:
        logger.exception("Auto-assignment failed")
```

---

### CHANGE 9: Wire CLI Inference to Real Client

**File:** `meshrun/app/client/inference.py`

**Current state:** All functions return hardcoded mock data.

**Required changes:** Replace the entire file with real implementations that call `meshrun.client.InferenceClient` and the Coordinator gRPC API.

```python
"""
Real inference client for MeshRun CLI.

Wires the CLI commands to the actual backend components.
"""

from __future__ import annotations

import logging
from typing import Any

import grpc

from meshrun.client.client import InferenceClient
from meshrun.coordinator.proto import coordinator_pb2, coordinator_pb2_grpc
from meshrun.app.config import settings

logger = logging.getLogger(__name__)

# Global client instances (lazy-initialized)
_inference_client: InferenceClient | None = None
_coordinator_channel: grpc.Channel | None = None
_coordinator_stub: coordinator_pb2_grpc.CoordinatorServiceStub | None = None


def _get_coordinator_stub() -> coordinator_pb2_grpc.CoordinatorServiceStub:
    """Get or create the gRPC stub for the Coordinator."""
    global _coordinator_channel, _coordinator_stub
    if _coordinator_stub is None:
        # Parse coordinator address from settings (strip http:// prefix if present)
        addr = settings.coordinator_url.replace("http://", "").replace("https://", "")
        _coordinator_channel = grpc.insecure_channel(addr)
        _coordinator_stub = coordinator_pb2_grpc.CoordinatorServiceStub(_coordinator_channel)
    return _coordinator_stub


def _get_inference_client() -> InferenceClient:
    """Get or create the InferenceClient."""
    global _inference_client
    if _inference_client is None:
        addr = settings.coordinator_url.replace("http://", "").replace("https://", "")
        _inference_client = InferenceClient(
            coordinator_address=addr,
            model_name=settings.default_model,
            model_url="",  # Will be set by the coordinator
        )
        _inference_client.initialize()
    return _inference_client


def submit_inference_job(prompt: str, priority: str = "normal") -> dict:
    """Submit a synchronous inference job and return the result."""
    try:
        client = _get_inference_client()
        output = client.submit_inference(prompt)
        return {
            "job_id": "sync",
            "status": "completed",
            "output": output,
            "tokens_generated": 0,  # Not tracked in current impl
            "total_latency": 0.0,
            "hop_latencies": {},
            "cost_saved_usd": 0.0,
            "co2_avoided_g": 0.0,
        }
    except Exception as exc:
        logger.exception("Inference failed")
        return {
            "job_id": "error",
            "status": "failed",
            "output": f"Error: {exc}",
            "tokens_generated": 0,
            "total_latency": 0.0,
            "hop_latencies": {},
            "cost_saved_usd": 0.0,
            "co2_avoided_g": 0.0,
        }


def submit_async_job(prompt: str, priority: str = "normal") -> dict:
    """Submit an async inference job and return queue info."""
    # TODO: Implement async job submission via priority queue
    return {
        "job_id": "not-implemented",
        "status": "error",
        "queue_position": 0,
        "estimated_wait": 0.0,
    }


def get_job_result(job_id: str) -> dict:
    """Retrieve the result of a previously submitted async job."""
    # TODO: Implement job result retrieval
    return {
        "job_id": job_id,
        "status": "not-implemented",
        "output": "",
        "tokens_generated": 0,
        "total_latency": 0.0,
        "hop_latencies": {},
        "cost_saved_usd": 0.0,
        "co2_avoided_g": 0.0,
    }


def register_node(node_id: str, layers: str, vram_gb: float) -> dict:
    """Register a machine as a worker node."""
    # This is handled by the worker CLI command, not here
    return {
        "success": False,
        "node_id": node_id,
        "layers_assigned": layers,
        "message": "Use 'meshrun worker start' to register a node",
    }


def deregister_node(node_id: str) -> dict:
    """Gracefully deregister a worker node."""
    try:
        stub = _get_coordinator_stub()
        request = coordinator_pb2.DeregisterRequest(node_id=node_id)
        response = stub.Deregister(request)
        return {
            "success": response.acknowledged,
            "message": response.message,
        }
    except Exception as exc:
        logger.exception("Deregister failed")
        return {"success": False, "message": str(exc)}


def get_network_status() -> dict:
    """Fetch full network status including nodes and queue."""
    try:
        stub = _get_coordinator_stub()
        
        # Get all nodes from the registry (we need a new RPC for this)
        # For now, return a placeholder
        # TODO: Add a GetNetworkStatus RPC to the proto
        
        return {
            "active_nodes": 0,
            "total_layers": 0,
            "covered_layers": 0,
            "model": settings.default_model,
            "queue_depth": 0,
            "nodes": [],
            "queue": [],
        }
    except Exception as exc:
        logger.exception("Failed to get network status")
        return {
            "active_nodes": 0,
            "total_layers": 0,
            "covered_layers": 0,
            "model": settings.default_model,
            "queue_depth": 0,
            "nodes": [],
            "queue": [],
        }


def get_credits(node_id: str) -> dict:
    """Fetch credit balance and history for a node."""
    # TODO: Implement credits tracking
    return {
        "node_id": node_id,
        "balance": 0.0,
        "compute_contributed_hours": 0.0,
        "priority_score": 0.0,
        "history": [],
    }


def detect_local_hardware() -> dict:
    """Detect local GPU and RAM for node registration."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            total_memory = props.total_memory // (1024 * 1024)  # Convert to MB
            
            # Get free memory
            free_memory = torch.cuda.memory_reserved(device) // (1024 * 1024)
            
            return {
                "vram_gb": total_memory / 1024,
                "ram_gb": 16.0,  # TODO: Use psutil to get actual RAM
                "gpu_name": props.name,
                "suggested_layers": f"0-{min(10, total_memory // 200)}",  # Rough estimate
                "can_run_model": total_memory >= 6000,  # 6GB minimum
            }
        else:
            return {
                "vram_gb": 0,
                "ram_gb": 16.0,
                "gpu_name": "No CUDA GPU",
                "suggested_layers": "",
                "can_run_model": False,
            }
    except ImportError:
        return {
            "vram_gb": 0,
            "ram_gb": 16.0,
            "gpu_name": "PyTorch not installed",
            "suggested_layers": "",
            "can_run_model": False,
        }
```

---

### CHANGE 10: Update Display Modules to Accept Dynamic Data

**File:** `meshrun/app/display/panels.py`

**Current state:** Functions use hardcoded values.

**Required changes:** Update each function to accept parameters:

```python
def show_submit_result(
    job_id: str, 
    prompt_preview: str,
    tokens: int = 0,
    latency: str = "N/A",
    hops: list[str] | None = None,
    cost_saved: str = "N/A",
    co2_avoided: str = "N/A",
) -> None:
    """Display inference result panel after a successful submission."""
    preview = prompt_preview[:60]
    hops = hops or []
    
    body = (
        f"[bold]Job ID:[/bold]          {job_id}\n"
        f"[bold]Prompt:[/bold]          {preview}\n"
        f"[bold]Tokens:[/bold]          {tokens}\n"
        f"[bold]Total latency:[/bold]   {latency}\n"
        f"[bold]Hop latency:[/bold]     {', '.join(hops)}\n"
        f"[bold]Cost saved:[/bold]      {cost_saved}\n"
        f"[bold]CO2 avoided:[/bold]     {co2_avoided}"
    )
    
    console.print(Panel(body, title="[bold green]Inference Complete[/bold green]", border_style="green"))


def show_credits_panel(
    node_id: str,
    balance: float = 0.0,
    compute: str = "0.0 GPU-hours",
    priority: float = 0.0,
) -> None:
    """Display the credits summary panel for a node."""
    body = (
        f"[bold]Node ID:[/bold]              {node_id}\n"
        f"[bold]Credit balance:[/bold]        {balance}\n"
        f"[bold]Compute contributed:[/bold]   {compute}\n"
        f"[bold]Priority score:[/bold]        {priority} (0.7 × compute + 0.3 × wait_time)"
    )
    
    console.print(Panel(body, title="[bold yellow]Your Credits[/bold yellow]", border_style="yellow"))


def show_network_summary(
    active_nodes: int = 0,
    layers_covered: str = "0 / 0",
    model_loaded: str = "N/A",
    queue_depth: str = "0 jobs",
) -> None:
    """Display the network summary panel."""
    body = (
        f"[bold]Active nodes:[/bold]         {active_nodes}\n"
        f"[bold]Total layers covered:[/bold] {layers_covered}\n"
        f"[bold]Model loaded:[/bold]         {model_loaded}\n"
        f"[bold]Queue depth:[/bold]          {queue_depth}"
    )
    
    console.print(Panel(body, title="[bold blue]Network Summary[/bold blue]", border_style="blue"))
```

**File:** `meshrun/app/display/tables.py`

**Current state:** Functions use hardcoded data.

**Required changes:** Update each function to accept parameters:

```python
from typing import Any


def show_nodes_table(nodes: list[dict[str, Any]] | None = None) -> None:
    """Render and print the MeshRun nodes table."""
    table = Table(title="MeshRun Nodes")

    table.add_column("Node ID", style="cyan")
    table.add_column("Address", style="dim")
    table.add_column("Layers", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Credits Earned", justify="right")
    table.add_column("Latency", justify="right")

    if nodes:
        for node in nodes:
            status = node.get("status", "unknown")
            status_display = {
                "active": "[green]Active[/green]",
                "idle": "[yellow]Idle[/yellow]",
                "unreachable": "[red]Unreachable[/red]",
                "unknown": "[dim]Unknown[/dim]",
            }.get(status, f"[dim]{status}[/dim]")
            
            table.add_row(
                node.get("node_id", "N/A"),
                node.get("address", "N/A"),
                node.get("layers", "N/A"),
                status_display,
                f"{node.get('credits', 0.0):.1f}",
                node.get("latency", "--"),
            )
    else:
        # No nodes registered
        table.add_row("[dim]No nodes registered[/dim]", "", "", "", "", "")

    console.print(table)


def show_status_table(jobs: list[dict[str, Any]] | None = None) -> None:
    """Render and print the queue status table."""
    table = Table(title="Queue Status")

    table.add_column("Position", justify="center")
    table.add_column("Job ID", style="cyan")
    table.add_column("Prompt Preview", style="dim")
    table.add_column("Priority Score", justify="right")
    table.add_column("Wait Time", justify="right")

    if jobs:
        for i, job in enumerate(jobs, 1):
            preview = job.get("prompt", "")[:40]
            table.add_row(
                str(i),
                job.get("job_id", "N/A"),
                f"{preview}...",
                f"{job.get('priority', 0.0):.1f}",
                job.get("wait_time", "0.0s"),
            )
    else:
        table.add_row("[dim]Queue empty[/dim]", "", "", "", "")

    console.print(table)


def show_credits_history(history: list[dict[str, Any]] | None = None) -> None:
    """Render and print the credit history table."""
    table = Table(title="Credit History")

    table.add_column("Time", style="dim")
    table.add_column("Event")
    table.add_column("Credits", justify="right", style="green")

    if history:
        for entry in history:
            credits = entry.get("credits", 0.0)
            credits_display = f"+{credits:.1f}" if credits >= 0 else f"{credits:.1f}"
            table.add_row(
                entry.get("time", "N/A"),
                entry.get("event", "N/A"),
                credits_display,
            )
    else:
        table.add_row("[dim]No credit history[/dim]", "", "")

    console.print(table)
```

---

### CHANGE 11: Dashboard Wiring to Real Coordinator Events

**File:** `meshrun/app/dashboard/events.py`

**Current state:** Generates fake events.

**Required changes:** Connect to the Coordinator's real-time event stream.

```python
"""Dashboard event streaming - connects to real Coordinator events."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

import grpc

from meshrun.coordinator.proto import coordinator_pb2, coordinator_pb2_grpc
from meshrun.app.config import settings

logger = logging.getLogger(__name__)


async def stream_events() -> AsyncGenerator[dict, None]:
    """Stream real-time events from the Coordinator.
    
    Yields event dictionaries with type and payload.
    """
    # Parse coordinator address
    addr = settings.coordinator_url.replace("http://", "").replace("https://", "")
    
    # Create gRPC channel
    channel = grpc.insecure_channel(addr)
    stub = coordinator_pb2_grpc.CoordinatorServiceStub(channel)
    
    try:
        # Subscribe to events (we need to add this RPC to the proto)
        # For now, poll the registry periodically
        while True:
            try:
                # Get network status
                # TODO: Add SubscribeEvents RPC to proto for real streaming
                # For now, emit periodic status updates
                yield {
                    "type": "heartbeat",
                    "payload": {
                        "timestamp": asyncio.get_event_loop().time(),
                    },
                }
                await asyncio.sleep(5.0)
            except Exception as exc:
                logger.warning("Event stream error: %s", exc)
                yield {
                    "type": "error",
                    "payload": {"message": str(exc)},
                }
                await asyncio.sleep(1.0)
    finally:
        channel.close()


async def get_live_nodes() -> list[dict]:
    """Fetch current node list from the Coordinator."""
    addr = settings.coordinator_url.replace("http://", "").replace("https://", "")
    channel = grpc.insecure_channel(addr)
    stub = coordinator_pb2_grpc.CoordinatorServiceStub(channel)
    
    try:
        # TODO: Add GetNetworkStatus RPC
        # For now, return empty list
        return []
    finally:
        channel.close()
```

**File:** `meshrun/app/dashboard/server.py`

**Current state:** Uses mock event generator.

**Required changes:** Import and use the real event stream:

```python
from meshrun.app.dashboard.events import stream_events, get_live_nodes

# Replace mock event generator with real one
# In the WebSocket handler:
async for event in stream_events():
    await websocket.send_json(event)
```

---

### CHANGE 12: Kiro Extension Sidebar Wiring

**File:** `meshrun/app/dashboard/kiro_sidebar.py` (new file)

**Current state:** The Kiro extension uses mock data.

**Required changes:** Create a module that the Kiro extension can import to get real data:

```python
"""Kiro extension sidebar integration.

This module provides the API that the Kiro IDE extension calls to display
real-time MeshRun status in the sidebar.
"""

from __future__ import annotations

import logging
from typing import Any

import grpc

from meshrun.coordinator.proto import coordinator_pb2, coordinator_pb2_grpc
from meshrun.app.config import settings

logger = logging.getLogger(__name__)


def get_sidebar_status() -> dict[str, Any]:
    """Get the current MeshRun status for the Kiro sidebar.
    
    Returns a dict with:
    - cluster_health: "healthy" | "degraded" | "down"
    - active_nodes: int
    - total_layers: int
    - covered_layers: int
    - queue_depth: int
    - model: str
    - nodes: list of node info dicts
    """
    addr = settings.coordinator_url.replace("http://", "").replace("https://", "")
    
    try:
        channel = grpc.insecure_channel(addr)
        stub = coordinator_pb2_grpc.CoordinatorServiceStub(channel)
        
        # TODO: Add GetNetworkStatus RPC to proto
        # For now, return a placeholder
        
        channel.close()
        
        return {
            "cluster_health": "unknown",
            "active_nodes": 0,
            "total_layers": 0,
            "covered_layers": 0,
            "queue_depth": 0,
            "model": settings.default_model,
            "nodes": [],
        }
    except Exception as exc:
        logger.warning("Failed to get sidebar status: %s", exc)
        return {
            "cluster_health": "down",
            "active_nodes": 0,
            "total_layers": 0,
            "covered_layers": 0,
            "queue_depth": 0,
            "model": settings.default_model,
            "nodes": [],
            "error": str(exc),
        }


def get_node_details(node_id: str) -> dict[str, Any]:
    """Get detailed info for a specific node."""
    # TODO: Implement after adding GetNodeDetails RPC
    return {
        "node_id": node_id,
        "status": "unknown",
        "layers": "N/A",
        "gpu_utilization": 0.0,
        "memory_used_mb": 0,
        "requests_served": 0,
    }
```

---

### CHANGE 13: Add GetNetworkStatus RPC to Proto

**File:** `meshrun/coordinator/proto/coordinator.proto`

**Current state:** No RPC to get the full network status.

**Required changes:** Add a new RPC and messages:

```protobuf
// Add to service CoordinatorService
rpc GetNetworkStatus(GetNetworkStatusRequest) returns (GetNetworkStatusResponse);

// Request message
message GetNetworkStatusRequest {
    // Empty for now, can add filters later
}

// Response message
message GetNetworkStatusResponse {
    int32 active_nodes = 1;
    int32 total_layers = 2;
    int32 covered_layers = 3;
    string model_id = 4;
    int32 queue_depth = 5;
    repeated NodeInfo nodes = 6;
    repeated QueueEntry queue = 7;
}

message NodeInfo {
    string node_id = 1;
    string address = 2;
    string grpc_address = 3;
    int32 layer_start = 4;
    int32 layer_end = 5;
    string status = 6;  // "active", "idle", "unhealthy"
    float gpu_utilization = 7;
    int64 memory_used_mb = 8;
    int64 memory_total_mb = 9;
    int32 requests_served = 10;
    float credits_earned = 11;
    int64 last_heartbeat_ms = 12;
}

message QueueEntry {
    string job_id = 1;
    string prompt_preview = 2;
    float priority_score = 3;
    int64 submit_time_ms = 4;
    int32 position = 5;
}
```

**Regenerate the Python stubs:**

```bash
cd meshrun/coordinator/proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. coordinator.proto
```

**File:** `meshrun/coordinator/server.py`

**Add the handler:**

```python
def GetNetworkStatus(
    self, 
    request: pb2.GetNetworkStatusRequest, 
    context: grpc.ServicerContext
) -> pb2.GetNetworkStatusResponse:
    """Return the current network status for CLI and dashboard."""
    import time
    
    nodes = self._registry.get_all_nodes()
    
    # Build NodeInfo list
    node_infos = []
    covered_layers = set()
    for node in nodes:
        # Determine status
        health = self._health_tracker.get_health(node.node_id)
        if health and health.is_healthy:
            status = "active"
        elif health and health.is_alive:
            status = "idle"
        else:
            status = "unhealthy"
        
        node_info = pb2.NodeInfo(
            node_id=node.node_id,
            address=node.address,
            grpc_address=node.grpc_address,
            layer_start=node.layer_start or 0,
            layer_end=node.layer_end or 0,
            status=status,
            gpu_utilization=node.gpu_utilization,
            memory_used_mb=node.memory_used_mb or 0,
            memory_total_mb=node.gpu_memory_total_mb,
            requests_served=0,  # TODO: Track this
            credits_earned=0.0,  # TODO: Track this
            last_heartbeat_ms=int(node.last_heartbeat * 1000) if node.last_heartbeat else 0,
        )
        node_infos.append(node_info)
        
        # Track covered layers
        if node.layer_start is not None and node.layer_end is not None:
            covered_layers.update(range(node.layer_start, node.layer_end + 1))
    
    # Get queue depth
    queue_depth = self._priority_queue.size()
    
    # Get model info from layer map
    model_id = ""
    total_layers = 0
    entries = self._layer_map.get_entries()
    if entries:
        model_id = entries[0].model_id
        # Find max layer_end across all entries
        total_layers = max(e.layer_end for e in entries) + 1
    
    return pb2.GetNetworkStatusResponse(
        active_nodes=len([n for n in node_infos if n.status in ("active", "idle")]),
        total_layers=total_layers,
        covered_layers=len(covered_layers),
        model_id=model_id,
        queue_depth=queue_depth,
        nodes=node_infos,
        queue=[],  # TODO: Add queue entries
    )
```

---

## Validation & Testing

### Unit Test Updates

After implementing all changes, update the following test files:

1. `meshrun/coordinator/test_server.py` — Add tests for:
   - Auto-assignment on registration
   - `GetNetworkStatus` RPC
   - `_push_assignments_to_workers`

2. `meshrun/worker/test_assignment_server.py` (new) — Test:
   - `AcceptLayerAssignment` handler
   - Integration with `WorkerNode.accept_layer_assignment`

3. `meshrun/app/client/test_inference.py` (new) — Test:
   - Real inference client wiring
   - Error handling

### Integration Test Script

Create `scripts/test_distributed.py`:

```python
#!/usr/bin/env python
"""Integration test for distributed inference over Tailscale."""

import subprocess
import sys
import time

def main():
    print("=== MeshRun Distributed Integration Test ===")
    
    # 1. Start coordinator
    print("\n[1/5] Starting Coordinator...")
    coord_proc = subprocess.Popen(
        ["meshrun", "coordinator", "--model", "llama-3b", "--layers", "28"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(2.0)  # Wait for coordinator to start
    
    # 2. Start worker 1
    print("\n[2/5] Starting Worker 1...")
    worker1_proc = subprocess.Popen(
        ["meshrun", "worker", "--coordinator", "localhost:50051"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(5.0)  # Wait for registration and assignment
    
    # 3. Start worker 2
    print("\n[3/5] Starting Worker 2...")
    worker2_proc = subprocess.Popen(
        ["meshrun", "worker", "--coordinator", "localhost:50051", "--data-port", "9101", "--grpc-port", "50053"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(5.0)
    
    # 4. Check status
    print("\n[4/5] Checking network status...")
    result = subprocess.run(
        ["meshrun", "status"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    
    # 5. Submit inference
    print("\n[5/5] Submitting inference...")
    result = subprocess.run(
        ["meshrun", "submit", "Hello, world!"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    
    # Cleanup
    print("\n[Cleanup] Stopping processes...")
    coord_proc.terminate()
    worker1_proc.terminate()
    worker2_proc.terminate()
    
    print("\n=== Test Complete ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

---

## Demo Deployment on Tailscale

### Prerequisites

1. Install Tailscale on all machines: https://tailscale.com/download
2. Authenticate with the same Tailscale account/organization
3. Verify connectivity: `tailscale ping <other-machine-ip>`

### Machine Setup

**Machine 1 (Coordinator):**
```bash
# Install MeshRun
git clone <repo>
cd meshrun
uv venv .venv
.venv\Scripts\activate  # Windows
uv pip install -e .

# Get Tailscale IP
tailscale ip --4  # e.g., 100.64.0.1

# Start coordinator
meshrun coordinator --host 0.0.0.0 --model llama-3b --layers 28 --model-url https://huggingface.co/...
```

**Machine 2 (Worker 1):**
```bash
# Install MeshRun (same as above)

# Get Tailscale IP
tailscale ip --4  # e.g., 100.64.0.2

# Start worker (point to coordinator's Tailscale IP)
meshrun worker --coordinator 100.64.0.1:50051 --advertise 100.64.0.2:9100
```

**Machine 3 (Worker 2):**
```bash
# Same as Worker 1, but with different ports
meshrun worker --coordinator 100.64.0.1:50051 --advertise 100.64.0.3:9100 --data-port 9100 --grpc-port 50052
```

**Machine 4 (Client):**
```bash
# Submit inference
meshrun submit "What is the capital of France?" --coordinator 100.64.0.1:50051
```

### Network Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Tailscale Network                        │
│                        (100.64.0.0/10)                          │
│                                                                  │
│  ┌──────────────┐                                               │
│  │ Coordinator  │                                               │
│  │ 100.64.0.1   │◄─────────────── gRPC (50051) ────────────────┤
│  │              │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         │ gRPC push (AcceptLayerAssignment)                      │
│         ▼                                                        │
│  ┌──────────────┐     ┌──────────────┐                          │
│  │  Worker 1    │────►│  Worker 2    │                          │
│  │ 100.64.0.2   │ TCP │ 100.64.0.3   │                          │
│  │ Layers 0-13  │────►│ Layers 14-27 │                          │
│  └──────────────┘     └──────────────┘                          │
│         ▲                                                        │
│         │ TCP (encrypted FORWARD)                                │
│         │                                                        │
│  ┌──────┴───────┐                                               │
│  │    Client    │                                               │
│  │ 100.64.0.4   │                                               │
│  └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Verification Checklist

- [ ] Coordinator starts and listens on 0.0.0.0:50051
- [ ] Worker 1 registers with coordinator (check logs)
- [ ] Worker 2 registers with coordinator (check logs)
- [ ] Auto-assignment triggers after each registration
- [ ] Workers receive AcceptLayerAssignment push
- [ ] Workers load shards and send ConfirmReady
- [ ] `meshrun status` shows all nodes as active
- [ ] `meshrun submit` returns generated text
- [ ] Dashboard shows real node data
- [ ] Kiro sidebar shows cluster health

---

## Summary of Changes

| Change | File(s)                                       | Description                                             |
| ------ | --------------------------------------------- | ------------------------------------------------------- |
| 1      | `pyproject.toml`                              | Unified package config with CLI entry points            |
| 2      | `meshrun/app/**/*.py`                         | Fix import paths to use `meshrun.app.*`                 |
| 3      | `meshrun/app/cli/commands/coordinator.py`     | CLI command to start coordinator                        |
| 4      | `meshrun/app/cli/commands/worker.py`          | CLI command to start worker with Tailscale IP detection |
| 5      | `meshrun/coordinator/server.py`               | Add model config storage                                |
| 6      | `meshrun/coordinator/server.py`               | Auto-assignment on worker registration                  |
| 7      | `meshrun/worker/assignment_server.py`         | Worker-side gRPC server for AcceptLayerAssignment       |
| 8      | `meshrun/coordinator/server.py`               | Push assignments to workers via gRPC                    |
| 9      | `meshrun/app/client/inference.py`             | Wire CLI to real InferenceClient                        |
| 10     | `meshrun/app/display/*.py`                    | Accept dynamic data in display functions                |
| 11     | `meshrun/app/dashboard/events.py`             | Connect dashboard to real events                        |
| 12     | `meshrun/app/dashboard/kiro_sidebar.py`       | Kiro extension API                                      |
| 13     | `meshrun/coordinator/proto/coordinator.proto` | Add GetNetworkStatus RPC                                |

---

## Implementation Order

1. **Phase 1: Core Infrastructure** (Changes 1, 2, 5, 13)
   - Unified pyproject.toml
   - Fix imports
   - Add model config storage
   - Add GetNetworkStatus RPC

2. **Phase 2: CLI Entry Points** (Changes 3, 4)
   - Coordinator CLI command
   - Worker CLI command

3. **Phase 3: Auto-Assignment** (Changes 6, 7, 8)
   - Auto-assignment on registration
   - Worker gRPC server
   - Push assignments to workers

4. **Phase 4: Client & Display** (Changes 9, 10, 11, 12)
   - Wire CLI inference
   - Update display modules
   - Dashboard wiring
   - Kiro sidebar

5. **Phase 5: Testing & Validation**
   - Unit tests
   - Integration tests
   - Tailscale demo

---

## Risk Mitigation

| Risk                             | Mitigation                                         |
| -------------------------------- | -------------------------------------------------- |
| gRPC connection failures         | Add retry logic with exponential backoff           |
| Worker crashes during shard load | Graceful error handling, log to coordinator        |
| Network partition                | Heartbeat timeout detection, re-registration       |
| Model URL unreachable            | Validate URL at coordinator startup, cache locally |
| GPU OOM during load              | Check available memory before load, fail fast      |

---

## Next Steps

1. Review and approve this plan
2. Implement Phase 1 (Core Infrastructure)
3. Test coordinator startup
4. Implement Phase 2 (CLI Entry Points)
5. Test worker registration
6. Continue through remaining phases
7. Run integration tests
8. Deploy on Tailscale for demo

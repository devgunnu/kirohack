"""
Worker Node — lifecycle management for a data plane worker.

Orchestrates the full worker node lifecycle: startup, registration with
the Coordinator, layer assignment acceptance, serving, and graceful
shutdown.  Wires together all sub-components (Resource Monitor,
Connection Pool, Shard Manager, Layer Engine, Layer Assignment Registry).

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from enum import IntEnum, auto
from pathlib import Path
from typing import Optional

from meshrun.worker.connection_pool import ConnectionPool
from meshrun.worker.coordinator_client import (
    CapacityInfo,
    ConfirmReadyRequest,
    ConfirmReadyResponse,
    CoordinatorClient,
    GrpcCoordinatorClient,
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterRequest,
    RegisterResponse,
    RegistrationStatus,
)
from meshrun.worker.layer_registry import (
    AssignmentDType,
    LayerAssignment,
    LayerAssignmentRegistry,
)
from meshrun.worker.resource_monitor import GpuMetrics, ResourceMonitor
from meshrun.worker.layer_engine import LayerEngine, build_layer_engine, warm_up
from meshrun.worker.serving import ServingConfig, ServingLoop
from meshrun.worker.shard_manager import (
    LayerRange,
    LoadStatus,
    ShardDType,
    ShardMetadata,
    load_shard,
)

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_GPU_MEMORY_LIMIT_MB: int = 4096
"""Default GPU memory limit when none is specified (4 GB)."""

_DEFAULT_DATA_PORT: int = 9100
"""Default TCP port for the data plane listener."""

_DEFAULT_GRPC_PORT: int = 50051
"""Default gRPC port for control plane communication."""

_FRAMEWORK_OVERHEAD_MB: int = 800
"""Estimated PyTorch/CUDA framework overhead in MB."""

_DEFAULT_HEARTBEAT_INTERVAL_S: float = 5.0
"""Default interval between heartbeat RPCs to the Coordinator."""


# ── Enumerations ─────────────────────────────────────────────────────────────


class NodeState(IntEnum):
    """Worker node lifecycle states."""

    INITIALIZING = auto()
    REGISTERING = auto()
    WAITING_ASSIGNMENT = auto()
    LOADING_SHARD = auto()
    VALIDATING = auto()
    READY = auto()
    SERVING = auto()
    DRAINING = auto()
    ERROR = auto()


# Valid state transitions.  Each key maps to the set of states it may
# transition *to*.  Any transition not listed here is illegal.
_VALID_TRANSITIONS: dict[NodeState, frozenset[NodeState]] = {
    NodeState.INITIALIZING: frozenset({NodeState.REGISTERING}),
    NodeState.REGISTERING: frozenset({NodeState.WAITING_ASSIGNMENT, NodeState.ERROR}),
    NodeState.WAITING_ASSIGNMENT: frozenset({NodeState.LOADING_SHARD, NodeState.ERROR}),
    NodeState.LOADING_SHARD: frozenset({NodeState.VALIDATING, NodeState.ERROR}),
    NodeState.VALIDATING: frozenset({NodeState.READY, NodeState.ERROR}),
    NodeState.READY: frozenset({NodeState.SERVING, NodeState.ERROR}),
    NodeState.SERVING: frozenset({NodeState.SERVING, NodeState.DRAINING, NodeState.ERROR}),
    NodeState.DRAINING: frozenset({NodeState.INITIALIZING}),
    NodeState.ERROR: frozenset({NodeState.REGISTERING}),
}


class InvalidStateTransition(RuntimeError):
    """Raised when a state transition violates the lifecycle graph."""

    def __init__(self, current: NodeState, target: NodeState) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"Invalid state transition: {current.name} → {target.name}"
        )


# ── Heartbeat Sender ────────────────────────────────────────────────────────


class HeartbeatSender:
    """Periodically sends heartbeat RPCs to the Coordinator.

    Runs a daemon thread that polls the Resource Monitor for a snapshot
    and sends it to the Coordinator at a configurable interval.  The
    thread exits when ``stop()`` is called or the node shuts down.

    Parameters
    ----------
    node_id:
        The worker node's unique identifier.
    coordinator_client:
        The gRPC client used to send heartbeat RPCs.
    resource_monitor:
        The Resource Monitor providing heartbeat snapshots.
    interval_s:
        Seconds between heartbeat RPCs.  Defaults to 5.0.
    """

    def __init__(
        self,
        *,
        node_id: str,
        coordinator_client: CoordinatorClient,
        resource_monitor: ResourceMonitor,
        interval_s: float = _DEFAULT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        self._node_id = node_id
        self._client = coordinator_client
        self._monitor = resource_monitor
        self._interval_s = interval_s

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consecutive_failures: int = 0

    @property
    def is_running(self) -> bool:
        """Whether the heartbeat thread is currently active."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive heartbeat send failures."""
        return self._consecutive_failures

    def start(self) -> None:
        """Start the periodic heartbeat sender thread.

        Does nothing if already running.
        """
        if self.is_running:
            logger.debug("HeartbeatSender already running — skipping start")
            return

        self._stop_event.clear()
        self._consecutive_failures = 0
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="heartbeat-sender",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "HeartbeatSender started for node %s (interval=%.1fs)",
            self._node_id,
            self._interval_s,
        )

    def stop(self) -> None:
        """Stop the heartbeat sender thread.

        Blocks until the thread exits (up to one interval + 1s).
        """
        if not self.is_running:
            return

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 1.0)
            self._thread = None
        logger.info("HeartbeatSender stopped for node %s", self._node_id)

    def send_once(self) -> HeartbeatResponse:
        """Send a single heartbeat synchronously.

        Useful for testing or for sending an immediate heartbeat outside
        the periodic loop.

        Returns
        -------
        HeartbeatResponse
            The Coordinator's acknowledgement.
        """
        snapshot = self._monitor.get_heartbeat_snapshot()
        request = HeartbeatRequest(
            node_id=self._node_id,
            gpu_utilization=snapshot.gpu_utilization,
            memory_used_mb=snapshot.memory_used_mb,
            active_requests=snapshot.active_requests,
        )
        return self._client.heartbeat(request)

    def _heartbeat_loop(self) -> None:
        """Background loop that sends heartbeats at the configured interval."""
        while not self._stop_event.is_set():
            try:
                self.send_once()
                self._consecutive_failures = 0
            except Exception:
                self._consecutive_failures += 1
                logger.warning(
                    "Heartbeat send failed for node %s "
                    "(consecutive failures: %d)",
                    self._node_id,
                    self._consecutive_failures,
                    exc_info=True,
                )

            self._stop_event.wait(timeout=self._interval_s)


# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class NodeCapacity:
    """Capacity descriptor sent to the Coordinator during registration.

    Captures the GPU resources available on this node, bounded by the
    user-configured memory limit.
    """

    gpu_memory_total_mb: int
    """Total physical GPU memory in MB."""

    gpu_memory_free_mb: int
    """Free GPU memory at startup in MB."""

    memory_limit_mb: int
    """User-configured maximum GPU memory this node may use."""

    gpu_utilization: float
    """Current GPU compute utilization (0.0–1.0)."""


@dataclass(frozen=True, slots=True)
class NodeConfig:
    """User-provided configuration for a worker node.

    Parameters
    ----------
    host:
        Bind address for the TCP data plane listener.
    data_port:
        TCP port for the data plane listener.
    grpc_port:
        Port for gRPC control plane communication.
    coordinator_address:
        ``host:port`` of the Coordinator's gRPC endpoint.
    gpu_memory_limit_mb:
        Maximum GPU memory this node is allowed to use (MB).
    device_index:
        CUDA device ordinal to use.  Defaults to 0.
    poll_interval_s:
        Resource Monitor polling interval in seconds.
    """

    host: str = "0.0.0.0"
    data_port: int = _DEFAULT_DATA_PORT
    grpc_port: int = _DEFAULT_GRPC_PORT
    coordinator_address: str = "localhost:50051"
    gpu_memory_limit_mb: int = _DEFAULT_GPU_MEMORY_LIMIT_MB
    device_index: int = 0
    poll_interval_s: float = 1.0
    heartbeat_interval_s: float = _DEFAULT_HEARTBEAT_INTERVAL_S


# ── Worker Node ─────────────────────────────────────────────────────────────


class WorkerNode:
    """Orchestrates the full worker node lifecycle.

    The node progresses through states:
    ``Initializing → Registering → WaitingAssignment → LoadingShard
    → Validating → Ready → Serving → Draining``

    Task 8.1 covers the startup phase: initialise the Resource Monitor,
    query local GPU resources, and generate a unique ``node_id``.
    """

    def __init__(
        self,
        config: Optional[NodeConfig] = None,
        *,
        resource_monitor: Optional[ResourceMonitor] = None,
        connection_pool: Optional[ConnectionPool] = None,
        layer_registry: Optional[LayerAssignmentRegistry] = None,
        coordinator_client: Optional[CoordinatorClient] = None,
    ) -> None:
        self._config = config or NodeConfig()
        self._state = NodeState.INITIALIZING

        # Generate a unique node identifier.
        self._node_id: str = uuid.uuid4().hex

        # Sub-components — injected or created during startup.
        self._resource_monitor = resource_monitor
        self._connection_pool = connection_pool or ConnectionPool()
        self._layer_registry = layer_registry or LayerAssignmentRegistry()
        self._coordinator_client = coordinator_client

        # Populated after GPU query during startup.
        self._capacity: Optional[NodeCapacity] = None

        # Populated after layer assignment acceptance.
        self._shard_metadata: Optional[ShardMetadata] = None

        # Populated after shard load, built from loaded tensors.
        self._layer_engine: Optional[LayerEngine] = None

        # Populated when serving starts.
        self._serving_loop: Optional[ServingLoop] = None

        # Populated when heartbeat sending starts.
        self._heartbeat_sender: Optional[HeartbeatSender] = None

        # Populated after layer assignment with the AES-256 session key
        # for encrypted data-plane traffic.
        self._session_key: Optional[bytes] = None

        logger.info(
            "WorkerNode created: node_id=%s, state=%s",
            self._node_id,
            self._state.name,
        )

    # ── State Transition (Task 8.5) ──────────────────────────────────────

    def _transition_to(self, target: NodeState) -> None:
        """Validate and execute a state transition.

        Parameters
        ----------
        target:
            The desired next state.

        Raises
        ------
        InvalidStateTransition
            If the transition from the current state to *target* is not
            permitted by the lifecycle graph.
        """
        allowed = _VALID_TRANSITIONS.get(self._state, frozenset())
        if target not in allowed:
            raise InvalidStateTransition(self._state, target)
        previous = self._state
        self._state = target
        logger.info(
            "Node %s state transition: %s → %s",
            self._node_id,
            previous.name,
            target.name,
        )

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def node_id(self) -> str:
        """This node's unique identifier."""
        return self._node_id

    @property
    def state(self) -> NodeState:
        """Current lifecycle state."""
        return self._state

    @property
    def config(self) -> NodeConfig:
        """Node configuration."""
        return self._config

    @property
    def capacity(self) -> Optional[NodeCapacity]:
        """GPU capacity snapshot taken at startup.  ``None`` before startup."""
        return self._capacity

    # ── Registration (Task 8.2) ──────────────────────────────────────────

    def register_with_coordinator(self) -> RegisterResponse:
        """Register this node with the Coordinator via gRPC.

        Sends a ``Register`` RPC containing the node's identity, network
        addresses, and GPU capacity (including the user-configured memory
        limit).  On success the node transitions to ``WAITING_ASSIGNMENT``.

        The Coordinator client is created lazily on first call if one was
        not injected via the constructor.

        Returns
        -------
        RegisterResponse
            The Coordinator's acknowledgement.

        Raises
        ------
        RuntimeError
            If the node is not in ``REGISTERING`` state, or if
            ``startup()`` has not been called (no capacity available).
        ConnectionError
            If the registration RPC fails due to a transport error.
        """
        if self._state != NodeState.REGISTERING:
            raise RuntimeError(
                f"register_with_coordinator() requires REGISTERING state, "
                f"current state is {self._state.name}"
            )
        if self._capacity is None:
            raise RuntimeError(
                "register_with_coordinator() requires startup() to have "
                "been called first (no capacity available)"
            )

        # Lazily create the gRPC client if not injected.
        if self._coordinator_client is None:
            self._coordinator_client = GrpcCoordinatorClient(
                self._config.coordinator_address
            )

        # Build the registration request.
        request = RegisterRequest(
            node_id=self._node_id,
            address=self.address,
            grpc_address=self.grpc_address,
            capacity=CapacityInfo(
                gpu_memory_total_mb=self._capacity.gpu_memory_total_mb,
                gpu_memory_free_mb=self._capacity.gpu_memory_free_mb,
                memory_limit_mb=self._capacity.memory_limit_mb,
                gpu_utilization=self._capacity.gpu_utilization,
            ),
            layers_hosted=None,  # No layers loaded yet at registration time.
        )

        # Send the Register RPC.
        try:
            response = self._coordinator_client.register(request)
        except Exception as exc:
            self._transition_to(NodeState.ERROR)
            logger.error(
                "Registration failed for node %s: %s", self._node_id, exc
            )
            raise ConnectionError(
                f"Failed to register with Coordinator at "
                f"{self._config.coordinator_address}: {exc}"
            ) from exc

        # Handle the response.
        if response.status == RegistrationStatus.OK:
            self._transition_to(NodeState.WAITING_ASSIGNMENT)
            logger.info(
                "Node %s registered successfully → %s  (message: %s)",
                self._node_id,
                self._state.name,
                response.message,
            )
        else:
            self._transition_to(NodeState.ERROR)
            logger.error(
                "Registration rejected for node %s: status=%s, message=%s",
                self._node_id,
                response.status.name,
                response.message,
            )

        return response

    # ── Layer Assignment (Task 8.3) ──────────────────────────────────────

    def accept_layer_assignment(
        self,
        *,
        model_id: str,
        model_url: str,
        layer_start: int,
        layer_end: int,
        dtype: AssignmentDType,
        is_final_node: bool,
        downstream_node: Optional[str] = None,
        upstream_nodes: tuple[str, ...] = (),
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        session_key: Optional[bytes] = None,
    ) -> ShardMetadata:
        """Accept a layer assignment from the Coordinator and begin shard loading.

        This handler:

        1. Validates the node is in ``WAITING_ASSIGNMENT`` state.
        2. Stores the assignment in the Layer Assignment Registry.
        3. Transitions the node to ``LOADING_SHARD``.
        4. Triggers the Shard Manager's ``load_shard`` to selectively
           download the assigned layer weights via HTTP Range requests.

        Parameters
        ----------
        model_id:
            Identifier of the model being served (e.g. ``'llama-3b'``).
        model_url:
            HTTP URL to the safetensors model file.
        layer_start:
            First assigned layer index (inclusive).
        layer_end:
            Last assigned layer index (inclusive).
        dtype:
            Quantization format for this shard (fp16 or int8).
        is_final_node:
            Whether this node hosts the last layers and produces logits.
        downstream_node:
            TCP ``host:port`` of the next pipeline node.  ``None`` if final.
        upstream_nodes:
            TCP addresses that may send data to this node.
        device:
            Target torch device (e.g. ``'cuda:0'``).  Defaults to
            ``'cuda:{config.device_index}'``.
        cache_dir:
            Local directory for cached weights.  Defaults to
            ``'.cache/meshrun'``.
        session_key:
            AES-256 session key (32 bytes) for encrypted data-plane
            traffic.  Distributed by the Coordinator during assignment.

        Returns
        -------
        ShardMetadata
            The shard metadata after the load attempt, reflecting the
            final status (``READY`` or ``ERROR``).

        Raises
        ------
        RuntimeError
            If the node is not in ``WAITING_ASSIGNMENT`` state.
        """
        if self._state != NodeState.WAITING_ASSIGNMENT:
            raise RuntimeError(
                f"accept_layer_assignment() requires WAITING_ASSIGNMENT state, "
                f"current state is {self._state.name}"
            )

        # ── 1. Store assignment in the registry ─────────────────────────
        self._session_key = session_key

        assignment = LayerAssignment(
            node_id=self._node_id,
            model_id=model_id,
            model_url=model_url,
            layer_start=layer_start,
            layer_end=layer_end,
            dtype=dtype,
            is_final_node=is_final_node,
            downstream_node=downstream_node,
            upstream_nodes=upstream_nodes,
        )
        self._layer_registry.accept_layer_assignment(assignment)

        logger.info(
            "Node %s accepted layer assignment: model=%s, layers %d–%d, "
            "dtype=%s, is_final=%s, downstream=%s",
            self._node_id,
            model_id,
            layer_start,
            layer_end,
            dtype.name,
            is_final_node,
            downstream_node,
        )

        # ── 2. Transition to LOADING_SHARD ──────────────────────────────
        self._transition_to(NodeState.LOADING_SHARD)

        # ── 3. Build shard metadata and trigger load ────────────────────
        resolved_device = device or f"cuda:{self._config.device_index}"
        resolved_cache = Path(cache_dir) if cache_dir else Path(".cache/meshrun")

        # Map AssignmentDType → ShardDType (values are identical: FP16=1, INT8=2)
        shard_dtype = ShardDType(dtype.value)

        metadata = ShardMetadata(
            model_id=model_id,
            model_url=model_url,
            layer_range=LayerRange(start=layer_start, end=layer_end),
            dtype=shard_dtype,
            cache_dir=resolved_cache,
        )

        # Determine pipeline position for embedding / LM head inclusion.
        is_first_node = layer_start == 0

        metadata = load_shard(
            metadata,
            is_first_node=is_first_node,
            is_final_node=is_final_node,
            device=resolved_device,
        )

        # ── 4. Update node state based on load result ───────────────────
        self._shard_metadata = metadata

        if metadata.load_status == LoadStatus.READY:
            self._transition_to(NodeState.VALIDATING)
            logger.info(
                "Node %s shard loaded successfully (%.1f MB) → %s",
                self._node_id,
                metadata.memory_footprint_mb,
                self._state.name,
            )
        else:
            self._transition_to(NodeState.ERROR)
            logger.error(
                "Node %s shard load failed: %s",
                self._node_id,
                metadata.error_message,
            )

        return metadata

    # ── Confirm Ready (Task 8.4) ─────────────────────────────────────────

    def confirm_ready(self) -> ConfirmReadyResponse:
        """Send a ``ConfirmReady`` RPC to the Coordinator after shard load + validation.

        This method is called after ``accept_layer_assignment`` has
        successfully loaded and validated the shard.  It:

        1. Validates the node is in ``VALIDATING`` state.
        2. Reads the loaded layer range from the Layer Assignment Registry.
        3. Sends a ``ConfirmReady`` RPC to the Coordinator.
        4. Transitions the node to ``READY`` on success, or ``ERROR`` on failure.

        Returns
        -------
        ConfirmReadyResponse
            The Coordinator's acknowledgement.

        Raises
        ------
        RuntimeError
            If the node is not in ``VALIDATING`` state, or if no layer
            assignment is stored, or if no Coordinator client is available.
        ConnectionError
            If the ConfirmReady RPC fails due to a transport error.
        """
        if self._state != NodeState.VALIDATING:
            raise RuntimeError(
                f"confirm_ready() requires VALIDATING state, "
                f"current state is {self._state.name}"
            )

        # Read the layer range from the registry.
        layer_range = self._layer_registry.get_layer_range()
        if layer_range is None:
            raise RuntimeError(
                "confirm_ready() requires a layer assignment in the registry"
            )

        if self._coordinator_client is None:
            raise RuntimeError(
                "confirm_ready() requires a Coordinator client — "
                "call register_with_coordinator() first"
            )

        request = ConfirmReadyRequest(
            node_id=self._node_id,
            layers_loaded=layer_range,
        )

        try:
            response = self._coordinator_client.confirm_ready(request)
        except Exception as exc:
            self._transition_to(NodeState.ERROR)
            logger.error(
                "ConfirmReady failed for node %s: %s", self._node_id, exc
            )
            raise ConnectionError(
                f"Failed to send ConfirmReady to Coordinator: {exc}"
            ) from exc

        if response.acknowledged:
            self._transition_to(NodeState.READY)
            logger.info(
                "Node %s confirmed ready → %s  (layers %d–%d, message: %s)",
                self._node_id,
                self._state.name,
                layer_range[0],
                layer_range[1],
                response.message,
            )
        else:
            self._transition_to(NodeState.ERROR)
            logger.error(
                "ConfirmReady rejected for node %s: %s",
                self._node_id,
                response.message,
            )

        return response

    @property
    def shard_metadata(self) -> Optional[ShardMetadata]:
        """The current shard metadata, or ``None`` if no shard has been loaded."""
        return self._shard_metadata

    @property
    def resource_monitor(self) -> Optional[ResourceMonitor]:
        """The Resource Monitor instance, or ``None`` before startup."""
        return self._resource_monitor

    @property
    def connection_pool(self) -> ConnectionPool:
        """The Connection Pool instance."""
        return self._connection_pool

    @property
    def layer_registry(self) -> LayerAssignmentRegistry:
        """The Layer Assignment Registry instance."""
        return self._layer_registry

    @property
    def address(self) -> str:
        """TCP data plane address as ``host:port``."""
        return f"{self._config.host}:{self._config.data_port}"

    @property
    def grpc_address(self) -> str:
        """gRPC control plane address as ``host:port``."""
        return f"{self._config.host}:{self._config.grpc_port}"

    @property
    def coordinator_client(self) -> Optional[CoordinatorClient]:
        """The Coordinator client, or ``None`` if not yet created."""
        return self._coordinator_client

    @property
    def serving_loop(self) -> Optional[ServingLoop]:
        """The serving loop, or ``None`` if not yet started."""
        return self._serving_loop

    @property
    def heartbeat_sender(self) -> Optional[HeartbeatSender]:
        """The heartbeat sender, or ``None`` if not yet started."""
        return self._heartbeat_sender

    @property
    def layer_engine(self) -> Optional[LayerEngine]:
        """The Layer Engine, or ``None`` if not yet built."""
        return self._layer_engine

    # ── Heartbeat (Task 9.3) ─────────────────────────────────────────────

    def start_heartbeat(self) -> HeartbeatSender:
        """Start periodic heartbeat sending to the Coordinator.

        Creates a :class:`HeartbeatSender` that polls the Resource Monitor
        for a snapshot and sends it to the Coordinator at the configured
        interval.  The sender runs on a daemon thread and stops
        automatically when the node shuts down.

        Returns
        -------
        HeartbeatSender
            The running heartbeat sender instance.

        Raises
        ------
        RuntimeError
            If the node is not in ``SERVING`` or ``READY`` state, or if
            the Resource Monitor or Coordinator client is not available.
        """
        if self._state not in (NodeState.SERVING, NodeState.READY):
            raise RuntimeError(
                f"start_heartbeat() requires SERVING or READY state, "
                f"current state is {self._state.name}"
            )
        if self._resource_monitor is None:
            raise RuntimeError(
                "start_heartbeat() requires a Resource Monitor — "
                "call startup() first"
            )
        if self._coordinator_client is None:
            raise RuntimeError(
                "start_heartbeat() requires a Coordinator client — "
                "call register_with_coordinator() first"
            )

        self._heartbeat_sender = HeartbeatSender(
            node_id=self._node_id,
            coordinator_client=self._coordinator_client,
            resource_monitor=self._resource_monitor,
            interval_s=self._config.heartbeat_interval_s,
        )
        self._heartbeat_sender.start()

        logger.info(
            "Node %s heartbeat sender started (interval=%.1fs)",
            self._node_id,
            self._config.heartbeat_interval_s,
        )

        return self._heartbeat_sender

    def stop_heartbeat(self) -> None:
        """Stop the periodic heartbeat sender if running."""
        if self._heartbeat_sender is not None:
            self._heartbeat_sender.stop()
            logger.info("Node %s heartbeat sender stopped", self._node_id)

    # ── Serving (Task 9.1) ───────────────────────────────────────────────

    def start_serving(
        self,
        *,
        layer_engine: object,
        device: Optional[str] = None,
    ) -> ServingLoop:
        """Transition to SERVING and start the main request processing loop.

        Starts the TCP listener and begins accepting Forward messages.
        Each accepted connection is handled in its own daemon thread.

        Parameters
        ----------
        layer_engine:
            A built :class:`~meshrun.worker.layer_engine.LayerEngine`
            ready for forward pass execution.
        device:
            Torch device string.  Defaults to ``'cuda:{config.device_index}'``.

        Returns
        -------
        ServingLoop
            The running serving loop instance.

        Raises
        ------
        RuntimeError
            If the node is not in ``READY`` state.
        """
        if self._state != NodeState.READY:
            raise RuntimeError(
                f"start_serving() requires READY state, "
                f"current state is {self._state.name}"
            )

        resolved_device = device or f"cuda:{self._config.device_index}"

        serving_config = ServingConfig(
            listen_host=self._config.host,
            listen_port=self._config.data_port,
        )

        self._serving_loop = ServingLoop(
            layer_engine=layer_engine,
            layer_registry=self._layer_registry,
            connection_pool=self._connection_pool,
            resource_monitor=self._resource_monitor,
            coordinator_client=self._coordinator_client,
            config=serving_config,
            device=resolved_device,
            session_key=self._session_key,
        )

        self._serving_loop.start()
        self._transition_to(NodeState.SERVING)

        # Start periodic heartbeat sending to the Coordinator.
        if self._coordinator_client is not None and self._resource_monitor is not None:
            self.start_heartbeat()

        logger.info(
            "Node %s now SERVING on %s:%d",
            self._node_id,
            self._config.host,
            self._config.data_port,
        )

        return self._serving_loop

    def build_engine_and_serve(
        self,
        *,
        device: Optional[str] = None,
    ) -> ServingLoop:
        """Build the Layer Engine from loaded shard tensors and start serving.

        Bridges the gap between shard loading (accept_layer_assignment) and
        the serving loop by:

        1. Extracting loaded tensors from the Shard Manager's metadata.
        2. Building a :class:`LayerEngine` via :func:`build_layer_engine`.
        3. Running a warm-up forward pass to pre-allocate GPU resources.
        4. Starting the serving loop (TCP listener + forward pipeline).

        This wires together the full forward pipeline:
        Message Handler → Layer Engine → Connection Pool.

        Parameters
        ----------
        device:
            Torch device string.  Defaults to ``'cuda:{config.device_index}'``.

        Returns
        -------
        ServingLoop
            The running serving loop instance.

        Raises
        ------
        RuntimeError
            If the node is not in ``READY`` state, or if no shard metadata
            is available (shard not loaded).
        ValueError
            If the loaded tensors cannot form a valid Layer Engine.
        """
        if self._state != NodeState.READY:
            raise RuntimeError(
                f"build_engine_and_serve() requires READY state, "
                f"current state is {self._state.name}"
            )

        if self._shard_metadata is None:
            raise RuntimeError(
                "No shard metadata available — call accept_layer_assignment first"
            )

        if self._shard_metadata.load_status != LoadStatus.READY:
            raise RuntimeError(
                f"Shard is not READY (status={self._shard_metadata.load_status.name})"
            )

        assignment = self._layer_registry.assignment
        if assignment is None:
            raise RuntimeError(
                "No layer assignment in registry — cannot build engine"
            )

        resolved_device = device or f"cuda:{self._config.device_index}"

        # ── 1. Build the Layer Engine from loaded tensors ────────────────
        logger.info(
            "Node %s building Layer Engine from %d loaded tensors "
            "(layers %d–%d, is_final=%s)",
            self._node_id,
            len(self._shard_metadata.loaded_tensors),
            assignment.layer_start,
            assignment.layer_end,
            assignment.is_final_node,
        )

        engine = build_layer_engine(
            loaded_tensors=self._shard_metadata.loaded_tensors,
            layer_start=assignment.layer_start,
            layer_end=assignment.layer_end,
            is_final_node=assignment.is_final_node,
        )
        self._layer_engine = engine

        # ── 2. Warm up GPU kernels ──────────────────────────────────────
        logger.info("Node %s warming up Layer Engine", self._node_id)
        warm_up(engine)

        # ── 3. Start the serving loop ───────────────────────────────────
        return self.start_serving(layer_engine=engine, device=resolved_device)

    # ── Startup (Task 8.1) ───────────────────────────────────────────────

    def startup(self) -> NodeCapacity:
        """Initialise the node: create Resource Monitor, query GPU, generate capacity.

        This is the first phase of the worker node lifecycle.  It:

        1. Creates and starts the Resource Monitor (if not injected).
        2. Polls GPU metrics to discover available resources.
        3. Builds a ``NodeCapacity`` snapshot bounded by the user-configured
           memory limit.
        4. Transitions the node to ``REGISTERING`` state.

        Returns
        -------
        NodeCapacity
            The capacity descriptor ready to be sent to the Coordinator.

        Raises
        ------
        RuntimeError
            If the node is not in ``INITIALIZING`` state.
        """
        if self._state != NodeState.INITIALIZING:
            raise RuntimeError(
                f"startup() requires INITIALIZING state, "
                f"current state is {self._state.name}"
            )

        logger.info("Node %s starting up…", self._node_id)

        # 1. Create the Resource Monitor if not injected.
        if self._resource_monitor is None:
            self._resource_monitor = ResourceMonitor(
                gpu_memory_limit_mb=self._config.gpu_memory_limit_mb,
                poll_interval_s=self._config.poll_interval_s,
                device_index=self._config.device_index,
            )

        # 2. Start the monitor's background polling thread.
        self._resource_monitor.start()

        # 3. Query current GPU resources.
        metrics: GpuMetrics = self._resource_monitor.poll_once()

        logger.info(
            "GPU resources: total=%d MB, free=%d MB, utilization=%.1f%%",
            metrics.gpu_memory_total_mb,
            metrics.gpu_memory_free_mb,
            metrics.gpu_utilization * 100,
        )

        # 4. Build capacity descriptor.
        self._capacity = NodeCapacity(
            gpu_memory_total_mb=metrics.gpu_memory_total_mb,
            gpu_memory_free_mb=metrics.gpu_memory_free_mb,
            memory_limit_mb=self._config.gpu_memory_limit_mb,
            gpu_utilization=metrics.gpu_utilization,
        )

        # 5. Transition to REGISTERING.
        self._transition_to(NodeState.REGISTERING)
        logger.info(
            "Node %s startup complete → %s  (capacity: total=%d MB, "
            "free=%d MB, limit=%d MB)",
            self._node_id,
            self._state.name,
            self._capacity.gpu_memory_total_mb,
            self._capacity.gpu_memory_free_mb,
            self._capacity.memory_limit_mb,
        )

        return self._capacity

    # ── Full Lifecycle Orchestration (Task 8.5) ──────────────────────────

    def run_lifecycle(
        self,
        *,
        model_id: str,
        model_url: str,
        layer_start: int,
        layer_end: int,
        dtype: AssignmentDType,
        is_final_node: bool,
        downstream_node: Optional[str] = None,
        upstream_nodes: tuple[str, ...] = (),
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Drive the full startup-to-ready lifecycle.

        Orchestrates the complete state machine:
        ``INITIALIZING → REGISTERING → WAITING_ASSIGNMENT → LOADING_SHARD
        → VALIDATING → READY``

        Each phase delegates to the corresponding method (``startup``,
        ``register_with_coordinator``, ``accept_layer_assignment``,
        ``confirm_ready``).  If any phase fails the node transitions to
        ``ERROR`` and the exception propagates to the caller.

        Parameters
        ----------
        model_id:
            Identifier of the model being served.
        model_url:
            HTTP URL to the safetensors model file.
        layer_start:
            First assigned layer index (inclusive).
        layer_end:
            Last assigned layer index (inclusive).
        dtype:
            Quantization format for this shard.
        is_final_node:
            Whether this node hosts the last layers.
        downstream_node:
            TCP ``host:port`` of the next pipeline node.
        upstream_nodes:
            TCP addresses that may send data to this node.
        device:
            Target torch device.  Defaults to ``'cuda:{config.device_index}'``.
        cache_dir:
            Local directory for cached weights.

        Raises
        ------
        RuntimeError
            If the node is not in ``INITIALIZING`` state at the start.
        ConnectionError
            If registration or confirm-ready RPCs fail.
        InvalidStateTransition
            If an unexpected state transition is attempted.
        """
        logger.info(
            "Node %s: beginning full lifecycle (INITIALIZING → READY)",
            self._node_id,
        )

        # Phase 1: INITIALIZING → REGISTERING
        self.startup()

        # Phase 2: REGISTERING → WAITING_ASSIGNMENT
        self.register_with_coordinator()
        if self._state == NodeState.ERROR:
            raise RuntimeError(
                f"Node {self._node_id} registration was rejected by Coordinator"
            )

        # Phase 3: WAITING_ASSIGNMENT → LOADING_SHARD → VALIDATING
        metadata = self.accept_layer_assignment(
            model_id=model_id,
            model_url=model_url,
            layer_start=layer_start,
            layer_end=layer_end,
            dtype=dtype,
            is_final_node=is_final_node,
            downstream_node=downstream_node,
            upstream_nodes=upstream_nodes,
            device=device,
            cache_dir=cache_dir,
        )
        if self._state == NodeState.ERROR:
            raise RuntimeError(
                f"Node {self._node_id} shard load failed: "
                f"{metadata.error_message}"
            )

        # Phase 4: VALIDATING → READY
        self.confirm_ready()
        if self._state == NodeState.ERROR:
            raise RuntimeError(
                f"Node {self._node_id} confirm-ready was rejected"
            )

        # Phase 5: READY → SERVING
        # Build the Layer Engine from loaded shard tensors, warm up GPU
        # kernels, and start the TCP serving loop that wires together:
        #   Message Handler → Layer Engine → Connection Pool
        self.build_engine_and_serve(device=device)

        logger.info(
            "Node %s lifecycle complete — state is %s",
            self._node_id,
            self._state.name,
        )

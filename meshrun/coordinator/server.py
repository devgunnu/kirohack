"""Coordinator gRPC server — wires RPC methods to internal components.

Implements the ``CoordinatorServiceServicer`` generated from the proto
definition and delegates all business logic to the registry, scheduler,
and key manager.  Also provides a ``CoordinatorServer`` class for
lifecycle management (start / stop).
"""

from __future__ import annotations

import logging
from concurrent import futures

import grpc

from meshrun.coordinator.key_manager import KeyManager
from meshrun.coordinator.proto import coordinator_pb2 as pb2
from meshrun.coordinator.proto import coordinator_pb2_grpc
from meshrun.coordinator.registry import (
    DuplicateNodeError,
    HealthTracker,
    InvalidTransitionError,
    NodeRegistry,
)
from meshrun.coordinator.scheduler import (
    InsufficientCapacityError,
    LayerMap,
    PriorityQueue,
    RouteError,
    build_route,
    compute_assignments,
    handle_failure,
)

logger = logging.getLogger(__name__)


# ── gRPC Servicer ────────────────────────────────────────────────────────────


class CoordinatorServicer(coordinator_pb2_grpc.CoordinatorServiceServicer):
    """Implements every RPC defined in ``CoordinatorService``.

    Each handler translates the incoming protobuf request into calls on
    the internal components (registry, scheduler, key manager) and
    translates the result back to a protobuf response.
    """

    def __init__(
        self,
        registry: NodeRegistry,
        health_tracker: HealthTracker,
        key_manager: KeyManager,
        layer_map: LayerMap,
        priority_queue: PriorityQueue,
    ) -> None:
        self._registry = registry
        self._health_tracker = health_tracker
        self._key_manager = key_manager
        self._layer_map = layer_map
        self._priority_queue = priority_queue

    # ── Register ─────────────────────────────────────────────────────

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

    # ── Heartbeat ────────────────────────────────────────────────────

    def Heartbeat(self, request: pb2.HeartbeatRequest, context: grpc.ServicerContext) -> pb2.HeartbeatResponse:
        """Periodic health signal from worker node."""
        found = self._registry.update_heartbeat(
            node_id=request.node_id,
            gpu_utilization=request.gpu_utilization,
            memory_used_mb=request.memory_used_mb,
            active_requests=request.active_requests,
        )
        if not found:
            return pb2.HeartbeatResponse(
                acknowledged=False,
                message=f"Node '{request.node_id}' not found in registry",
            )
        return pb2.HeartbeatResponse(acknowledged=True, message="OK")

    # ── ConfirmReady ─────────────────────────────────────────────────

    def ConfirmReady(self, request: pb2.ConfirmReadyRequest, context: grpc.ServicerContext) -> pb2.ConfirmReadyResponse:
        """Worker signals shard loaded and ready to serve."""
        try:
            self._registry.mark_node_healthy(request.node_id)
            return pb2.ConfirmReadyResponse(
                acknowledged=True,
                message=f"Node '{request.node_id}' marked HEALTHY",
            )
        except (KeyError, InvalidTransitionError) as exc:
            return pb2.ConfirmReadyResponse(
                acknowledged=False,
                message=str(exc),
            )

    # ── Deregister ───────────────────────────────────────────────────

    def Deregister(self, request: pb2.DeregisterRequest, context: grpc.ServicerContext) -> pb2.DeregisterResponse:
        """Worker node graceful removal."""
        removed = self._registry.deregister_node(request.node_id)
        if not removed:
            return pb2.DeregisterResponse(
                acknowledged=False,
                message=f"Node '{request.node_id}' not found in registry",
            )
        return pb2.DeregisterResponse(
            acknowledged=True,
            message=f"Node '{request.node_id}' deregistered",
        )

    # ── RequestRoute ─────────────────────────────────────────────────

    def RequestRoute(self, request: pb2.RequestRouteRequest, context: grpc.ServicerContext) -> pb2.RequestRouteResponse:
        """Client requests execution path + session key."""
        try:
            path = build_route(
                model_id=request.model_id,
                layer_map=self._layer_map,
                registry=self._registry,
                key_manager=self._key_manager,
            )
            proto_nodes = [
                pb2.RouteNode(
                    node_id=n.node_id,
                    address=n.address,
                    layer_start=n.layer_start,
                    layer_end=n.layer_end,
                )
                for n in path.nodes
            ]
            backup_entries = [
                pb2.BackupMapEntry(node_id=nid, backup_address=addr)
                for nid, addr in path.backup_map.items()
            ]
            return pb2.RequestRouteResponse(
                request_id=int(path.request_id, 16) if path.request_id else 0,
                session_key=path.session_key,
                nodes=proto_nodes,
                backup_map=backup_entries,
            )
        except RouteError as exc:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return pb2.RequestRouteResponse()

    # ── ReportFailure ────────────────────────────────────────────────

    def ReportFailure(self, request: pb2.ReportFailureRequest, context: grpc.ServicerContext) -> pb2.ReportFailureResponse:
        """Worker reports downstream node failure."""
        reroute = handle_failure(
            request_id=str(request.request_id),
            failed_node_id=request.failed_node_id,
            layer_map=self._layer_map,
            registry=self._registry,
        )
        return pb2.ReportFailureResponse(
            acknowledged=True,
            reroute=pb2.RerouteInfo(
                backup_addr=reroute.backup_addr or "",
                message=reroute.message,
            ),
            message=reroute.message,
        )

    # ── TriggerAssignment ────────────────────────────────────────────

    def TriggerAssignment(self, request: pb2.TriggerAssignmentRequest, context: grpc.ServicerContext) -> pb2.TriggerAssignmentResponse:
        """Admin triggers layer assignment computation."""
        dtype_map = {
            pb2.DTYPE_FP16: "fp16",
            pb2.DTYPE_INT8: "int8",
        }
        dtype_str = dtype_map.get(request.dtype)
        if dtype_str is None:
            return pb2.TriggerAssignmentResponse(
                success=False,
                message=f"Unsupported dtype: {request.dtype}",
            )

        try:
            healthy_nodes = self._registry.get_all_healthy_nodes()
            plan = compute_assignments(
                model_id=request.model_id,
                total_layers=request.total_layers,
                dtype=dtype_str,
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

            return pb2.TriggerAssignmentResponse(
                success=True,
                message=(
                    f"Assigned {request.total_layers} layers across "
                    f"{len(plan.assignments)} nodes for model '{request.model_id}'"
                ),
            )
        except (ValueError, InsufficientCapacityError) as exc:
            return pb2.TriggerAssignmentResponse(
                success=False,
                message=str(exc),
            )

    # ── AcceptLayerAssignment ────────────────────────────────────────

    def AcceptLayerAssignment(self, request: pb2.AcceptLayerAssignmentRequest, context: grpc.ServicerContext) -> pb2.AcceptLayerAssignmentResponse:
        """Coordinator pushes assignment to worker (via worker's gRPC address).

        This RPC is defined in the proto for completeness but is invoked
        by the Coordinator *as a client* calling the worker's gRPC
        endpoint.  On the Coordinator side this is a no-op stub.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details(
            "AcceptLayerAssignment is pushed to workers; "
            "it is not handled on the Coordinator"
        )
        return pb2.AcceptLayerAssignmentResponse(acknowledged=False)


# ── Server Lifecycle ─────────────────────────────────────────────────────────

DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 50051
DEFAULT_MAX_WORKERS: int = 10


class CoordinatorServer:
    """Manages the full lifecycle of the Coordinator gRPC server.

    Initialises all internal components (registry, health tracker,
    scheduler, key manager) and exposes ``start`` / ``stop`` methods.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_workers: int = DEFAULT_MAX_WORKERS,
        heartbeat_interval_s: float = 5.0,
        missed_threshold: int = 3,
        dead_threshold: int = 5,
    ) -> None:
        self._host = host
        self._port = port

        # Internal components
        self._registry = NodeRegistry()
        self._key_manager = KeyManager()
        self._layer_map = LayerMap()
        self._priority_queue = PriorityQueue()
        self._health_tracker = HealthTracker(
            registry=self._registry,
            heartbeat_interval_s=heartbeat_interval_s,
            missed_threshold=missed_threshold,
            dead_threshold=dead_threshold,
        )

        # gRPC server
        self._servicer = CoordinatorServicer(
            registry=self._registry,
            health_tracker=self._health_tracker,
            key_manager=self._key_manager,
            layer_map=self._layer_map,
            priority_queue=self._priority_queue,
        )
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
        )
        coordinator_pb2_grpc.add_CoordinatorServiceServicer_to_server(
            self._servicer, self._server,
        )

    # ── Public API ───────────────────────────────────────────────────

    def start(self) -> None:
        """Start the gRPC server and the health tracker background thread."""
        address = f"{self._host}:{self._port}"
        self._server.add_insecure_port(address)
        self._server.start()
        self._health_tracker.start()
        logger.info("Coordinator server started on %s", address)

    def stop(self, grace: float = 5.0) -> None:
        """Gracefully shut down the server and all background threads."""
        logger.info("Coordinator server shutting down (grace=%.1fs)…", grace)
        self._health_tracker.stop()
        self._server.stop(grace=grace)
        logger.info("Coordinator server stopped")

    # ── Accessors (useful for testing / admin) ───────────────────────

    @property
    def registry(self) -> NodeRegistry:
        return self._registry

    @property
    def key_manager(self) -> KeyManager:
        return self._key_manager

    @property
    def layer_map(self) -> LayerMap:
        return self._layer_map

    @property
    def priority_queue(self) -> PriorityQueue:
        return self._priority_queue

    @property
    def servicer(self) -> CoordinatorServicer:
        return self._servicer

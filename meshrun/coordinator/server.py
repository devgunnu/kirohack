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
    AssignmentPlan,
    InsufficientCapacityError,
    LayerMap,
    PriorityQueue,
    RouteError,
    build_route,
    compute_assignments,
    handle_failure,
)
from meshrun.coordinator.registry import NodeStatus

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
        server: "CoordinatorServer",
    ) -> None:
        self._registry = registry
        self._health_tracker = health_tracker
        self._key_manager = key_manager
        self._layer_map = layer_map
        self._priority_queue = priority_queue
        self._server = server

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

    # ── GetNetworkStatus ─────────────────────────────────────────────

    def GetNetworkStatus(
        self,
        request: pb2.GetNetworkStatusRequest,
        context: grpc.ServicerContext,
    ) -> pb2.GetNetworkStatusResponse:
        """Return the current network status for CLI and dashboard."""
        nodes = self._registry.get_all_nodes()

        node_infos: list[pb2.NodeInfo] = []
        covered_layers: set[int] = set()
        for node in nodes:
            # Map NodeStatus to display string
            if node.status == NodeStatus.HEALTHY:
                status = "active"
            elif node.status == NodeStatus.UNHEALTHY:
                status = "idle"
            else:
                status = "unhealthy"

            node_info = pb2.NodeInfo(
                node_id=node.node_id,
                address=node.address,
                grpc_address=node.grpc_address,
                layer_start=node.layer_start if node.layer_start is not None else 0,
                layer_end=node.layer_end if node.layer_end is not None else 0,
                status=status,
                gpu_utilization=node.gpu_utilization,
                memory_used_mb=node.memory_used_mb or 0,
                memory_total_mb=node.gpu_memory_total_mb,
                requests_served=0,
                credits_earned=0.0,
                last_heartbeat_ms=int(node.last_seen * 1000) if node.last_seen else 0,
            )
            node_infos.append(node_info)

            if node.layer_start is not None and node.layer_end is not None:
                covered_layers.update(range(node.layer_start, node.layer_end + 1))

        # Queue depth
        try:
            queue_depth = len(self._priority_queue)
        except TypeError:
            queue_depth = 0

        # Model info from server's model config (if set), else infer from layer map
        model_id = ""
        total_layers = 0
        model_config = self._server._model_config
        if model_config:
            model_id = model_config.get("model_id", "")
            total_layers = model_config.get("total_layers", 0)
        else:
            entries = self._layer_map.get_all_entries()
            if entries:
                total_layers = max(e.layer_end for e in entries) + 1

        return pb2.GetNetworkStatusResponse(
            active_nodes=len([n for n in node_infos if n.status in ("active", "idle")]),
            total_layers=total_layers,
            covered_layers=len(covered_layers),
            model_id=model_id,
            queue_depth=queue_depth,
            nodes=node_infos,
            queue=[],
        )

    # ── Auto-Assignment Helpers ──────────────────────────────────────

    def _maybe_trigger_assignment(self) -> None:
        """Check if we have model config and enough nodes, then trigger assignment."""
        model_config = self._server._model_config
        if not model_config:
            logger.debug("No model config set, skipping auto-assignment")
            return

        healthy_nodes = self._registry.get_all_healthy_nodes()
        # Allow REGISTERED nodes too, since auto-assignment triggers on register
        # before nodes have a chance to send ConfirmReady.
        all_nodes = self._registry.get_all_nodes()
        candidate_nodes = [
            n for n in all_nodes
            if n.status in (NodeStatus.REGISTERED, NodeStatus.HEALTHY)
        ]
        nodes_for_plan = candidate_nodes if candidate_nodes else healthy_nodes
        if not nodes_for_plan:
            logger.debug("No candidate nodes, skipping auto-assignment")
            return

        try:
            plan = compute_assignments(
                model_id=model_config["model_id"],
                total_layers=model_config["total_layers"],
                dtype=model_config["dtype"],
                nodes=nodes_for_plan,
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
        except Exception:
            logger.exception("Auto-assignment failed")

    def _push_assignments_to_workers(self, plan: AssignmentPlan) -> None:
        """Push AcceptLayerAssignment RPCs to all workers in the plan."""
        # Build a lookup of node_id -> grpc_address from the registry
        nodes = self._registry.get_all_nodes()
        grpc_addresses: dict[str, str] = {n.node_id: n.grpc_address for n in nodes}

        # Map dtype string to proto enum
        dtype_to_proto = {
            "fp16": pb2.DTYPE_FP16,
            "int8": pb2.DTYPE_INT8,
        }

        model_config = self._server._model_config
        if not model_config:
            logger.warning("No model config set, cannot push assignments")
            return

        proto_dtype = dtype_to_proto.get(model_config["dtype"], pb2.DTYPE_INT8)
        total_layers = model_config["total_layers"]

        entries = list(plan.assignments)

        for idx, entry in enumerate(entries):
            grpc_addr = grpc_addresses.get(entry.primary_node_id)
            if not grpc_addr:
                logger.warning(
                    "No gRPC address for node %s, skipping assignment push",
                    entry.primary_node_id,
                )
                continue

            # Determine if this is the final node
            is_final = (entry.layer_end == total_layers - 1)

            # Find downstream node (next entry in the plan)
            downstream_addr = ""
            if idx + 1 < len(entries):
                downstream_addr = entries[idx + 1].primary_address

            # Find upstream nodes (previous entry in the plan)
            upstream_addrs: list[str] = []
            if idx > 0:
                upstream_addrs.append(entries[idx - 1].primary_address)

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

        # Model configuration (set via TriggerAssignment or CLI)
        self._model_config: dict | None = None

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
            server=self,
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

    @property
    def model_config(self) -> dict | None:
        """The current model configuration for auto-assignment."""
        return self._model_config

    def set_model_config(
        self,
        model_id: str,
        total_layers: int,
        dtype: str,
        model_url: str,
    ) -> None:
        """Set the model configuration for auto-assignment."""
        self._model_config = {
            "model_id": model_id,
            "total_layers": total_layers,
            "dtype": dtype,
            "model_url": model_url,
        }

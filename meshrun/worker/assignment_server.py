"""Worker-side gRPC server for receiving layer assignment pushes from the Coordinator."""

from __future__ import annotations

import logging
from concurrent import futures
from typing import TYPE_CHECKING

import grpc

from meshrun.coordinator.proto import coordinator_pb2 as pb2
from meshrun.coordinator.proto import coordinator_pb2_grpc

if TYPE_CHECKING:
    from meshrun.worker.node import WorkerNode

logger = logging.getLogger(__name__)


class WorkerAssignmentServicer(coordinator_pb2_grpc.CoordinatorServiceServicer):
    """Receives AcceptLayerAssignment pushes from the Coordinator."""

    def __init__(self, worker_node: "WorkerNode") -> None:
        self._worker = worker_node

    def AcceptLayerAssignment(
        self,
        request: pb2.AcceptLayerAssignmentRequest,
        context: grpc.ServicerContext,
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

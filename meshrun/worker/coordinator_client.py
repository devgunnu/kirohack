"""
Coordinator Client — gRPC client for control plane communication.

Encapsulates all gRPC calls from a Worker Node to the Coordinator:
registration, heartbeat, ready confirmation, and failure reporting.

The client is designed around a simple protocol-buffer-style interface
using plain dataclasses for request/response types.  When the real
Coordinator gRPC service is available, the ``GrpcCoordinatorClient``
implementation translates these into actual gRPC calls.  For testing
and offline development, a stub implementation can be injected.

Validates: Requirements 7.1, 7.2, 7.5
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional

logger = logging.getLogger(__name__)


# ── Custom Exceptions ────────────────────────────────────────────────────────


class CoordinatorUnavailableError(Exception):
    """Raised when the Coordinator is unreachable (UNAVAILABLE)."""


class CoordinatorDeadlineExceededError(Exception):
    """Raised when a gRPC call to the Coordinator times out (DEADLINE_EXCEEDED)."""


class CoordinatorRpcError(Exception):
    """Raised for any other gRPC error from the Coordinator."""


# ── Response Status ──────────────────────────────────────────────────────────


class RegistrationStatus(IntEnum):
    """Outcome of a ``Register`` RPC."""

    OK = auto()
    REJECTED = auto()
    ERROR = auto()


# ── Request / Response Data Structures ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CapacityInfo:
    """GPU capacity descriptor included in the registration request."""

    gpu_memory_total_mb: int
    gpu_memory_free_mb: int
    memory_limit_mb: int
    gpu_utilization: float


@dataclass(frozen=True, slots=True)
class RegisterRequest:
    """Payload for the ``Register`` RPC sent to the Coordinator.

    Contains everything the Coordinator needs to add this node to its
    registry and make layer-assignment decisions.
    """

    node_id: str
    """Unique identifier for this worker node."""

    address: str
    """TCP ``host:port`` for the data plane listener."""

    grpc_address: str
    """gRPC ``host:port`` for control plane callbacks."""

    capacity: CapacityInfo
    """GPU capacity descriptor."""

    layers_hosted: tuple[int, int] | None = None
    """``(layer_start, layer_end)`` if layers are already loaded, else ``None``."""


@dataclass(frozen=True, slots=True)
class RegisterResponse:
    """Response from the Coordinator's ``Register`` RPC."""

    status: RegistrationStatus
    message: str = ""


@dataclass(frozen=True, slots=True)
class ConfirmReadyRequest:
    """Payload for the ``ConfirmReady`` RPC sent to the Coordinator.

    Signals that the worker node has successfully loaded and validated
    its assigned shard and is ready to serve inference requests.
    """

    node_id: str
    """Unique identifier for this worker node."""

    layers_loaded: tuple[int, int]
    """``(layer_start, layer_end)`` inclusive range of loaded layers."""


@dataclass(frozen=True, slots=True)
class ConfirmReadyResponse:
    """Response from the Coordinator's ``ConfirmReady`` RPC."""

    acknowledged: bool
    """Whether the Coordinator accepted the ready confirmation."""

    message: str = ""


@dataclass(frozen=True, slots=True)
class HeartbeatRequest:
    """Payload for the ``Heartbeat`` RPC sent to the Coordinator.

    Contains the node's identity and current resource metrics so the
    Coordinator can update its health tracker and scheduling decisions.
    """

    node_id: str
    """Unique identifier for this worker node."""

    gpu_utilization: float
    """Current GPU compute utilization (0.0–1.0)."""

    memory_used_mb: int
    """Current GPU memory usage in MB."""

    active_requests: int
    """Number of forward passes currently in-flight."""


@dataclass(frozen=True, slots=True)
class HeartbeatResponse:
    """Response from the Coordinator's ``Heartbeat`` RPC."""

    acknowledged: bool
    """Whether the Coordinator accepted the heartbeat."""

    message: str = ""


@dataclass(frozen=True, slots=True)
class ReportFailureRequest:
    """Payload for the ``ReportFailure`` RPC sent to the Coordinator.

    Sent when a worker node detects that its downstream peer is
    unreachable so the Coordinator can provide a backup route.
    """

    request_id: int
    """The inference request that was in-flight when the failure occurred."""

    failed_node_id: str
    """Identifier (or ``host:port``) of the downstream node that failed."""

    reporting_node_id: str
    """Identifier of the node reporting the failure."""


@dataclass(frozen=True, slots=True)
class RerouteInfo:
    """Backup routing information returned by the Coordinator.

    Contains the address of a backup node that can handle the same
    layer range as the failed downstream node.
    """

    backup_addr: Optional[str] = None
    """TCP ``host:port`` of the backup node, or ``None`` if no backup available."""

    message: str = ""


@dataclass(frozen=True, slots=True)
class ReportFailureResponse:
    """Response from the Coordinator's ``ReportFailure`` RPC."""

    acknowledged: bool
    """Whether the Coordinator accepted the failure report."""

    reroute: Optional[RerouteInfo] = None
    """Backup routing info, present when a backup node is available."""

    message: str = ""


# ── Abstract Interface ───────────────────────────────────────────────────────


class CoordinatorClient(abc.ABC):
    """Abstract interface for communicating with the Coordinator.

    Concrete implementations handle the transport (gRPC, in-process stub,
    etc.).  The Worker Node only depends on this interface.
    """

    @abc.abstractmethod
    def register(self, request: RegisterRequest) -> RegisterResponse:
        """Send a ``Register`` RPC to the Coordinator.

        Parameters
        ----------
        request:
            Registration payload with node identity and capacity.

        Returns
        -------
        RegisterResponse
            The Coordinator's acknowledgement.
        """
        ...

    @abc.abstractmethod
    def confirm_ready(self, request: ConfirmReadyRequest) -> ConfirmReadyResponse:
        """Send a ``ConfirmReady`` RPC to the Coordinator.

        Signals that the worker node has loaded and validated its shard
        and is ready to receive Forward requests.

        Parameters
        ----------
        request:
            Confirmation payload with node identity and loaded layers.

        Returns
        -------
        ConfirmReadyResponse
            The Coordinator's acknowledgement.
        """
        ...

    @abc.abstractmethod
    def heartbeat(self, request: HeartbeatRequest) -> HeartbeatResponse:
        """Send a ``Heartbeat`` RPC to the Coordinator.

        Reports the node's current resource metrics so the Coordinator
        can update its health tracker and scheduling decisions.

        Parameters
        ----------
        request:
            Heartbeat payload with node identity and load metrics.

        Returns
        -------
        HeartbeatResponse
            The Coordinator's acknowledgement.
        """
        ...

    @abc.abstractmethod
    def report_failure(self, request: ReportFailureRequest) -> ReportFailureResponse:
        """Send a ``ReportFailure`` RPC to the Coordinator.

        Called when a downstream node is unreachable so the Coordinator
        can provide backup routing information.

        Parameters
        ----------
        request:
            Failure report with the request ID and failed node identity.

        Returns
        -------
        ReportFailureResponse
            The Coordinator's acknowledgement with optional reroute info.
        """
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Release any transport resources (channels, connections)."""
        ...


# ── gRPC Implementation ─────────────────────────────────────────────────────


class GrpcCoordinatorClient(CoordinatorClient):
    """Coordinator client that communicates over gRPC.

    Requires ``grpcio`` to be installed.  The actual proto-generated stubs
    will be wired in once the Coordinator team publishes the service
    definition.  For now this implementation creates a gRPC channel and
    provides the scaffolding for the ``Register`` call.
    """

    def __init__(self, coordinator_address: str) -> None:
        self._address = coordinator_address
        self._channel = None
        self._stub = None
        self._connect()

    # ── Connection management ────────────────────────────────────────────

    def _connect(self) -> None:
        """Establish the gRPC channel and CoordinatorService stub."""
        try:
            import grpc  # type: ignore[import-untyped]
            from meshrun.coordinator.proto import coordinator_pb2_grpc

            self._channel = grpc.insecure_channel(self._address)
            self._stub = coordinator_pb2_grpc.CoordinatorServiceStub(
                self._channel
            )
            logger.info(
                "gRPC channel + stub opened to Coordinator at %s",
                self._address,
            )
        except ImportError:
            logger.warning(
                "grpcio or proto stubs not available — "
                "GrpcCoordinatorClient will not be able to communicate "
                "with the Coordinator"
            )
            self._channel = None
            self._stub = None

    def _handle_grpc_error(self, exc: Exception, rpc_name: str):
        """Inspect a gRPC exception, log it, and raise a typed Python error.

        Parameters
        ----------
        exc:
            The exception caught from a stub call.
        rpc_name:
            Human-readable name of the RPC (e.g. ``"Register"``).

        Raises
        ------
        CoordinatorUnavailableError
            If the gRPC status code is UNAVAILABLE.
        CoordinatorDeadlineExceededError
            If the gRPC status code is DEADLINE_EXCEEDED.
        CoordinatorRpcError
            For all other gRPC errors.
        """
        try:
            import grpc  # type: ignore[import-untyped]
        except ImportError:
            raise exc from None

        if not isinstance(exc, grpc.RpcError):
            raise exc

        code = exc.code()
        details = exc.details()
        logger.error(
            "%s RPC failed: code=%s, details=%s",
            rpc_name,
            code.name,
            details,
        )

        if code == grpc.StatusCode.UNAVAILABLE:
            raise CoordinatorUnavailableError(
                f"{rpc_name} failed: Coordinator unavailable — {details}"
            ) from exc
        elif code == grpc.StatusCode.DEADLINE_EXCEEDED:
            raise CoordinatorDeadlineExceededError(
                f"{rpc_name} failed: deadline exceeded — {details}"
            ) from exc
        else:
            raise CoordinatorRpcError(
                f"{rpc_name} failed: {code.name} — {details}"
            ) from exc

    # ── CoordinatorClient interface ──────────────────────────────────────

    def register(self, request: RegisterRequest) -> RegisterResponse:
        """Send a ``Register`` RPC to the Coordinator.

        Translates the ``RegisterRequest`` dataclass into the generated
        protobuf message, invokes the stub, and translates the response
        back to a ``RegisterResponse`` dataclass.
        """
        from meshrun.coordinator.proto import coordinator_pb2

        logger.info(
            "Registering with Coordinator at %s: node_id=%s, "
            "address=%s, grpc_address=%s, capacity=(total=%d MB, "
            "free=%d MB, limit=%d MB, util=%.1f%%)",
            self._address,
            request.node_id,
            request.address,
            request.grpc_address,
            request.capacity.gpu_memory_total_mb,
            request.capacity.gpu_memory_free_mb,
            request.capacity.memory_limit_mb,
            request.capacity.gpu_utilization * 100,
        )

        if self._stub is None:
            logger.warning(
                "No gRPC stub available — returning synthetic OK"
            )
            return RegisterResponse(
                status=RegistrationStatus.OK,
                message="synthetic-ack (no grpc stub)",
            )

        # Build protobuf Capacity message
        pb_capacity = coordinator_pb2.Capacity(
            gpu_memory_total_mb=request.capacity.gpu_memory_total_mb,
            gpu_memory_free_mb=request.capacity.gpu_memory_free_mb,
            memory_limit_mb=request.capacity.memory_limit_mb,
            gpu_utilization=request.capacity.gpu_utilization,
        )

        # Build protobuf RegisterRequest
        pb_request = coordinator_pb2.RegisterRequest(
            node_id=request.node_id,
            address=request.address,
            grpc_address=request.grpc_address,
            capacity=pb_capacity,
            layers_hosted_start=request.layers_hosted[0] if request.layers_hosted else 0,
            layers_hosted_end=request.layers_hosted[1] if request.layers_hosted else 0,
        )

        # Map proto RegistrationStatus enum → Python RegistrationStatus
        _PROTO_STATUS_MAP = {
            coordinator_pb2.REGISTRATION_STATUS_OK: RegistrationStatus.OK,
            coordinator_pb2.REGISTRATION_STATUS_REJECTED: RegistrationStatus.REJECTED,
            coordinator_pb2.REGISTRATION_STATUS_ERROR: RegistrationStatus.ERROR,
        }

        try:
            pb_response = self._stub.Register(pb_request)
        except Exception as exc:
            return self._handle_grpc_error(exc, "Register")

        status = _PROTO_STATUS_MAP.get(
            pb_response.status, RegistrationStatus.ERROR
        )

        logger.info(
            "Register response from Coordinator: status=%s, message=%s",
            status.name,
            pb_response.message,
        )

        return RegisterResponse(
            status=status,
            message=pb_response.message,
        )

    def confirm_ready(self, request: ConfirmReadyRequest) -> ConfirmReadyResponse:
        """Send a ``ConfirmReady`` RPC to the Coordinator.

        Translates the ``ConfirmReadyRequest`` dataclass into the generated
        protobuf message, invokes the stub, and translates the response
        back to a ``ConfirmReadyResponse`` dataclass.
        """
        from meshrun.coordinator.proto import coordinator_pb2

        logger.info(
            "Sending ConfirmReady to Coordinator at %s: node_id=%s, "
            "layers_loaded=%s",
            self._address,
            request.node_id,
            request.layers_loaded,
        )

        if self._stub is None:
            logger.warning(
                "No gRPC stub available — returning synthetic ack"
            )
            return ConfirmReadyResponse(
                acknowledged=True,
                message="synthetic-ack (no grpc stub)",
            )

        # Build protobuf ConfirmReadyRequest
        pb_request = coordinator_pb2.ConfirmReadyRequest(
            node_id=request.node_id,
            layer_start=request.layers_loaded[0],
            layer_end=request.layers_loaded[1],
        )

        try:
            pb_response = self._stub.ConfirmReady(pb_request)
        except Exception as exc:
            return self._handle_grpc_error(exc, "ConfirmReady")

        logger.info(
            "ConfirmReady response from Coordinator: acknowledged=%s, "
            "message=%s",
            pb_response.acknowledged,
            pb_response.message,
        )

        return ConfirmReadyResponse(
            acknowledged=pb_response.acknowledged,
            message=pb_response.message,
        )

    def heartbeat(self, request: HeartbeatRequest) -> HeartbeatResponse:
        """Send a ``Heartbeat`` RPC to the Coordinator.

        Translates the ``HeartbeatRequest`` dataclass into the generated
        protobuf message, invokes the stub, and translates the response
        back to a ``HeartbeatResponse`` dataclass.
        """
        from meshrun.coordinator.proto import coordinator_pb2

        logger.debug(
            "Sending Heartbeat to Coordinator at %s: node_id=%s, "
            "gpu_util=%.1f%%, mem_used=%d MB, active=%d",
            self._address,
            request.node_id,
            request.gpu_utilization * 100,
            request.memory_used_mb,
            request.active_requests,
        )

        if self._stub is None:
            logger.warning(
                "No gRPC stub available — returning synthetic ack"
            )
            return HeartbeatResponse(
                acknowledged=True,
                message="synthetic-ack (no grpc stub)",
            )

        # Build protobuf HeartbeatRequest
        pb_request = coordinator_pb2.HeartbeatRequest(
            node_id=request.node_id,
            gpu_utilization=request.gpu_utilization,
            memory_used_mb=request.memory_used_mb,
            active_requests=request.active_requests,
        )

        try:
            pb_response = self._stub.Heartbeat(pb_request)
        except Exception as exc:
            return self._handle_grpc_error(exc, "Heartbeat")

        logger.debug(
            "Heartbeat response from Coordinator: acknowledged=%s, "
            "message=%s",
            pb_response.acknowledged,
            pb_response.message,
        )

        return HeartbeatResponse(
            acknowledged=pb_response.acknowledged,
            message=pb_response.message,
        )

    def report_failure(self, request: ReportFailureRequest) -> ReportFailureResponse:
        """Send a ``ReportFailure`` RPC to the Coordinator.

        Translates the ``ReportFailureRequest`` dataclass into the generated
        protobuf message, invokes the stub, and translates the response
        back to a ``ReportFailureResponse`` dataclass with ``RerouteInfo``.
        """
        from meshrun.coordinator.proto import coordinator_pb2

        logger.warning(
            "Reporting failure to Coordinator at %s: request_id=%d, "
            "failed_node=%s, reporting_node=%s",
            self._address,
            request.request_id,
            request.failed_node_id,
            request.reporting_node_id,
        )

        if self._stub is None:
            logger.warning(
                "No gRPC stub available — returning synthetic ack "
                "with no backup"
            )
            return ReportFailureResponse(
                acknowledged=True,
                reroute=None,
                message="synthetic-ack (no grpc stub)",
            )

        # Build protobuf ReportFailureRequest
        pb_request = coordinator_pb2.ReportFailureRequest(
            request_id=request.request_id,
            failed_node_id=request.failed_node_id,
            reporting_node_id=request.reporting_node_id,
        )

        try:
            pb_response = self._stub.ReportFailure(pb_request)
        except Exception as exc:
            return self._handle_grpc_error(exc, "ReportFailure")

        # Translate RerouteInfo if present
        reroute = None
        if pb_response.HasField("reroute"):
            reroute = RerouteInfo(
                backup_addr=pb_response.reroute.backup_addr or None,
                message=pb_response.reroute.message,
            )

        logger.info(
            "ReportFailure response from Coordinator: acknowledged=%s, "
            "reroute=%s, message=%s",
            pb_response.acknowledged,
            reroute,
            pb_response.message,
        )

        return ReportFailureResponse(
            acknowledged=pb_response.acknowledged,
            reroute=reroute,
            message=pb_response.message,
        )

    def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            logger.info("gRPC channel to %s closed", self._address)
            self._channel = None


# ── Stub Implementation (for testing / offline dev) ──────────────────────────


class StubCoordinatorClient(CoordinatorClient):
    """In-process stub that records calls for testing.

    Always returns ``RegistrationStatus.OK`` unless configured otherwise.
    """

    def __init__(
        self,
        *,
        register_status: RegistrationStatus = RegistrationStatus.OK,
        register_message: str = "stub-ack",
        confirm_ready_acknowledged: bool = True,
        confirm_ready_message: str = "stub-ack",
        heartbeat_acknowledged: bool = True,
        heartbeat_message: str = "stub-ack",
        report_failure_response: Optional[ReportFailureResponse] = None,
    ) -> None:
        self._register_status = register_status
        self._register_message = register_message
        self._confirm_ready_acknowledged = confirm_ready_acknowledged
        self._confirm_ready_message = confirm_ready_message
        self._heartbeat_acknowledged = heartbeat_acknowledged
        self._heartbeat_message = heartbeat_message
        self._report_failure_response = report_failure_response
        self.register_calls: list[RegisterRequest] = []
        self.confirm_ready_calls: list[ConfirmReadyRequest] = []
        self.heartbeat_calls: list[HeartbeatRequest] = []
        self.report_failure_calls: list[ReportFailureRequest] = []

    def register(self, request: RegisterRequest) -> RegisterResponse:
        self.register_calls.append(request)
        return RegisterResponse(
            status=self._register_status,
            message=self._register_message,
        )

    def confirm_ready(self, request: ConfirmReadyRequest) -> ConfirmReadyResponse:
        self.confirm_ready_calls.append(request)
        return ConfirmReadyResponse(
            acknowledged=self._confirm_ready_acknowledged,
            message=self._confirm_ready_message,
        )

    def heartbeat(self, request: HeartbeatRequest) -> HeartbeatResponse:
        self.heartbeat_calls.append(request)
        return HeartbeatResponse(
            acknowledged=self._heartbeat_acknowledged,
            message=self._heartbeat_message,
        )

    def report_failure(self, request: ReportFailureRequest) -> ReportFailureResponse:
        self.report_failure_calls.append(request)
        if self._report_failure_response is not None:
            return self._report_failure_response
        return ReportFailureResponse(
            acknowledged=True,
            reroute=None,
            message="stub-ack",
        )

    def close(self) -> None:
        pass

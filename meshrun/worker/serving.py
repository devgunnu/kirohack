"""
Serving Loop — main request processing loop for the data plane.

Accepts incoming TCP connections, reads Forward messages using the binary
protocol, runs the forward pass through the Layer Engine, and sends
results to the downstream node (or back to the client if final node).

Validates: Requirements 7.3
"""

from __future__ import annotations

import logging
import socket
import threading
from dataclasses import dataclass, field
from typing import Optional

from meshrun.worker.connection_pool import ConnectionPool
from meshrun.worker.coordinator_client import (
    CoordinatorClient,
    ReportFailureRequest,
)
from meshrun.worker.layer_registry import LayerAssignmentRegistry
from meshrun.worker.protocol import (
    DType,
    Header,
    MessageType,
    read_message,
    read_message_secure,
    write_message,
    write_message_secure,
)
from meshrun.worker.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


# ── Optional torch import ────────────────────────────────────────────────────

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ServingConfig:
    """Configuration for the serving loop.

    Parameters
    ----------
    listen_host:
        Bind address for the TCP listener.
    listen_port:
        TCP port for the data plane listener.
    session_key:
        Optional AES-256 session key (32 bytes) for encrypted data plane
        traffic.  When provided, the serving loop uses
        ``read_message_secure`` / ``write_message_secure`` instead of the
        plaintext variants.
    """

    listen_host: str = "0.0.0.0"
    listen_port: int = 9100
    session_key: Optional[bytes] = None


@dataclass(slots=True)
class ServingStats:
    """Mutable counters for serving loop observability."""

    requests_processed: int = 0
    requests_failed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_success(self) -> None:
        """Increment the successful request counter."""
        with self._lock:
            self.requests_processed += 1

    def record_failure(self) -> None:
        """Increment the failed request counter."""
        with self._lock:
            self.requests_failed += 1


# ── Tensor ↔ Protocol Conversion Helpers ────────────────────────────────────


def _tensor_to_flat_list(tensor: object, dtype_enum: int) -> list[float] | list[int]:
    """Convert a torch.Tensor to a flat list suitable for write_message.

    Parameters
    ----------
    tensor:
        A torch.Tensor (any shape).  Will be flattened to row-major order.
    dtype_enum:
        Protocol DType enum value (DType.FP16 or DType.INT8).

    Returns
    -------
    list[float] | list[int]
        Flat list of numeric values in row-major order.
    """
    flat = tensor.detach().cpu().contiguous().view(-1)
    if dtype_enum == DType.INT8:
        return flat.to(torch.int8).tolist()
    return flat.to(torch.float32).tolist()


def _flat_list_to_tensor(
    data: list[float] | list[int],
    shape: tuple[int, ...],
    dtype_enum: int,
    device: str,
) -> object:
    """Convert a flat list from read_message back to a torch.Tensor.

    Parameters
    ----------
    data:
        Flat list of numeric values (row-major order).
    shape:
        Target tensor shape.
    dtype_enum:
        Protocol DType enum value.
    device:
        Target torch device (e.g. ``'cuda:0'``).

    Returns
    -------
    torch.Tensor
        Reconstructed tensor on the specified device.
    """
    if dtype_enum == DType.INT8:
        t = torch.tensor(data, dtype=torch.int8)
    else:
        t = torch.tensor(data, dtype=torch.float16)
    return t.reshape(shape).to(device)


def _build_response_header(
    request_header: Header,
    output_tensor: object,
    dtype_enum: int,
    is_final_node: bool,
) -> Header:
    """Build a response header from the request header and output tensor.

    Parameters
    ----------
    request_header:
        The original incoming message header (carries request_id, step_id).
    output_tensor:
        The output torch.Tensor from the forward pass.
    dtype_enum:
        Protocol DType enum value.
    is_final_node:
        Whether this node is the final node (RESULT vs FORWARD).

    Returns
    -------
    Header
        A new header describing the output tensor.
    """
    shape = tuple(output_tensor.shape)
    num_dims = len(shape)

    # Pad dims to 4 elements
    dims = tuple(shape) + (0,) * (4 - num_dims)

    # Compute payload size
    element_count = 1
    for d in shape:
        element_count *= d
    dtype_size = 2 if dtype_enum == DType.FP16 else 1
    payload_size = element_count * dtype_size

    msg_type = MessageType.RESULT if is_final_node else MessageType.FORWARD

    return Header(
        message_type=int(msg_type),
        request_id=request_header.request_id,
        step_id=request_header.step_id,
        payload_size=payload_size,
        dtype=dtype_enum,
        num_dims=num_dims,
        dims=dims,
        reserved=0,
    )


def _build_error_header(request_id: int, step_id: int) -> Header:
    """Build an ERROR header to send back upstream when all downstream attempts fail.

    The ERROR message carries no tensor payload — dims are all 1 with a
    single fp16 element (2 bytes) as a minimal placeholder.

    Parameters
    ----------
    request_id:
        The original request ID from the incoming Forward message.
    step_id:
        The original step ID from the incoming Forward message.

    Returns
    -------
    Header
        An ERROR header with minimal payload descriptor.
    """
    return Header(
        message_type=int(MessageType.ERROR),
        request_id=request_id,
        step_id=step_id,
        payload_size=2,  # 1 fp16 element = 2 bytes
        dtype=int(DType.FP16),
        num_dims=1,
        dims=(1, 0, 0, 0),
        reserved=0,
    )


def _send_error_upstream(
    client_sock: socket.socket,
    request_id: int,
    step_id: int,
    session_key: Optional[bytes] = None,
) -> None:
    """Send an ERROR message back to the upstream node / client.

    Called when both the primary downstream and backup node have failed.
    The upstream caller can inspect the message_type to detect the error.

    Parameters
    ----------
    client_sock:
        The TCP socket connected to the upstream sender.
    request_id:
        The original request ID.
    step_id:
        The original step ID.
    session_key:
        Optional AES-256 session key. When provided, the message is
        sent encrypted via ``write_message_secure``; otherwise falls
        back to plaintext ``write_message``.
    """
    error_header = _build_error_header(request_id, step_id)
    # Minimal payload: a single zero-valued fp16 element.
    error_payload: list[float] = [0.0]
    try:
        if session_key is not None:
            write_message_secure(
                client_sock, error_header, error_payload, session_key
            )
        else:
            write_message(client_sock, error_header, error_payload)
        logger.info(
            "Sent ERROR response upstream for request_id=%d", request_id
        )
    except (ConnectionError, OSError, socket.timeout) as exc:
        logger.error(
            "Failed to send ERROR upstream for request_id=%d: %s",
            request_id,
            exc,
        )


def _send_downstream(
    *,
    response_header: Header,
    output_list: list[float] | list[int],
    downstream_addr: str,
    request_id: int,
    connection_pool: ConnectionPool,
    coordinator_client: Optional[CoordinatorClient],
    node_id: Optional[str],
    session_key: Optional[bytes] = None,
) -> bool:
    """Attempt to send to downstream, with failure reporting and backup retry.

    On TCP send failure:
    1. Detect the failure (ConnectionError / OSError / socket.timeout).
    2. Close the broken connection in the pool.
    3. Call ReportFailure on the Coordinator to get a backup address.
    4. Retry the send to the backup node (single retry).

    Returns ``True`` if the send succeeded (to primary or backup),
    ``False`` if both attempts failed.
    """
    host, port_str = downstream_addr.rsplit(":", 1)
    downstream_tuple = (host, int(port_str))

    # ── Primary send attempt ────────────────────────────────────────────
    downstream_sock = connection_pool.get_connection(downstream_tuple)
    if downstream_sock is None:
        logger.error(
            "Failed to connect to downstream %s for request_id=%d",
            downstream_addr,
            request_id,
        )
        # Fall through to failure reporting below.
    else:
        try:
            write_message_secure(downstream_sock, response_header, output_list, session_key)
            logger.debug(
                "Forwarded request_id=%d to downstream %s",
                request_id,
                downstream_addr,
            )
            return True
        except (ConnectionError, OSError, socket.timeout) as exc:
            logger.warning(
                "TCP send to downstream %s failed for request_id=%d: %s",
                downstream_addr,
                request_id,
                exc,
            )
            # Close the broken connection so the pool doesn't reuse it.
            try:
                connection_pool.close_connection(downstream_tuple)
            except KeyError:
                pass

    # ── Report failure to Coordinator ───────────────────────────────────
    if coordinator_client is None:
        logger.error(
            "No Coordinator client available — cannot report failure "
            "for request_id=%d",
            request_id,
        )
        return False

    try:
        report_resp = coordinator_client.report_failure(
            ReportFailureRequest(
                request_id=request_id,
                failed_node_id=downstream_addr,
                reporting_node_id=node_id or "unknown",
            )
        )
    except Exception:
        logger.exception(
            "ReportFailure RPC failed for request_id=%d", request_id
        )
        return False

    if not report_resp.acknowledged:
        logger.error(
            "Coordinator did not acknowledge failure report for "
            "request_id=%d: %s",
            request_id,
            report_resp.message,
        )
        return False

    # ── Retry to backup node ────────────────────────────────────────────
    reroute = report_resp.reroute
    if reroute is None or reroute.backup_addr is None:
        logger.error(
            "No backup node available for request_id=%d (downstream %s)",
            request_id,
            downstream_addr,
        )
        return False

    backup_addr = reroute.backup_addr
    logger.info(
        "Rerouting request_id=%d to backup node %s",
        request_id,
        backup_addr,
    )

    backup_host, backup_port_str = backup_addr.rsplit(":", 1)
    backup_tuple = (backup_host, int(backup_port_str))

    backup_sock = connection_pool.get_connection(backup_tuple)
    if backup_sock is None:
        logger.error(
            "Failed to connect to backup node %s for request_id=%d",
            backup_addr,
            request_id,
        )
        return False

    try:
        write_message_secure(backup_sock, response_header, output_list, session_key)
        logger.info(
            "Successfully rerouted request_id=%d to backup %s",
            request_id,
            backup_addr,
        )
        return True
    except (ConnectionError, OSError, socket.timeout) as exc:
        logger.error(
            "Backup send to %s also failed for request_id=%d: %s",
            backup_addr,
            request_id,
            exc,
        )
        try:
            connection_pool.close_connection(backup_tuple)
        except KeyError:
            pass
        return False


# ── Per-Connection Handler ──────────────────────────────────────────────────


def _handle_connection(
    client_sock: socket.socket,
    client_addr: tuple[str, int],
    *,
    layer_engine: object,
    layer_registry: LayerAssignmentRegistry,
    connection_pool: ConnectionPool,
    resource_monitor: Optional[ResourceMonitor],
    coordinator_client: Optional[CoordinatorClient],
    stats: ServingStats,
    shutdown_event: threading.Event,
    device: str = "cuda:0",
    session_key: Optional[bytes] = None,
) -> None:
    """Process messages from a single upstream connection until it closes.

    Reads Forward messages in a loop, runs the forward pass, and sends
    results downstream.  The loop exits when the connection is closed by
    the peer, a shutdown is signalled, or an unrecoverable error occurs.

    When a downstream send fails, the handler:
    1. Detects the TCP failure.
    2. Reports the failure to the Coordinator via ``ReportFailure`` RPC.
    3. Receives ``RerouteInfo`` with a backup node address.
    4. Retries the send to the backup node (single retry).

    Parameters
    ----------
    client_sock:
        The accepted TCP socket from the upstream node or client.
    client_addr:
        Remote address as ``(host, port)``.
    layer_engine:
        A :class:`~meshrun.worker.layer_engine.LayerEngine` instance.
    layer_registry:
        The Layer Assignment Registry for routing decisions.
    connection_pool:
        The Connection Pool for outbound connections.
    resource_monitor:
        Optional Resource Monitor for tracking active requests.
    coordinator_client:
        Optional Coordinator client for failure reporting.
    stats:
        Shared serving statistics counters.
    shutdown_event:
        Event that is set when the node is shutting down.
    device:
        Torch device string for tensor reconstruction.
    session_key:
        Optional 32-byte AES-256 session key.  When provided, all
        reads use ``read_message_secure`` and all writes use
        ``write_message_secure``.  When ``None``, plaintext
        ``read_message`` / ``write_message`` are used.
    """
    # Import forward here to avoid circular imports at module level.
    from meshrun.worker.layer_engine import forward

    logger.info(
        "Serving connection from %s:%d", client_addr[0], client_addr[1]
    )

    is_final = layer_registry.is_final_node()
    downstream_addr = layer_registry.get_downstream_address()
    dtype_enum_val = layer_registry.get_dtype()
    node_id = layer_registry.get_node_id()
    # Default to FP16 if no assignment (shouldn't happen in practice).
    dtype_enum = int(dtype_enum_val) if dtype_enum_val is not None else int(DType.FP16)

    while not shutdown_event.is_set():
        try:
            # ── 1. Read incoming Forward message ────────────────────────
            if session_key is not None:
                header, tensor_data = read_message_secure(client_sock, session_key)
            else:
                header, tensor_data = read_message(client_sock)

            if header.message_type != int(MessageType.FORWARD):
                logger.warning(
                    "Ignoring non-FORWARD message (type=%d) from %s:%d",
                    header.message_type,
                    client_addr[0],
                    client_addr[1],
                )
                continue

            logger.debug(
                "Received FORWARD: request_id=%d, step_id=%d, "
                "payload_size=%d from %s:%d",
                header.request_id,
                header.step_id,
                header.payload_size,
                client_addr[0],
                client_addr[1],
            )

            # Track active request.
            if resource_monitor is not None:
                resource_monitor.increment_active_requests()

            try:
                # ── 2. Reconstruct input tensor ─────────────────────────
                active_dims = tuple(
                    header.dims[i] for i in range(header.num_dims)
                )
                hidden_states = _flat_list_to_tensor(
                    tensor_data, active_dims, header.dtype, device
                )

                # ── 3. Run forward pass ─────────────────────────────────
                output = forward(layer_engine, hidden_states, header.step_id)

                # ── 4. Build response and send ──────────────────────────
                response_header = _build_response_header(
                    header, output, dtype_enum, is_final,
                )
                output_list = _tensor_to_flat_list(output, dtype_enum)

                if is_final:
                    # Final node: send RESULT back on the same connection.
                    if session_key is not None:
                        write_message_secure(client_sock, response_header, output_list, session_key)
                    else:
                        write_message(client_sock, response_header, output_list)
                    logger.debug(
                        "Sent RESULT for request_id=%d back to client",
                        header.request_id,
                    )
                else:
                    # Intermediate node: forward to downstream with
                    # failure handling (report + reroute to backup).
                    if downstream_addr is None:
                        logger.error(
                            "No downstream address configured but node "
                            "is not final — dropping request_id=%d",
                            header.request_id,
                        )
                        stats.record_failure()
                        continue

                    sent = _send_downstream(
                        response_header=response_header,
                        output_list=output_list,
                        downstream_addr=downstream_addr,
                        request_id=header.request_id,
                        connection_pool=connection_pool,
                        coordinator_client=coordinator_client,
                        node_id=node_id,
                        session_key=session_key,
                    )
                    if not sent:
                        # Both primary and backup failed — send ERROR
                        # back upstream so the caller knows this request
                        # could not be completed.
                        _send_error_upstream(
                            client_sock,
                            header.request_id,
                            header.step_id,
                            session_key=session_key,
                        )
                        stats.record_failure()
                        continue

                stats.record_success()

            finally:
                # Always decrement active requests.
                if resource_monitor is not None:
                    resource_monitor.decrement_active_requests()

        except ConnectionError:
            logger.info(
                "Connection closed by %s:%d",
                client_addr[0],
                client_addr[1],
            )
            break
        except socket.timeout:
            # Timeout on recv — loop back and check shutdown flag.
            continue
        except Exception:
            logger.exception(
                "Error processing message from %s:%d",
                client_addr[0],
                client_addr[1],
            )
            stats.record_failure()
            break

    # Clean up the client socket.
    try:
        client_sock.close()
    except OSError:
        pass

    logger.info(
        "Finished serving connection from %s:%d", client_addr[0], client_addr[1]
    )


# ── Serving Loop ────────────────────────────────────────────────────────────


class ServingLoop:
    """Main serving loop that accepts TCP connections and processes Forward messages.

    Wires together the Connection Pool's TCP listener with the per-connection
    handler that reads messages, runs the Layer Engine, and forwards results.

    Usage::

        loop = ServingLoop(
            layer_engine=engine,
            layer_registry=registry,
            connection_pool=pool,
            resource_monitor=monitor,
            coordinator_client=coordinator,
            config=ServingConfig(listen_host="0.0.0.0", listen_port=9100),
        )
        loop.start()   # non-blocking — spawns listener thread
        ...
        loop.stop()     # graceful shutdown

    Parameters
    ----------
    layer_engine:
        A built :class:`~meshrun.worker.layer_engine.LayerEngine`.
    layer_registry:
        The Layer Assignment Registry for routing decisions.
    connection_pool:
        The Connection Pool for outbound and inbound connections.
    resource_monitor:
        Optional Resource Monitor for active request tracking.
    coordinator_client:
        Optional Coordinator client for downstream failure reporting.
    config:
        Serving configuration (listen address and port).
    device:
        Torch device string for tensor reconstruction.
    """

    def __init__(
        self,
        *,
        layer_engine: object,
        layer_registry: LayerAssignmentRegistry,
        connection_pool: ConnectionPool,
        resource_monitor: Optional[ResourceMonitor] = None,
        coordinator_client: Optional[CoordinatorClient] = None,
        config: Optional[ServingConfig] = None,
        device: str = "cuda:0",
        session_key: Optional[bytes] = None,
    ) -> None:
        self._engine = layer_engine
        self._registry = layer_registry
        self._pool = connection_pool
        self._monitor = resource_monitor
        self._coordinator_client = coordinator_client
        self._config = config or ServingConfig()
        self._device = device
        # Explicit session_key parameter takes precedence; fall back to config.
        self._session_key = session_key if session_key is not None else self._config.session_key

        self._shutdown_event = threading.Event()
        self._stats = ServingStats()
        self._started = False

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def stats(self) -> ServingStats:
        """Current serving statistics."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Whether the serving loop is currently accepting connections."""
        return self._started and not self._shutdown_event.is_set()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start accepting TCP connections and processing Forward messages.

        This is non-blocking — the TCP listener runs on a background
        daemon thread managed by the Connection Pool.  Each accepted
        connection is handled in its own daemon thread.

        Raises
        ------
        RuntimeError
            If the serving loop is already running, or if no layer
            assignment is stored in the registry.
        """
        if self._started:
            raise RuntimeError("ServingLoop is already running")

        if not self._registry.has_assignment:
            raise RuntimeError(
                "Cannot start serving — no layer assignment in registry"
            )

        self._shutdown_event.clear()

        listen_addr = (self._config.listen_host, self._config.listen_port)

        # Use the Connection Pool's accept_incoming with our handler as
        # the on_connection callback.  The pool spawns a daemon thread
        # per accepted connection.
        self._pool.accept_incoming(
            listen_addr=listen_addr,
            on_connection=self._on_connection,
        )

        self._started = True
        logger.info(
            "ServingLoop started on %s:%d",
            self._config.listen_host,
            self._config.listen_port,
        )

    def stop(self) -> None:
        """Signal the serving loop to stop accepting new connections.

        Sets the shutdown event so that in-flight connection handlers
        will exit after completing their current request.  Does not
        forcibly close active connections — they drain naturally.
        """
        if not self._started:
            return

        self._shutdown_event.set()
        self._started = False
        logger.info(
            "ServingLoop stopped (processed=%d, failed=%d)",
            self._stats.requests_processed,
            self._stats.requests_failed,
        )

    # ── Internal ─────────────────────────────────────────────────────────

    def _on_connection(
        self, client_sock: socket.socket, client_addr: tuple[str, int]
    ) -> None:
        """Callback invoked by the Connection Pool for each accepted connection.

        Delegates to :func:`_handle_connection` which runs the read →
        forward → write loop until the connection closes or shutdown is
        signalled.
        """
        _handle_connection(
            client_sock,
            client_addr,
            layer_engine=self._engine,
            layer_registry=self._registry,
            connection_pool=self._pool,
            resource_monitor=self._monitor,
            coordinator_client=self._coordinator_client,
            stats=self._stats,
            shutdown_event=self._shutdown_event,
            device=self._device,
            session_key=self._session_key,
        )

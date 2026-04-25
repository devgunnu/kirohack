"""Encrypted TCP transport for the MeshRun Inference Client.

Wraps the existing ``protocol.py`` secure read/write functions with
connection management, providing a clean interface for sending encrypted
FORWARD messages and receiving encrypted RESULT messages from the worker
pipeline.
"""

from __future__ import annotations

import logging
import socket

from cryptography.exceptions import InvalidTag

from meshrun.worker.protocol import (
    DType,
    Header,
    MessageType,
    read_message_secure,
    write_message_secure,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONNECT_TIMEOUT_S = 5.0


class SecureTransport:
    """Manages encrypted TCP connections to worker pipeline nodes.

    Handles connection lifecycle and delegates encryption/decryption to
    ``protocol.py``'s AES-256-GCM secure message functions.
    """

    def __init__(self, connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT_S) -> None:
        self._connect_timeout = connect_timeout
        self._sockets: list[socket.socket] = []

    def connect(self, node_addr: str) -> socket.socket:
        """Establish a TCP connection to a worker node.

        Args:
            node_addr: Worker address as ``"host:port"``.

        Returns:
            Connected TCP socket.

        Raises:
            ValueError: If *node_addr* is not in ``host:port`` format.
            ConnectionError: If the connection cannot be established.
        """
        parts = node_addr.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid node address '{node_addr}', expected 'host:port'"
            )
        host, port_str = parts
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(
                f"Invalid port in address '{node_addr}': '{port_str}' is not an integer"
            )

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout)
        try:
            sock.connect((host, port))
        except OSError as exc:
            sock.close()
            raise ConnectionError(
                f"Failed to connect to {node_addr}: {exc}"
            ) from exc

        self._sockets.append(sock)
        logger.debug("Connected to %s", node_addr)
        return sock

    def send_forward(
        self,
        sock: socket.socket,
        hidden_states: object,
        session_key: bytes,
        request_id: int,
        step_id: int,
    ) -> None:
        """Encrypt and send a FORWARD message to a worker node.

        Builds a protocol ``Header`` by inspecting the tensor's shape and
        dtype, flattens the tensor to a list, then delegates to
        ``write_message_secure``.

        Args:
            sock: Connected TCP socket (from :meth:`connect`).
            hidden_states: PyTorch tensor (e.g. ``[1, seq_len, hidden_dim]``).
            session_key: 32-byte AES-256 session key.
            request_id: Unique request identifier.
            step_id: Token generation step index.

        Raises:
            RuntimeError: If PyTorch is not available.
            ValueError: If the tensor has more than 4 dimensions or its
                dtype is not supported by the protocol.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover
            raise RuntimeError(
                "PyTorch is required for send_forward but is not installed"
            )

        from meshrun.worker.protocol import DTYPE_SIZE, MAX_DIMS

        tensor = hidden_states
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"hidden_states must be a torch.Tensor, got {type(tensor).__name__}"
            )

        # Map torch dtype to protocol DType
        _TORCH_DTYPE_MAP = {
            torch.float16: DType.FP16,
            torch.int8: DType.INT8,
        }
        proto_dtype = _TORCH_DTYPE_MAP.get(tensor.dtype)
        if proto_dtype is None:
            raise ValueError(
                f"Unsupported tensor dtype {tensor.dtype}; "
                f"expected one of {list(_TORCH_DTYPE_MAP.keys())}"
            )

        shape = tuple(tensor.shape)
        num_dims = len(shape)
        if num_dims < 1 or num_dims > MAX_DIMS:
            raise ValueError(
                f"num_dims must be in [1, {MAX_DIMS}], got {num_dims}"
            )

        # Pad dims to 4 elements
        dims = shape + (0,) * (MAX_DIMS - num_dims)

        num_elements = tensor.numel()
        element_size = DTYPE_SIZE[proto_dtype]
        payload_size = num_elements * element_size

        header = Header(
            message_type=MessageType.FORWARD,
            request_id=request_id,
            step_id=step_id,
            payload_size=payload_size,
            dtype=proto_dtype,
            num_dims=num_dims,
            dims=dims,
        )

        # Flatten tensor to a Python list for write_message_secure
        flat_data: list[float] | list[int]
        if proto_dtype == DType.INT8:
            flat_data = tensor.contiguous().view(-1).tolist()
        else:
            flat_data = tensor.contiguous().view(-1).tolist()

        write_message_secure(sock, header, flat_data, session_key)
        logger.debug(
            "Sent FORWARD request_id=%d step_id=%d shape=%s dtype=%s",
            request_id,
            step_id,
            shape,
            tensor.dtype,
        )

    def receive_result(
        self,
        sock: socket.socket,
        session_key: bytes,
    ) -> tuple[Header, list[float] | list[int]]:
        """Receive and decrypt a RESULT message from the pipeline.

        Args:
            sock: TCP socket connected to the pipeline (same socket used
                for :meth:`send_forward`).
            session_key: 32-byte AES-256 session key.

        Returns:
            ``(header, tensor_data)`` tuple with the decrypted result.

        Raises:
            RuntimeError: If decryption fails (wrong key or tampered data),
                if the received message is an ERROR, or if the message type
                is unexpected.
        """
        try:
            header, tensor_data = read_message_secure(sock, session_key)
        except InvalidTag:
            raise RuntimeError(
                "Decryption failed: invalid session key or tampered data"
            )

        if header.message_type == MessageType.ERROR:
            raise RuntimeError(
                f"Pipeline returned ERROR for request_id={header.request_id}: "
                f"step_id={header.step_id}"
            )

        if header.message_type != MessageType.RESULT:
            raise RuntimeError(
                f"Expected RESULT message, got message_type={header.message_type}"
            )

        logger.debug(
            "Received RESULT request_id=%d dims=%s",
            header.request_id,
            header.dims[: header.num_dims],
        )
        return header, tensor_data

    def close(self) -> None:
        """Close all open TCP sockets managed by this transport."""
        for sock in self._sockets:
            try:
                sock.close()
            except OSError:
                pass
        self._sockets.clear()
        logger.debug("All transport sockets closed")

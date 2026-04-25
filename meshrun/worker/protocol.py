"""
TCP Binary Protocol — Header definition for the MeshRun data plane.

Fixed 32-byte header layout:
    Offset  Field         Size     Type    Description
    ------  -----------   ------   ------  -----------
    0       message_type  1 byte   uint8   1=FORWARD, 2=RESULT, 3=ERROR, 4=HEARTBEAT_DATA
    1       request_id    4 bytes  uint32  Unique request identifier
    5       step_id       4 bytes  uint32  Token generation step
    9       payload_size  4 bytes  uint32  Exact payload size in bytes
    13      dtype         1 byte   uint8   1=fp16 (2 bytes/elem), 2=int8 (1 byte/elem)
    14      num_dims      1 byte   uint8   Number of tensor dimensions (1–4)
    15      dims[0]       4 bytes  uint32  Dimension 0
    19      dims[1]       4 bytes  uint32  Dimension 1
    23      dims[2]       4 bytes  uint32  Dimension 2
    27      dims[3]       4 bytes  uint32  Dimension 3
    31      reserved      1 byte   uint8   Padding / future use
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

# ── Constants ────────────────────────────────────────────────────────────────

HEADER_SIZE = 32
"""Every serialized header is exactly 32 bytes."""

HEADER_STRUCT_FORMAT = "<BIIIBB4IB"
"""
Little-endian struct format matching the header layout:
    B  = uint8   message_type   (1 byte)
    I  = uint32  request_id     (4 bytes)
    I  = uint32  step_id        (4 bytes)
    I  = uint32  payload_size   (4 bytes)
    B  = uint8   dtype          (1 byte)
    B  = uint8   num_dims       (1 byte)
    4I = uint32  dims[4]        (16 bytes)
    B  = uint8   reserved       (1 byte)
Total = 1 + 4 + 4 + 4 + 1 + 1 + 16 + 1 = 32 bytes
"""

MAX_DIMS = 4
"""Maximum number of tensor dimensions supported by the protocol."""


# ── Enumerations ─────────────────────────────────────────────────────────────

class MessageType(IntEnum):
    """Message types carried over the data-plane TCP protocol."""
    FORWARD = 1
    RESULT = 2
    ERROR = 3
    HEARTBEAT_DATA = 4


class DType(IntEnum):
    """Tensor data types supported by the protocol."""
    FP16 = 1  # 2 bytes per element (IEEE 754 half-precision, little-endian)
    INT8 = 2  # 1 byte per element  (signed 8-bit integer)


DTYPE_SIZE = {
    DType.FP16: 2,
    DType.INT8: 1,
}
"""Mapping from DType enum value to the byte size of a single element."""


# ── Header Data Structure ────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Header:
    """Fixed 32-byte binary header for the MeshRun TCP protocol.

    All integer fields use little-endian byte order.
    """

    message_type: int
    """Message type (uint8). Must be a valid ``MessageType`` value."""

    request_id: int
    """Unique request identifier (uint32)."""

    step_id: int
    """Token generation step index (uint32). 0 for the first token."""

    payload_size: int
    """Exact size of the tensor payload in bytes (uint32)."""

    dtype: int
    """Tensor element type (uint8). Must be a valid ``DType`` value."""

    num_dims: int
    """Number of active tensor dimensions (uint8), in range [1, 4]."""

    dims: Tuple[int, int, int, int] = (0, 0, 0, 0)
    """Tensor dimensions as four uint32 values.

    Active dimensions (index < num_dims) must be > 0.
    Unused dimensions (index >= num_dims) must be 0.
    """

    reserved: int = 0
    """Reserved byte for future use (uint8). Should be 0."""

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> None:
        """Validate all header fields per the protocol specification.

        Raises:
            ValueError: If any field violates the protocol constraints.
        """
        # 1. message_type ∈ {1, 2, 3, 4}
        valid_message_types = {int(mt) for mt in MessageType}
        if self.message_type not in valid_message_types:
            raise ValueError(
                f"Invalid message_type {self.message_type}, "
                f"expected one of {sorted(valid_message_types)}"
            )

        # 2. dtype ∈ {1, 2}
        valid_dtypes = {int(dt) for dt in DType}
        if self.dtype not in valid_dtypes:
            raise ValueError(
                f"Invalid dtype {self.dtype}, "
                f"expected one of {sorted(valid_dtypes)}"
            )

        # 3. num_dims ∈ {1, 2, 3, 4}
        if self.num_dims < 1 or self.num_dims > MAX_DIMS:
            raise ValueError(
                f"Invalid num_dims {self.num_dims}, expected 1–{MAX_DIMS}"
            )

        # 4. dims consistency: dims[i] > 0 for i < num_dims,
        #    dims[i] == 0 for i >= num_dims
        for i in range(self.num_dims):
            if self.dims[i] <= 0:
                raise ValueError(
                    f"dims[{i}] must be > 0 for active dimension, "
                    f"got {self.dims[i]}"
                )
        for i in range(self.num_dims, MAX_DIMS):
            if self.dims[i] != 0:
                raise ValueError(
                    f"dims[{i}] must be 0 for unused dimension, "
                    f"got {self.dims[i]}"
                )

        # 5. payload_size == product(active dims) * dtype_size
        product = 1
        for i in range(self.num_dims):
            product *= self.dims[i]
        expected_payload = product * DTYPE_SIZE[DType(self.dtype)]
        if self.payload_size != expected_payload:
            raise ValueError(
                f"payload_size {self.payload_size} does not match "
                f"expected {expected_payload} "
                f"(product(dims) * dtype_size)"
            )

    # ── Serialization ────────────────────────────────────────────────────

    def pack(self) -> bytes:
        """Serialize this header into exactly 32 bytes (little-endian).

        Returns:
            A ``bytes`` object of length :data:`HEADER_SIZE` (32).
        """
        data = struct.pack(
            HEADER_STRUCT_FORMAT,
            self.message_type,
            self.request_id,
            self.step_id,
            self.payload_size,
            self.dtype,
            self.num_dims,
            self.dims[0],
            self.dims[1],
            self.dims[2],
            self.dims[3],
            self.reserved,
        )
        assert len(data) == HEADER_SIZE, (
            f"Header pack produced {len(data)} bytes, expected {HEADER_SIZE}"
        )
        return data

    # ── Deserialization ──────────────────────────────────────────────────

    @classmethod
    def unpack(cls, data: bytes | bytearray) -> Header:
        """Deserialize exactly 32 bytes into a :class:`Header` instance.

        This is the inverse of :meth:`pack` — for any valid header *h*,
        ``Header.unpack(h.pack()) == h``.

        Args:
            data: A ``bytes`` or ``bytearray`` of exactly :data:`HEADER_SIZE`
                (32) bytes.

        Returns:
            A new :class:`Header` with all fields extracted from *data*.

        Raises:
            ValueError: If *data* is not exactly 32 bytes.
        """
        if len(data) != HEADER_SIZE:
            raise ValueError(
                f"Expected {HEADER_SIZE} bytes, got {len(data)}"
            )

        (
            message_type,
            request_id,
            step_id,
            payload_size,
            dtype,
            num_dims,
            d0, d1, d2, d3,
            reserved,
        ) = struct.unpack(HEADER_STRUCT_FORMAT, data)

        header = cls(
            message_type=message_type,
            request_id=request_id,
            step_id=step_id,
            payload_size=payload_size,
            dtype=dtype,
            num_dims=num_dims,
            dims=(d0, d1, d2, d3),
            reserved=reserved,
        )
        header.validate()
        return header


# ── Reliable TCP Framing ─────────────────────────────────────────────────────


def read_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from *sock*, looping on ``recv()`` as needed.

    TCP is a stream protocol — a single ``recv()`` call may return fewer
    bytes than requested.  This function accumulates data in a buffer
    until exactly *n* bytes have been collected.

    Args:
        sock: A socket (or any object with a ``recv(int)`` method).
        n: The exact number of bytes to read.  Must be ≥ 0.

    Returns:
        A ``bytes`` object of length *n*.

    Raises:
        ConnectionError: If the remote end closes the connection (EOF)
            before *n* bytes have been received.
        socket.timeout: If a ``recv()`` call times out.
    """
    if n == 0:
        return b""

    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            raise
        if not chunk:
            raise ConnectionError(
                f"Connection closed: received {len(buf)} of {n} expected bytes"
            )
        buf.extend(chunk)
    return bytes(buf)


def tensor_to_bytes(
    elements: list[float] | list[int],
    dtype: int,
) -> bytes:
    """Serialize a flat list of numeric values to raw contiguous bytes.

    Elements are written in row-major (C-contiguous) order.  The caller is
    responsible for flattening a multi-dimensional tensor before calling
    this function.

    Args:
        elements: Flat sequence of numeric values (row-major order).
        dtype: One of :attr:`DType.FP16` or :attr:`DType.INT8`.

    Returns:
        Raw ``bytes`` with length ``len(elements) * DTYPE_SIZE[dtype]``.

    Raises:
        ValueError: If *dtype* is not a recognised :class:`DType` value, or
            if any element is out of range for the target dtype.
    """
    if dtype == DType.FP16:
        # IEEE 754 half-precision, little-endian ('e' = half float)
        fmt = f"<{len(elements)}e"
        try:
            return struct.pack(fmt, *elements)
        except (struct.error, OverflowError) as e:
            raise ValueError(
                f"Cannot serialize elements to fp16: {e}"
            ) from e
    elif dtype == DType.INT8:
        # Signed 8-bit integer ('b' = signed char)
        # Convert float values to int (truncate toward zero)
        int_elements = []
        for val in elements:
            if isinstance(val, float):
                # Truncate toward zero like int() does
                int_val = int(val)
                if int_val < -128 or int_val > 127:
                    raise ValueError(
                        f"Value {val} truncated to {int_val} is out of int8 range [-128, 127]"
                    )
                int_elements.append(int_val)
            else:
                if val < -128 or val > 127:
                    raise ValueError(
                        f"Value {val} is out of int8 range [-128, 127]"
                    )
                int_elements.append(val)
        
        fmt = f"<{len(int_elements)}b"
        return struct.pack(fmt, *int_elements)
    else:
        valid = {int(dt) for dt in DType}
        raise ValueError(
            f"Invalid dtype {dtype}, expected one of {sorted(valid)}"
        )


def write_all(sock: socket.socket, data: bytes | bytearray) -> None:
    """Write all of *data* to *sock*, looping on ``send()`` as needed.

    TCP is a stream protocol — a single ``send()`` call may accept fewer
    bytes than provided.  This function loops until every byte has been
    transmitted.

    Args:
        sock: A socket (or any object with a ``send(bytes)`` method).
        data: The bytes to write.  May be empty.

    Raises:
        ConnectionError: If ``send()`` returns 0 (connection broken).
        socket.timeout: If a ``send()`` call times out.
    """
    if not data:
        return

    total_sent = 0
    while total_sent < len(data):
        try:
            sent = sock.send(data[total_sent:])
        except socket.timeout:
            raise
        if sent == 0:
            raise ConnectionError(
                f"Connection closed: sent {total_sent} of {len(data)} bytes"
            )
        total_sent += sent

def bytes_to_tensor(
    data: bytes | bytearray,
    dtype: int,
    dims: tuple[int, int, int, int],
    num_dims: int,
) -> list[float] | list[int]:
    """Deserialize raw contiguous bytes to a flat list of numeric values.

    This is the inverse of :func:`tensor_to_bytes`.  The bytes must be in
    row-major (C-contiguous) order, matching the layout produced by
    ``tensor_to_bytes``.

    Args:
        data: Raw bytes containing the serialized tensor.
        dtype: One of :attr:`DType.FP16` or :attr:`DType.INT8`.
        dims: Four-element tuple of tensor dimensions (as stored in header).
        num_dims: Number of active dimensions (1–4).

    Returns:
        Flat list of numeric values (row-major order).  For fp16 dtype,
        values are Python floats (converted from half-precision).
        For int8 dtype, values are Python integers.

    Raises:
        ValueError: If *dtype* is not a recognised :class:`DType` value,
            if *data* length does not match expected size based on *dims*
            and *dtype*, or if *num_dims* is not in range [1, 4].
    """
    # Validate dtype
    if dtype not in {int(dt) for dt in DType}:
        valid = {int(dt) for dt in DType}
        raise ValueError(
            f"Invalid dtype {dtype}, expected one of {sorted(valid)}"
        )

    # Validate num_dims
    if num_dims < 1 or num_dims > MAX_DIMS:
        raise ValueError(
            f"Invalid num_dims {num_dims}, expected 1–{MAX_DIMS}"
        )

    # Calculate expected data size
    product = 1
    for i in range(num_dims):
        if dims[i] < 0:
            raise ValueError(
                f"dims[{i}] must be >= 0, got {dims[i]}"
            )
        product *= dims[i]
    
    # Check unused dimensions are zero
    for i in range(num_dims, MAX_DIMS):
        if dims[i] != 0:
            raise ValueError(
                f"dims[{i}] must be 0 for unused dimension, got {dims[i]}"
            )

    expected_size = product * DTYPE_SIZE[DType(dtype)]
    if len(data) != expected_size:
        raise ValueError(
            f"Data length {len(data)} does not match expected size "
            f"{expected_size} (product(dims) * dtype_size)"
        )

    # Deserialize based on dtype
    if dtype == DType.FP16:
        # IEEE 754 half-precision, little-endian ('e' = half float)
        fmt = f"<{product}e"
        try:
            return list(struct.unpack(fmt, data))
        except struct.error as e:
            raise ValueError(f"Cannot deserialize bytes to fp16: {e}") from e
    elif dtype == DType.INT8:
        # Signed 8-bit integer ('b' = signed char)
        fmt = f"<{product}b"
        try:
            return list(struct.unpack(fmt, data))
        except struct.error as e:
            raise ValueError(f"Cannot deserialize bytes to int8: {e}") from e
    else:
        # This should never happen due to earlier validation
        valid = {int(dt) for dt in DType}
        raise ValueError(
            f"Invalid dtype {dtype}, expected one of {sorted(valid)}"
        )
def read_message(sock: socket.socket) -> tuple[Header, list[float] | list[int]]:
    """Read a complete message from a TCP socket.
    
    The read flow is:
    1. read_exact(32) to read the header bytes
    2. Validate the header using Header.unpack() which calls validate()
    3. read_exact(payload_size) to read the payload bytes
    4. Reconstruct tensor using bytes_to_tensor() with dtype and dims from the header
    
    Args:
        sock: A socket (or any object with a ``recv(int)`` method).
        
    Returns:
        A tuple (header, tensor_data) where:
        - header is the parsed Header instance
        - tensor_data is the reconstructed tensor as a flat list of numeric values
          (floats for fp16, integers for int8)
    
    Raises:
        ConnectionError: If the connection is closed before reading all bytes.
        socket.timeout: If a ``recv()`` call times out.
        ValueError: If the header is invalid (invalid fields, payload_size mismatch, etc.)
    """
    # 1. Read exactly 32 bytes for the header
    header_bytes = read_exact(sock, HEADER_SIZE)
    
    # 2. Unpack and validate the header (Header.unpack() calls validate())
    header = Header.unpack(header_bytes)
    
    # 3. Read exactly payload_size bytes for the payload
    payload_bytes = read_exact(sock, header.payload_size)
    
    # 4. Reconstruct tensor from raw bytes using dtype and dims from the header
    tensor_data = bytes_to_tensor(
        payload_bytes,
        header.dtype,
        header.dims,
        header.num_dims
    )
    
    return header, tensor_data
def write_message(
    sock: socket.socket,
    header: Header,
    tensor_data: list[float] | list[int],
) -> None:
    """Write a complete message (header + tensor payload) to a TCP socket.
    
    The write flow is:
    1. Validate tensor_data length matches header.dims
    2. Serialize tensor_data to bytes using tensor_to_bytes() with dtype from header
    3. Ensure header.payload_size matches serialized tensor byte count
    4. Pack header using header.pack()
    5. Write header + payload as single contiguous write using write_all()
    
    This function is the inverse of read_message() — for any valid header and
    tensor_data, read_message(sock) after write_message(sock, header, tensor_data)
    should reconstruct the same header and tensor_data.
    
    Args:
        sock: A socket (or any object with a ``send(bytes)`` method).
        header: Header instance containing message metadata and tensor shape/dtype.
        tensor_data: Flat list of numeric values (row-major order) representing
            the tensor. Must match the shape specified in header.dims.
    
    Returns:
        None
    
    Raises:
        ValueError: If tensor_data length does not match expected size based on
            header.dims, or if header validation fails.
        ConnectionError: If the connection is broken during write.
        socket.timeout: If a ``send()`` call times out.
    """
    # 1. Validate tensor_data length matches header.dims
    # Calculate expected number of elements from dims
    expected_elements = 1
    for i in range(header.num_dims):
        expected_elements *= header.dims[i]
    
    if len(tensor_data) != expected_elements:
        raise ValueError(
            f"Tensor data length {len(tensor_data)} does not match "
            f"expected {expected_elements} from dims {header.dims[:header.num_dims]}"
        )
    
    # 2. Serialize tensor_data to bytes using tensor_to_bytes() with dtype from header
    payload_bytes = tensor_to_bytes(tensor_data, header.dtype)
    
    # 3. Ensure header.payload_size matches serialized tensor byte count
    if header.payload_size != len(payload_bytes):
        raise ValueError(
            f"Header payload_size {header.payload_size} does not match "
            f"actual payload size {len(payload_bytes)}"
        )
    
    # 4. Pack header using header.pack()
    header_bytes = header.pack()
    
    # 5. Write header + payload as single contiguous write using write_all()
    # Concatenate header and payload for single write operation
    message_bytes = header_bytes + payload_bytes
    write_all(sock, message_bytes)
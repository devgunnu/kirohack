---
name: "tcp-binary-protocol"
displayName: "TCP Binary Protocol & Networking"
description: "Expert guidance for building custom TCP binary protocols in Python. Covers struct-based serialization, reliable framing, connection management, socket programming patterns, and high-performance data plane design."
keywords: ["tcp", "socket", "binary-protocol", "struct", "networking"]
author: "MeshRun Team"
---

# TCP Binary Protocol & Networking

## Overview

This power provides expert-level guidance for building custom TCP binary protocols in Python. It covers the full stack of low-level networking: socket programming, binary serialization with `struct`, reliable framing over TCP streams, connection pooling, and performance patterns for high-throughput data planes.

Key capabilities covered:
- Designing fixed-size binary headers with `struct.pack` / `struct.unpack`
- Reliable TCP framing (handling partial reads/writes)
- Binary tensor/data serialization in row-major order
- Connection lifecycle management and pooling
- Byte order (endianness) and alignment
- Error handling patterns for production TCP services
- Performance optimization for data-intensive protocols

## Onboarding

### Prerequisites

- Python 3.10+ (for `match` statements, union types with `|`)
- No external dependencies — uses only Python stdlib: `socket`, `struct`, `enum`, `dataclasses`

### Key Modules

```python
import socket       # TCP/UDP socket API
import struct       # Binary packing/unpacking
from enum import IntEnum
from dataclasses import dataclass
```

### Verification

```python
import struct
import socket

# Verify struct works with your format
data = struct.pack("<BII", 1, 42, 1024)
msg_type, req_id, size = struct.unpack("<BII", data)
assert (msg_type, req_id, size) == (1, 42, 1024)

# Verify socket module
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.close()
print("TCP + struct working correctly")
```

## Core Concepts

### TCP is a Stream Protocol, Not a Message Protocol

TCP delivers a continuous byte stream with no message boundaries. A single `recv()` call may return:
- Fewer bytes than requested (partial read)
- Data from multiple logical messages concatenated together
- Any arbitrary split of the byte stream

**This means you must always:**
1. Loop on `recv()` until you have exactly N bytes
2. Define your own message framing (header with length field)
3. Never assume one `send()` = one `recv()`

### Binary Protocol Design Principles

1. **Fixed-size headers** — Parse the header first to learn the payload size
2. **Little-endian byte order** — Use `<` prefix in struct format strings (matches x86/ARM)
3. **Explicit field sizes** — Every field has a defined byte width and type
4. **Validate before processing** — Check all header fields before reading the payload
5. **No text serialization in the data plane** — No JSON, Protobuf, or MessagePack for hot-path data

## struct Format String Reference

The `struct` module uses format characters to define binary layouts:

### Byte Order Prefixes

| Prefix | Byte Order                 | Alignment                       |
| ------ | -------------------------- | ------------------------------- |
| `<`    | Little-endian              | No padding                      |
| `>`    | Big-endian (network order) | No padding                      |
| `!`    | Network (= big-endian)     | No padding                      |
| `=`    | Native                     | No padding                      |
| `@`    | Native                     | Native alignment (with padding) |

**Recommendation:** Use `<` (little-endian, no padding) for custom protocols on modern hardware.

### Format Characters

| Char | C Type                | Python Type | Size (bytes) |
| ---- | --------------------- | ----------- | ------------ |
| `B`  | `unsigned char`       | int         | 1            |
| `b`  | `signed char`         | int         | 1            |
| `H`  | `unsigned short`      | int         | 2            |
| `h`  | `signed short`        | int         | 2            |
| `I`  | `unsigned int`        | int         | 4            |
| `i`  | `signed int`          | int         | 4            |
| `Q`  | `unsigned long long`  | int         | 8            |
| `q`  | `signed long long`    | int         | 8            |
| `e`  | half float (IEEE 754) | float       | 2            |
| `f`  | float                 | float       | 4            |
| `d`  | double                | float       | 8            |
| `?`  | `_Bool`               | bool        | 1            |
| `s`  | `char[]`              | bytes       | 1 each       |
| `x`  | pad byte              | —           | 1            |

**Repeat counts:** `4I` means four `uint32` values (16 bytes total). `32s` means 32-byte string.

### Calculating Total Size

```python
import struct

fmt = "<BIIIBB4IB"
size = struct.calcsize(fmt)  # 32
```

## Common Workflows

### Workflow 1: Design a Fixed-Size Binary Header

A well-designed header tells the receiver exactly how to interpret the payload.

```python
from __future__ import annotations
import struct
from dataclasses import dataclass
from enum import IntEnum

HEADER_SIZE = 32
HEADER_FORMAT = "<BIIIBB4IB"

class MessageType(IntEnum):
    FORWARD = 1
    RESULT = 2
    ERROR = 3
    HEARTBEAT = 4

class DType(IntEnum):
    FP16 = 1   # 2 bytes per element
    INT8 = 2   # 1 byte per element

DTYPE_SIZE = {DType.FP16: 2, DType.INT8: 1}

@dataclass(frozen=True, slots=True)
class Header:
    message_type: int
    request_id: int
    step_id: int
    payload_size: int
    dtype: int
    num_dims: int
    dims: tuple[int, int, int, int] = (0, 0, 0, 0)
    reserved: int = 0

    def pack(self) -> bytes:
        data = struct.pack(
            HEADER_FORMAT,
            self.message_type, self.request_id, self.step_id,
            self.payload_size, self.dtype, self.num_dims,
            *self.dims, self.reserved,
        )
        assert len(data) == HEADER_SIZE
        return data

    @classmethod
    def unpack(cls, data: bytes | bytearray) -> Header:
        if len(data) != HEADER_SIZE:
            raise ValueError(f"Expected {HEADER_SIZE} bytes, got {len(data)}")
        (msg_type, req_id, step_id, payload_size,
         dtype, num_dims, d0, d1, d2, d3, reserved
        ) = struct.unpack(HEADER_FORMAT, data)
        return cls(
            message_type=msg_type, request_id=req_id,
            step_id=step_id, payload_size=payload_size,
            dtype=dtype, num_dims=num_dims,
            dims=(d0, d1, d2, d3), reserved=reserved,
        )
```

**Design rules:**
- Use `dataclass(frozen=True, slots=True)` for immutable value types
- Use `IntEnum` for protocol constants (type-safe, debuggable)
- Always assert packed size matches expected constant
- Validate after unpacking, before processing

### Workflow 2: Reliable TCP Read and Write

Never assume `recv()` or `send()` handles all bytes in one call.

```python
import socket

def read_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes, looping on recv() as needed."""
    if n == 0:
        return b""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Connection closed: received {len(buf)} of {n} bytes"
            )
        buf.extend(chunk)
    return bytes(buf)

def write_all(sock: socket.socket, data: bytes | bytearray) -> None:
    """Write all bytes, looping on send() as needed."""
    if not data:
        return
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:])
        if sent == 0:
            raise ConnectionError(
                f"Connection closed: sent {total_sent} of {len(data)} bytes"
            )
        total_sent += sent
```

**Why not `sendall()`?** Python's `socket.sendall()` works for most cases, but a manual loop gives you control over timeout handling and progress tracking. For `recv()`, there is no built-in equivalent — you must always loop.

### Workflow 3: Complete Message Read/Write

Combine header parsing with payload reading for a complete framed protocol.

```python
def read_message(sock: socket.socket) -> tuple[Header, bytes]:
    """Read a complete framed message: header + payload."""
    # 1. Read fixed-size header
    header_bytes = read_exact(sock, HEADER_SIZE)

    # 2. Parse and validate header
    header = Header.unpack(header_bytes)
    header.validate()  # Check field constraints

    # 3. Read exactly payload_size bytes
    payload = read_exact(sock, header.payload_size)

    return header, payload

def write_message(sock: socket.socket, header: Header, payload: bytes) -> None:
    """Write a complete framed message as a single contiguous write."""
    header_bytes = header.pack()
    # Single write avoids Nagle's algorithm delays
    write_all(sock, header_bytes + payload)
```

**Key pattern:** Always write header + payload in a single `write_all()` call. This avoids TCP sending the header as a separate small packet (Nagle's algorithm interaction).

### Workflow 4: Tensor Serialization Over TCP

Serialize numeric arrays to raw bytes for wire transmission.

```python
import struct

def tensor_to_bytes(elements: list[float] | list[int], dtype: int) -> bytes:
    """Serialize a flat list of values to raw contiguous bytes."""
    if dtype == DType.FP16:
        fmt = f"<{len(elements)}e"   # half-precision floats
        return struct.pack(fmt, *elements)
    elif dtype == DType.INT8:
        fmt = f"<{len(elements)}b"   # signed 8-bit integers
        return struct.pack(fmt, *elements)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def bytes_to_tensor(
    data: bytes, dtype: int, num_elements: int
) -> list[float] | list[int]:
    """Deserialize raw bytes back to a flat list of values."""
    if dtype == DType.FP16:
        fmt = f"<{num_elements}e"
        return list(struct.unpack(fmt, data))
    elif dtype == DType.INT8:
        fmt = f"<{num_elements}b"
        return list(struct.unpack(fmt, data))
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
```

**Row-major (C-contiguous) order:** Multi-dimensional tensors are flattened so the last dimension varies fastest. A 2×3 tensor `[[1,2,3],[4,5,6]]` becomes `[1,2,3,4,5,6]` in memory.

### Workflow 5: TCP Server and Client Setup

```python
import socket

# ── Server ──
def start_server(host: str, port: int) -> socket.socket:
    """Create and bind a TCP server socket."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    return server

def accept_connection(server: socket.socket) -> socket.socket:
    """Accept a single client connection."""
    conn, addr = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return conn

# ── Client ──
def connect(host: str, port: int, timeout: float = 10.0) -> socket.socket:
    """Connect to a TCP server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect((host, port))
    return sock
```

**`TCP_NODELAY`:** Disables Nagle's algorithm, which buffers small writes. For binary protocols sending header+payload together, this reduces latency. Always enable it for data plane connections.

### Workflow 6: Connection Pool Pattern

Reuse persistent TCP connections across requests.

```python
import socket
import threading
from collections import defaultdict

class ConnectionPool:
    """Thread-safe pool of persistent TCP connections keyed by (host, port)."""

    def __init__(self, max_per_host: int = 2):
        self._max = max_per_host
        self._pools: dict[tuple[str, int], list[socket.socket]] = defaultdict(list)
        self._lock = threading.Lock()

    def get(self, host: str, port: int) -> socket.socket:
        key = (host, port)
        with self._lock:
            pool = self._pools[key]
            if pool:
                return pool.pop()
        # No idle connection — create new one
        return connect(host, port)

    def put(self, host: str, port: int, sock: socket.socket) -> None:
        key = (host, port)
        with self._lock:
            pool = self._pools[key]
            if len(pool) < self._max:
                pool.append(sock)
                return
        # Pool full — close the connection
        sock.close()

    def close_all(self) -> None:
        with self._lock:
            for pool in self._pools.values():
                for sock in pool:
                    sock.close()
            self._pools.clear()
```

### Workflow 7: Header Validation Pattern

Always validate headers before reading payloads to prevent resource exhaustion attacks.

```python
MAX_DIMS = 4

def validate_header(header: Header) -> None:
    """Validate all header fields per protocol spec."""
    # 1. message_type must be a known value
    valid_types = {int(mt) for mt in MessageType}
    if header.message_type not in valid_types:
        raise ValueError(f"Invalid message_type: {header.message_type}")

    # 2. dtype must be a known value
    valid_dtypes = {int(dt) for dt in DType}
    if header.dtype not in valid_dtypes:
        raise ValueError(f"Invalid dtype: {header.dtype}")

    # 3. num_dims in [1, MAX_DIMS]
    if not (1 <= header.num_dims <= MAX_DIMS):
        raise ValueError(f"Invalid num_dims: {header.num_dims}")

    # 4. Active dims > 0, unused dims == 0
    for i in range(header.num_dims):
        if header.dims[i] <= 0:
            raise ValueError(f"dims[{i}] must be > 0, got {header.dims[i]}")
    for i in range(header.num_dims, MAX_DIMS):
        if header.dims[i] != 0:
            raise ValueError(f"dims[{i}] must be 0, got {header.dims[i]}")

    # 5. payload_size must match product(active_dims) * dtype_size
    product = 1
    for i in range(header.num_dims):
        product *= header.dims[i]
    expected = product * DTYPE_SIZE[DType(header.dtype)]
    if header.payload_size != expected:
        raise ValueError(
            f"payload_size {header.payload_size} != expected {expected}"
        )
```

**Why validate before reading payload?** A malicious or buggy sender could claim `payload_size = 4GB`. Validating that `payload_size` matches `product(dims) * dtype_size` catches this before you allocate memory.

## Socket Options Reference

| Option         | Level         | Purpose                                    | Typical Value             |
| -------------- | ------------- | ------------------------------------------ | ------------------------- |
| `SO_REUSEADDR` | `SOL_SOCKET`  | Allow rebinding to port after restart      | `1`                       |
| `TCP_NODELAY`  | `IPPROTO_TCP` | Disable Nagle's algorithm (reduce latency) | `1`                       |
| `SO_KEEPALIVE` | `SOL_SOCKET`  | Detect dead connections                    | `1`                       |
| `SO_RCVBUF`    | `SOL_SOCKET`  | Receive buffer size                        | OS default or larger      |
| `SO_SNDBUF`    | `SOL_SOCKET`  | Send buffer size                           | OS default or larger      |
| `SO_LINGER`    | `SOL_SOCKET`  | Control close behavior                     | `struct.pack('ii', 1, 0)` |

## Troubleshooting

### Error: "Connection closed: received X of Y expected bytes"

**Cause:** The remote end closed the connection mid-message.
**Solutions:**
1. Check if the remote process crashed or timed out
2. Implement reconnection logic with exponential backoff
3. Add heartbeat messages to detect dead connections early

### Error: struct.error "unpack requires a buffer of N bytes"

**Cause:** The data buffer doesn't match the format string's expected size.
**Solutions:**
```python
# Always verify data length before unpacking
expected = struct.calcsize(HEADER_FORMAT)
if len(data) != expected:
    raise ValueError(f"Expected {expected} bytes, got {len(data)}")
```

### Error: "Address already in use" (EADDRINUSE)

**Cause:** Previous server process left the socket in TIME_WAIT state.
**Solution:**
```python
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
```

### Partial Writes / Sends Hanging

**Cause:** Send buffer full, receiver not consuming data fast enough.
**Solutions:**
1. Increase `SO_SNDBUF` / `SO_RCVBUF`
2. Set socket timeouts: `sock.settimeout(10.0)`
3. Check for deadlocks (both sides trying to send without reading)

### Half-Precision Float Precision Loss

**Cause:** `struct.pack('<e', value)` loses precision for values outside fp16 range.
**Solution:**
```python
import struct
# fp16 range: ±65504, min subnormal: ~5.96e-8
# Check before packing:
if abs(value) > 65504:
    raise OverflowError(f"Value {value} exceeds fp16 range")
```

### Byte Order Mismatch

**Cause:** Sender uses little-endian, receiver uses big-endian (or vice versa).
**Solution:** Always use the same prefix in format strings on both sides. Convention: `<` (little-endian) for modern x86/ARM systems.

## Best Practices

- Always use `read_exact()` loops for TCP reads — never assume one `recv()` returns all requested bytes
- Always use `write_all()` loops for TCP writes — `send()` may accept fewer bytes than provided
- Validate headers before reading payloads to prevent memory exhaustion from malformed messages
- Use `dataclass(frozen=True, slots=True)` for protocol header types — immutable, memory-efficient
- Use `IntEnum` for protocol constants — type-safe, prints readable names in debug output
- Enable `TCP_NODELAY` on data plane connections to minimize latency
- Write header + payload in a single `write_all()` call to avoid small-packet overhead
- Use little-endian (`<`) byte order consistently — matches x86/ARM native order
- Keep the header fixed-size so you always know exactly how many bytes to read first
- Add a `reserved` byte or field for future protocol extensions without breaking compatibility
- Use `struct.calcsize()` to verify your format string matches the expected header size
- For connection pools, use thread-safe access with locks and limit connections per host

## struct Format Quick Reference

```python
import struct

# Pack values into bytes
data = struct.pack("<BII", 1, 42, 1024)

# Unpack bytes into values
msg_type, req_id, size = struct.unpack("<BII", data)

# Calculate size of a format
size = struct.calcsize("<BII")  # 9

# Pack multiple same-type values
floats = struct.pack("<4e", 1.0, 2.0, 3.0, 4.0)  # 4 half-floats

# Unpack into tuple
values = struct.unpack("<4e", floats)  # (1.0, 2.0, 3.0, 4.0)
```

---

**Stdlib Modules:** `socket`, `struct`, `enum`, `dataclasses`
**Python Docs:** [socket](https://docs.python.org/3/library/socket.html) · [struct](https://docs.python.org/3/library/struct.html)

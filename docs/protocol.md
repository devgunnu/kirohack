# TCP Binary Protocol

The data plane uses a custom binary protocol optimized for minimal overhead. No serialization libraries (JSON, Protobuf, MessagePack) are used. Tensors are transmitted as raw contiguous memory with a fixed-size header for framing.

Source: `meshrun/worker/protocol.py`

## Design Principles

- Fixed 32-byte header for predictable parsing
- Zero-copy friendly: payload is raw tensor bytes matching GPU memory layout
- No compression: quantization (fp16/int8) is the size reduction strategy
- Framing via `payload_size` field: receiver always knows exactly how many bytes to read

## Header Layout (32 bytes, little-endian)

| Offset | Field        | Size    | Type   | Description                                    |
| ------ | ------------ | ------- | ------ | ---------------------------------------------- |
| 0      | message_type | 1 byte  | uint8  | 1=FORWARD, 2=RESULT, 3=ERROR, 4=HEARTBEAT_DATA |
| 1      | request_id   | 4 bytes | uint32 | Unique request identifier                      |
| 5      | step_id      | 4 bytes | uint32 | Token generation step (0 for first token)      |
| 9      | payload_size | 4 bytes | uint32 | Exact payload size in bytes                    |
| 13     | dtype        | 1 byte  | uint8  | 1=fp16 (2 bytes/elem), 2=int8 (1 byte/elem)    |
| 14     | num_dims     | 1 byte  | uint8  | Number of tensor dimensions (1-4)              |
| 15     | dims[0]      | 4 bytes | uint32 | Dimension 0 (e.g., batch_size)                 |
| 19     | dims[1]      | 4 bytes | uint32 | Dimension 1 (e.g., seq_len)                    |
| 23     | dims[2]      | 4 bytes | uint32 | Dimension 2 (e.g., hidden_dim)                 |
| 27     | dims[3]      | 4 bytes | uint32 | Dimension 3 (unused → 0)                       |
| 31     | reserved     | 1 byte  | uint8  | Padding / future use                           |

Struct format string: `<BIIIBB4IB`

## Header Validation Rules

- `message_type` ∈ {1, 2, 3, 4}
- `dtype` ∈ {1, 2}
- `num_dims` ∈ {1, 2, 3, 4}
- `dims[i]` > 0 for i < num_dims, `dims[i]` = 0 for i >= num_dims
- `payload_size` == product(dims[0:num_dims]) * dtype_size(dtype)

## Message Types

| Value | Name           | Description                                       |
| ----- | -------------- | ------------------------------------------------- |
| 1     | FORWARD        | Hidden states tensor flowing through the pipeline |
| 2     | RESULT         | Final logits returned to client                   |
| 3     | ERROR          | Error response sent upstream                      |
| 4     | HEARTBEAT_DATA | Data-plane heartbeat (reserved)                   |

## Data Types

| Value | Name | Bytes/Element | Format                                 |
| ----- | ---- | ------------- | -------------------------------------- |
| 1     | FP16 | 2             | IEEE 754 half-precision, little-endian |
| 2     | INT8 | 1             | Signed 8-bit integer                   |

## Payload

Tensor data is serialized as raw contiguous bytes in row-major (C-contiguous) order. Size equals `payload_size` from the header. Memory layout matches dtype and dims exactly.

## Typical Message Sizes

| Scenario                      | Shape          | Dtype | Header | Payload | Total   |
| ----------------------------- | -------------- | ----- | ------ | ------- | ------- |
| Single token, hidden_dim=4096 | [1, 1, 4096]   | fp16  | 32 B   | 8 KB    | ~8 KB   |
| 128 token sequence            | [1, 128, 4096] | fp16  | 32 B   | 1 MB    | ~1 MB   |
| 128 token sequence            | [1, 128, 4096] | int8  | 32 B   | 512 KB  | ~512 KB |
| Batch of 4, 128 tokens        | [4, 128, 4096] | fp16  | 32 B   | 4 MB    | ~4 MB   |

## Reliable TCP Framing

TCP is a stream protocol. Individual `recv()` / `send()` calls may transfer fewer bytes than requested.

### read_exact(sock, n)

Loops on `recv()` until exactly `n` bytes are accumulated. Raises `ConnectionError` on EOF before `n` bytes, propagates `socket.timeout`.

### write_all(sock, data)

Loops on `send()` until all bytes are transmitted. Raises `ConnectionError` if `send()` returns 0 (broken connection), propagates `socket.timeout`.

## Message Read Flow

1. `read_exact(32)` — read the fixed-size header
2. `Header.unpack()` — parse and validate header fields
3. `read_exact(payload_size)` — read the tensor payload
4. `bytes_to_tensor()` — reconstruct tensor from raw bytes using dtype and dims

## Message Write Flow

1. Validate tensor data length matches header dims
2. `tensor_to_bytes()` — serialize tensor to raw contiguous bytes
3. Verify `payload_size` matches serialized byte count
4. `header.pack()` — serialize header to 32 bytes
5. `write_all(header + payload)` — send as single contiguous write

## Usage Example

```python
from meshrun.worker.protocol import (
    Header, MessageType, DType, DTYPE_SIZE,
    read_exact, write_all, tensor_to_bytes, bytes_to_tensor,
    read_message, write_message,
)

# Create a header for a 1D fp16 tensor with 10 elements
header = Header(
    message_type=MessageType.FORWARD,
    request_id=1,
    step_id=0,
    payload_size=10 * DTYPE_SIZE[DType.FP16],  # 20 bytes
    dtype=DType.FP16,
    num_dims=1,
    dims=(10, 0, 0, 0),
)
header.validate()  # raises ValueError if invalid

# Serialize header
raw = header.pack()  # exactly 32 bytes
assert len(raw) == 32

# Deserialize header
restored = Header.unpack(raw)  # validates automatically
assert restored == header

# Serialize tensor data
data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
payload = tensor_to_bytes(data, DType.FP16)

# Deserialize tensor data
values = bytes_to_tensor(payload, DType.FP16, dims=(10, 0, 0, 0), num_dims=1)

# Write/read complete messages over a socket
write_message(sock, header, data)
recv_header, recv_data = read_message(sock)
```

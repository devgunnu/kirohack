# TCP Binary Protocol

The data plane uses a custom binary protocol optimized for minimal overhead. No serialization libraries (JSON, Protobuf, MessagePack) are used. Tensors are transmitted as raw contiguous memory with a fixed-size header for framing.

Source: `meshrun/worker/protocol.py`

## Design Principles

- Fixed 32-byte header for predictable parsing
- Zero-copy friendly: payload is raw tensor bytes matching GPU memory layout
- No compression: quantization (fp16/int8) is the size reduction strategy
- Framing via `payload_size` field: receiver always knows exactly how many bytes to read
- Optional AES-256-GCM encryption layer for secure inter-node communication

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

## Message Read Flow (Plaintext)

1. `read_exact(32)` — read the fixed-size header
2. `Header.unpack()` — parse and validate header fields
3. `read_exact(payload_size)` — read the tensor payload
4. `bytes_to_tensor()` — reconstruct tensor from raw bytes using dtype and dims

## Message Write Flow (Plaintext)

1. Validate tensor data length matches header dims
2. `tensor_to_bytes()` — serialize tensor to raw contiguous bytes
3. Verify `payload_size` matches serialized byte count
4. `header.pack()` — serialize header to 32 bytes
5. `write_all(header + payload)` — send as single contiguous write

## Security Layer — AES-256-GCM Encryption

The protocol includes an optional encryption layer using AES-256-GCM authenticated encryption. When enabled, the entire plaintext message (32-byte header + tensor payload) is encrypted before transmission.

### Wire Format (Encrypted)

```
[4-byte encrypted_len (big-endian)][12-byte nonce][ciphertext][16-byte GCM tag]
```

- The 4-byte length prefix enables framing: receiver calls `read_exact(4)` to get the blob length, then `read_exact(length)` for the encrypted data
- Each message uses a fresh random 12-byte (96-bit) nonce
- The 16-byte GCM authentication tag detects tampering or wrong keys

### Encrypted Message Read Flow

1. `read_exact(4)` — read the 4-byte length prefix (big-endian uint32)
2. `read_exact(length)` — read the encrypted blob
3. Decrypt with AES-256-GCM using the session key
4. Parse the decrypted plaintext as header (32 bytes) + payload
5. `Header.unpack()` — validate header fields
6. `bytes_to_tensor()` — reconstruct tensor

### Encrypted Message Write Flow

1. Validate and serialize header + tensor payload (same as plaintext)
2. Concatenate header bytes + payload bytes into plaintext
3. Encrypt with AES-256-GCM using a fresh nonce
4. Prepend 4-byte length prefix
5. `write_all(length_prefix + encrypted_blob)`

### Key Management

Session keys (32-byte AES-256) are generated per-pipeline by the Coordinator and distributed to workers during layer assignment via the gRPC control plane. The `meshrun/security/crypto.py` module provides standalone helpers:

- `generate_session_key()` — 32 random bytes via `os.urandom(32)`
- `derive_key_from_password(password, salt)` — PBKDF2-HMAC-SHA256 key derivation
- `encrypt(plaintext, key, aad)` / `decrypt(blob, key, aad)` — core AES-GCM operations
- `pack_for_wire(plaintext, key)` / `unpack_from_wire(wire_data, key)` — TCP framing helpers

## Usage Example

```python
from meshrun.worker.protocol import (
    Header, MessageType, DType, DTYPE_SIZE,
    read_message, write_message,
    read_message_secure, write_message_secure,
    generate_session_key,
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
header.validate()

# Serialize / deserialize header
raw = header.pack()       # exactly 32 bytes
restored = Header.unpack(raw)  # validates automatically
assert restored == header

# Plaintext message I/O
data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
write_message(sock, header, data)
recv_header, recv_data = read_message(sock)

# Encrypted message I/O
key = generate_session_key()
write_message_secure(sock, header, data, key)
recv_header, recv_data = read_message_secure(sock, key)
```

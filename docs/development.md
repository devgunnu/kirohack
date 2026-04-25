# Development Guide

## Environment

- Python 3.13+
- Package manager: uv
- Shell: PowerShell (Windows)
- Always activate the virtual environment before running commands

```powershell
.venv\Scripts\activate
```

## Code Style

- `dataclass(frozen=True, slots=True)` for immutable value types
- `IntEnum` for protocol enumerations
- Type hints on all function signatures with `from __future__ import annotations`
- Docstrings on all public classes and functions (Google/NumPy style)
- Constants in `UPPER_SNAKE_CASE` at module level
- Private helpers prefixed with `_`
- One sub-component per file

## Linting and Formatting

```powershell
ruff check .
ruff format .
```

## Type Checking

```powershell
mypy meshrun/
```

## Testing

Test files live alongside source files: `meshrun/worker/test_protocol.py` tests `meshrun/worker/protocol.py`.

```powershell
# Run all tests
python -m pytest meshrun/ -v

# Run specific test file
python -m pytest meshrun/worker/test_protocol.py -v

# Run secure protocol tests
python -m pytest meshrun/worker/test_secure_protocol.py -v

# Run with coverage
python -m pytest meshrun/ --cov=meshrun --cov-report=html -v
```

### Testing Conventions

- Use `_valid_header(**overrides)` helper pattern for constructing test fixtures
- Group tests into classes by concern (e.g., `TestValidateAccepts`, `TestRoundTrip`)
- Use `pytest.raises` with `match=` for error message assertions
- Mock sockets with `unittest.mock.MagicMock` for TCP tests
- Use `StubCoordinatorClient` for testing worker nodes without a live Coordinator

### Property-Based Testing

Uses Hypothesis with `@st.composite` strategies for generating valid domain objects.

```python
from hypothesis import given, settings
from hypothesis import strategies as st

@st.composite
def valid_headers(draw):
    # Generate arbitrary valid Header instances
    ...

class TestRoundTrip:
    @given(header=valid_headers())
    @settings(max_examples=200)
    def test_roundtrip_property(self, header):
        assert Header.unpack(header.pack()) == header
```

Run with more examples:

```powershell
python -m pytest meshrun/ -v --hypothesis-max-examples=500
```

## Project Layout

```
meshrun/
  client/
    client.py                # End-to-end inference orchestration
    tokenizer.py             # HuggingFace tokenizer + embedding engine
    transport.py             # Encrypted TCP transport (AES-256-GCM)
  coordinator/
    server.py                # gRPC server and servicer implementation
    registry.py              # Node registry and health tracker
    scheduler.py             # Layer assignment, route building, priority queue
    key_manager.py           # AES-256 session key lifecycle
    proto/                   # Protobuf definitions and generated stubs
      coordinator.proto      # gRPC service definition
      coordinator_pb2.py     # Generated protobuf messages
      coordinator_pb2_grpc.py # Generated gRPC stubs
  worker/
    protocol.py              # TCP binary protocol + AES-256-GCM encryption
    connection_pool.py       # Persistent TCP connection management
    shard_manager.py         # Safetensors selective download, GPU loading
    layer_engine.py          # Forward pass execution
    resource_monitor.py      # GPU memory/utilization tracking
    layer_registry.py        # Layer assignment storage
    coordinator_client.py    # gRPC client for Coordinator communication
    node.py                  # Worker node lifecycle orchestration
    serving.py               # Encrypted request processing pipeline
    test_protocol.py         # Tests for protocol.py
    test_secure_protocol.py  # Tests for secure protocol (AES-GCM)
  security/
    crypto.py                # Standalone AES-256-GCM encryption helpers
```

## Binary Protocol Rules

- Fixed 32-byte header, little-endian, packed with `struct`
- Format string: `<BIIIBB4IB`
- No serialization libraries in the data plane
- Tensors serialized as raw contiguous bytes (row-major, C-contiguous)
- `read_exact(n)` must loop on `recv()` until exactly n bytes accumulated
- `write_all(data)` must loop on `send()` until all bytes transmitted
- Always validate headers before reading payloads

## Architecture Rules

- Control plane (gRPC) and data plane (TCP) are strictly separated
- One persistent TCP connection per node pair, reused across requests
- Blocking I/O is acceptable for POC scope
- Failure handling: single retry to backup node, then fail-fast
- Resource monitoring is observe-only; memory limits are user-configured
- Session keys for encryption are distributed via the gRPC control plane
- All data plane messages use `read_message_secure` / `write_message_secure`

## gRPC Proto Development

The proto file is at `meshrun/coordinator/proto/coordinator.proto`. After modifying it, regenerate stubs:

```powershell
python -m grpc_tools.protoc `
  -I meshrun/coordinator/proto `
  --python_out=meshrun/coordinator/proto `
  --grpc_python_out=meshrun/coordinator/proto `
  meshrun/coordinator/proto/coordinator.proto
```

The generated files (`coordinator_pb2.py`, `coordinator_pb2_grpc.py`) are used by both the Coordinator server and the Worker Node's `GrpcCoordinatorClient`.

## Adding New Components

1. Create a new file under the appropriate package (`meshrun/worker/`, `meshrun/coordinator/`, or `meshrun/client/`)
2. Follow existing patterns: `dataclass(frozen=True, slots=True)` for value types, `IntEnum` for enums
3. Add type hints to all function signatures
4. Write docstrings on all public classes and functions
5. Create a corresponding test file alongside the source (e.g., `test_mycomponent.py`)
6. Update the relevant doc file in `docs/`
7. Update `docs/api-reference.md` with the public interface

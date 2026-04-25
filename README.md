# MeshRun

Distributed AI inference pipeline for ~3B parameter LLMs. Pipelines transformer layer execution across a small GPU cluster using a Coordinator (gRPC control plane) and Worker Nodes (custom TCP binary protocol data plane) with AES-256-GCM encrypted tensor transport.

## Quick Start

```powershell
uv venv .venv
.venv\Scripts\activate
uv pip install -e .
uv pip install pytest hypothesis ruff mypy
```

## Documentation

Full documentation lives in [`docs/`](docs/README.md):

- [Installation Guide](docs/installation.md) — Environment setup, dependencies, first run
- [Architecture Overview](docs/architecture.md) — System design, components, data flow
- [TCP Binary Protocol](docs/protocol.md) — Wire format, header layout, framing, encryption
- [Worker Node Components](docs/worker-components.md) — Connection Pool, Shard Manager, Layer Engine, Resource Monitor, Layer Registry
- [Worker Node Lifecycle](docs/worker-lifecycle.md) — Startup, registration, serving loop, shutdown
- [Coordinator Server](docs/coordinator.md) — gRPC service, node registry, scheduler, key management
- [Inference Client](docs/client.md) — Tokenizer, embedding, secure transport, end-to-end inference
- [API Reference](docs/api-reference.md) — Public interfaces for all modules
- [Development Guide](docs/development.md) — Code style, testing, contribution workflow
- [Usage Guide](docs/usage.md) — Running the full pipeline, configuration, operational patterns

## Project Structure

```
meshrun/
  client/            # Inference Client
    client.py             # End-to-end inference orchestration
    tokenizer.py          # HuggingFace tokenizer + embedding engine
    transport.py          # Encrypted TCP transport (AES-256-GCM)
  coordinator/       # Coordinator Server (control plane)
    server.py             # gRPC server and servicer
    registry.py           # Node registry and health tracker
    scheduler.py          # Layer assignment, route building, priority queue
    key_manager.py        # AES-256 session key lifecycle
    proto/                # Protobuf definitions and generated stubs
  worker/            # Worker Node (data plane)
    protocol.py           # TCP binary protocol + AES-256-GCM encryption
    connection_pool.py    # Persistent TCP connection management
    shard_manager.py      # Safetensors selective download, GPU loading
    layer_engine.py       # Forward pass execution
    resource_monitor.py   # GPU metrics tracking
    layer_registry.py     # Layer assignment storage
    coordinator_client.py # gRPC client for Coordinator
    node.py               # Worker node lifecycle
    serving.py            # Encrypted request processing pipeline
  security/          # Standalone encryption utilities
```

## License

See LICENSE file.

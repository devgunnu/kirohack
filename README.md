# MeshRun

Distributed AI inference pipeline for ~3B parameter LLMs. Pipelines transformer layer execution across a small GPU cluster using a Coordinator (gRPC control plane) and Worker Nodes (custom TCP binary protocol data plane).

## Quick Start

```powershell
uv venv .venv
.venv\Scripts\activate
uv pip install -e .
uv pip install pytest hypothesis ruff mypy cryptography
```

## Documentation

Full documentation lives in [`docs/`](docs/README.md):

- [Installation Guide](docs/installation.md) — Environment setup, dependencies, first run
- [Architecture Overview](docs/architecture.md) — System design, components, data flow
- [TCP Binary Protocol](docs/protocol.md) — Wire format, header layout, framing, encryption
- [Worker Node Components](docs/worker-components.md) — Connection Pool, Shard Manager, Layer Engine, Resource Monitor, Layer Registry
- [Worker Node Lifecycle](docs/worker-lifecycle.md) — Startup, registration, serving loop, shutdown
- [API Reference](docs/api-reference.md) — Public interfaces for all modules
- [Development Guide](docs/development.md) — Code style, testing, contribution workflow
- [Usage Guide](docs/usage.md) — Running worker nodes, configuration, operational patterns

## Project Structure

```
meshrun/
  app/           # Application entry points
  coordinator/   # Control plane (gRPC) — planned
  worker/        # Data plane (complete)
    protocol.py           # TCP binary protocol + AES-256-GCM encryption
    connection_pool.py    # Persistent TCP connection management
    shard_manager.py      # Safetensors selective download, GPU loading
    layer_engine.py       # Forward pass execution
    resource_monitor.py   # GPU metrics tracking
    layer_registry.py     # Layer assignment storage
    coordinator_client.py # gRPC client for Coordinator
    node.py               # Worker node lifecycle
    serving.py            # Request processing pipeline
  security/      # Standalone encryption utilities
```

## License

See LICENSE file.

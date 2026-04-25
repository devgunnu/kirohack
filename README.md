# MeshRun

Distributed AI inference pipeline for ~3B parameter LLMs. Pipelines transformer layer execution across a small GPU cluster using a Coordinator (gRPC control plane) and Worker Nodes (custom TCP binary protocol data plane).

## Quick Start

```powershell
uv venv .venv
.venv\Scripts\activate
uv pip install -e .
uv pip install pytest hypothesis ruff mypy
```

## Documentation

Full documentation lives in [`docs/`](docs/README.md):

- [Installation Guide](docs/installation.md)
- [Architecture Overview](docs/architecture.md)
- [TCP Binary Protocol](docs/protocol.md)
- [Worker Node Components](docs/worker-components.md)
- [API Reference](docs/api-reference.md)
- [Development Guide](docs/development.md)

## Project Structure

```
meshrun/
  app/           # Application entry points
  coordinator/   # Control plane (gRPC)
  worker/        # Data plane (TCP protocol, shard manager, connection pool)
  security/      # Security utilities (out of scope for POC)
```

## License

See LICENSE file.

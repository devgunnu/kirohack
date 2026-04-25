# MeshRun Documentation

MeshRun is a distributed AI inference pipeline for ~3B parameter LLMs. It pipelines transformer layer execution across a small GPU cluster using a Coordinator (control plane, gRPC) and Worker Nodes (data plane, custom TCP binary protocol).

## Documentation Index

| Document                                       | Description                                                    |
| ---------------------------------------------- | -------------------------------------------------------------- |
| [Installation Guide](installation.md)          | Environment setup, dependencies, and first run                 |
| [Architecture Overview](architecture.md)       | System design, components, and data flow                       |
| [TCP Binary Protocol](protocol.md)             | Wire format, header layout, and framing rules                  |
| [Worker Node Components](worker-components.md) | Connection Pool, Shard Manager, Layer Engine, Resource Monitor |
| [API Reference](api-reference.md)              | Public interfaces for all implemented modules                  |
| [Development Guide](development.md)            | Code style, testing, and contribution workflow                 |

## Quick Start

```powershell
# Clone and set up
uv venv .venv
.venv\Scripts\activate
uv pip install -e .
uv pip install pytest hypothesis ruff mypy
```

See [Installation Guide](installation.md) for full details.

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
| [Usage Guide](usage.md)                        | Running worker nodes, configuration, and operational patterns  |

## Quick Start

```powershell
# Clone and set up
uv venv .venv
.venv\Scripts\activate
uv pip install -e .
uv pip install pytest hypothesis ruff mypy
```

See [Installation Guide](installation.md) for full details.

## Current Implementation Status

| Component                 | Status      | File                                        |
| ------------------------- | ----------- | ------------------------------------------- |
| TCP Binary Protocol       | Complete    | `meshrun/worker/protocol.py`                |
| Connection Pool           | Complete    | `meshrun/worker/connection_pool.py`         |
| Shard Manager             | Complete    | `meshrun/worker/shard_manager.py`           |
| Layer Engine              | Complete    | `meshrun/worker/layer_engine.py`            |
| Resource Monitor          | Complete    | `meshrun/worker/resource_monitor.py`        |
| Layer Assignment Registry | Not Started | Planned: `meshrun/worker/layer_registry.py` |
| Worker Node Lifecycle     | Not Started | Tasks 8-10 in implementation plan           |
| Coordinator               | Not Started | `meshrun/coordinator/`                      |
| Client                    | Not Started | `meshrun/app/`                              |

# MeshRun Documentation

MeshRun is a distributed AI inference pipeline for ~3B parameter LLMs. It pipelines transformer layer execution across a small GPU cluster using a Coordinator (control plane, gRPC) and Worker Nodes (data plane, custom TCP binary protocol).

## Documentation Index

| Document                                       | Description                                                     |
| ---------------------------------------------- | --------------------------------------------------------------- |
| [Installation Guide](installation.md)          | Environment setup, dependencies, and first run                  |
| [Architecture Overview](architecture.md)       | System design, components, and data flow                        |
| [TCP Binary Protocol](protocol.md)             | Wire format, header layout, framing rules, and encryption       |
| [Worker Node Components](worker-components.md) | Connection Pool, Shard Manager, Layer Engine, Resource Monitor  |
| [Worker Node Lifecycle](worker-lifecycle.md)   | Node startup, registration, serving loop, and graceful shutdown |
| [Coordinator Server](coordinator.md)           | gRPC service, node registry, scheduler, key management          |
| [Inference Client](client.md)                  | Tokenizer, embedding, secure transport, end-to-end inference    |
| [API Reference](api-reference.md)              | Public interfaces for all implemented modules                   |
| [Development Guide](development.md)            | Code style, testing, and contribution workflow                  |
| [Usage Guide](usage.md)                        | Running nodes, configuration, and operational patterns          |

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

| Component                 | Status   | Location                               |
| ------------------------- | -------- | -------------------------------------- |
| TCP Binary Protocol       | Complete | `meshrun/worker/protocol.py`           |
| Secure Protocol (AES-GCM) | Complete | `meshrun/worker/protocol.py`           |
| Connection Pool           | Complete | `meshrun/worker/connection_pool.py`    |
| Shard Manager             | Complete | `meshrun/worker/shard_manager.py`      |
| Layer Engine              | Complete | `meshrun/worker/layer_engine.py`       |
| Resource Monitor          | Complete | `meshrun/worker/resource_monitor.py`   |
| Layer Assignment Registry | Complete | `meshrun/worker/layer_registry.py`     |
| Coordinator Client        | Complete | `meshrun/worker/coordinator_client.py` |
| Worker Node Lifecycle     | Complete | `meshrun/worker/node.py`               |
| Serving Loop (Encrypted)  | Complete | `meshrun/worker/serving.py`            |
| Coordinator Server (gRPC) | Complete | `meshrun/coordinator/server.py`        |
| Node Registry & Health    | Complete | `meshrun/coordinator/registry.py`      |
| Scheduler & Layer Map     | Complete | `meshrun/coordinator/scheduler.py`     |
| Key Manager               | Complete | `meshrun/coordinator/key_manager.py`   |
| gRPC Proto Stubs          | Complete | `meshrun/coordinator/proto/`           |
| Inference Client          | Complete | `meshrun/client/client.py`             |
| Tokenizer & Embedding     | Complete | `meshrun/client/tokenizer.py`          |
| Secure Transport          | Complete | `meshrun/client/transport.py`          |
| Graceful Shutdown         | Pending  | Task 10 in worker data plane spec      |
| Security Module           | Complete | `meshrun/security/crypto.py`           |

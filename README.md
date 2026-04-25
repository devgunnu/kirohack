# MeshRun

**A distributed AI inference network that splits large language models across multiple machines, enabling collaborative and secure inference without centralized data centers.**

---

## The Problem

AI data centers consume enormous energy and demand keeps doubling — building more is unsustainable. Powerful machines exist everywhere but most people have no access to them. Running large AI models requires expensive, concentrated hardware. The compute is out there across millions of idle devices, but there's no infrastructure to share it.

## The Solution

MeshRun is a distributed AI inference network where people pool their machines to run one model together. Each device processes a few transformer layers and passes encrypted results to the next. A coordinator handles routing and scheduling, while every node's contribution is tracked — making inference shared, secure, and sustainable.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MESHRUN ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │       CLIENT         │
                    │  Tokenize + Embed    │
                    │  Decode Output       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │     COORDINATOR      │
                    │   (Control Plane)    │
                    │                      │
                    │  • Node Registry     │
                    │  • Health Monitoring  │
                    │  • Route Building    │
                    │  • Priority Queue    │
                    │  • Fault Tolerance   │
                    └──────────┬───────────┘
                               │ gRPC
          ┌────────────────────┼────────────────────┐
          │                    │                    │
┌─────────▼─────────┐ ┌───────▼────────┐ ┌────────▼────────┐
│    WORKER NODE A   │ │  WORKER NODE B │ │  WORKER NODE C  │
│   (Layers 0–9)    │ │  (Layers 10–19)│ │  (Layers 20–29) │
│                    │ │                │ │                  │
│ ┌────────────────┐ │ │ ┌────────────┐ │ │ ┌──────────────┐ │
│ │ Shard Manager  │ │ │ │   Shard    │ │ │ │    Shard     │ │
│ │ (Load layers)  │ │ │ │  Manager   │ │ │ │   Manager    │ │
│ ├────────────────┤ │ │ ├────────────┤ │ │ ├──────────────┤ │
│ │ Layer Engine   │ │ │ │   Layer    │ │ │ │    Layer     │ │
│ │ (Forward pass) │ │ │ │  Engine    │ │ │ │   Engine     │ │
│ ├────────────────┤ │ │ ├────────────┤ │ │ ├──────────────┤ │
│ │ Resource Mon.  │ │ │ │  Resource  │ │ │ │   Resource   │ │
│ │ (GPU tracking) │ │ │ │  Monitor   │ │ │ │   Monitor    │ │
│ └────────────────┘ │ │ └────────────┘ │ │ └──────────────┘ │
└─────────┬──────────┘ └───────┬────────┘ └────────┬─────────┘
          │                    │                    │
          │    TCP + AES-256-GCM Encrypted Hops     │
          │◄──────────────────►│◄──────────────────►│
          │   32-byte binary   │   32-byte binary   │
          │     protocol       │     protocol       │
```

### How Data Flows

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  CLIENT  │───►│  NODE A  │───►│  NODE B  │───►│  NODE C  │
│          │    │ Layers   │    │ Layers   │    │ Layers   │
│ Tokenize │    │  0–9     │    │ 10–19    │    │ 20–29    │
│ Embed    │    │          │    │          │    │          │
│          │◄───────────────────────────────────│ Returns  │
│ Decode   │    │          │    │          │    │ logits   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘

Each arrow = [32-byte header][encrypted tensor payload]
             AES-256-GCM • fresh nonce per hop • zero data loss
```

### Control Plane vs Data Plane

```
┌─────────────────────────────────────────────────────┐
│                   CONTROL PLANE (gRPC)              │
│                                                     │
│  Coordinator ◄──► Node Registration                 │
│              ◄──► Heartbeat Health Checks            │
│              ◄──► Layer Assignment                   │
│              ◄──► Route Building                     │
│              ◄──► Failure Rerouting                  │
│              ◄──► Priority Queue Scheduling          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   DATA PLANE (Custom TCP)            │
│                                                     │
│  Node A ──────► Node B ──────► Node C               │
│         tensor         tensor         tensor        │
│         stream         stream         stream        │
│                                                     │
│  • 32-byte binary header (no JSON/Protobuf)         │
│  • Raw tensor bytes (fp16/int8, row-major)          │
│  • Persistent TCP connections (reused per pair)     │
│  • AES-256-GCM encryption at every hop              │
│  • Reliable framing (read_exact / write_all)        │
└─────────────────────────────────────────────────────┘
```

### Worker Node Internals

```
┌─────────────────────────────────────────────────┐
│              WORKER NODE                         │
│                                                 │
│  ┌─────────────┐    ┌──────────────────┐        │
│  │ TCP Listener │───►│ Message Handler  │        │
│  │ (incoming)   │    │ read_exact(32)   │        │
│  └─────────────┘    │ validate header  │        │
│                      │ read_exact(N)    │        │
│                      │ decrypt payload  │        │
│                      └────────┬─────────┘        │
│                               │                  │
│                      ┌────────▼─────────┐        │
│                      │  Layer Engine    │        │
│                      │  Forward pass    │        │
│                      │  through layers  │        │
│                      └────────┬─────────┘        │
│                               │                  │
│                      ┌────────▼─────────┐        │
│                      │ Connection Pool  │        │
│                      │ encrypt + send   │        │
│                      │ to next node     │        │
│                      └──────────────────┘        │
│                                                 │
│  ┌──────────────┐  ┌────────────────────┐       │
│  │Shard Manager │  │ Resource Monitor   │       │
│  │Load assigned │  │ GPU mem tracking   │       │
│  │layers only   │  │ Heartbeat reports  │       │
│  └──────────────┘  └────────────────────┘       │
│                                                 │
│  ┌──────────────────────────────────────┐       │
│  │ Layer Assignment Registry            │       │
│  │ node_id, layers, dtype, downstream   │       │
│  └──────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

### Security Layer

```
┌─────────────────────────────────────────────────────┐
│              ENCRYPTED HOP-TO-HOP TRANSPORT          │
│                                                     │
│  Wire format per message:                           │
│  ┌──────────┬──────────┬────────────┬──────────┐    │
│  │ 4 bytes  │ 12 bytes │  N bytes   │ 16 bytes │    │
│  │ length   │  nonce   │ ciphertext │ GCM tag  │    │
│  └──────────┴──────────┴────────────┴──────────┘    │
│                                                     │
│  • AES-256-GCM authenticated encryption             │
│  • 256-bit random session key                       │
│  • Fresh 96-bit nonce per message                   │
│  • Tamper detection via 128-bit auth tag            │
│  • Verified zero numerical loss (fp16 + int8)       │
│  • ~0.3% latency overhead on total inference time   │
└─────────────────────────────────────────────────────┘
```

---

## Screenshots

### Client — Distributed Inference
<!-- Add screenshot: terminal showing prompt → encrypted tensor flow → response -->
![Client Inference](docs/images/client-inference.png)

### Node Dashboard — Resource Monitoring
<!-- Add screenshot: dashboard showing connected nodes, GPU usage, active requests -->
![Node Dashboard](docs/images/node-dashboard.png)

### Node Connections — Live Pipeline
<!-- Add screenshot: terminal showing node-to-node encrypted tensor streaming -->
![Node Connections](docs/images/node-connections.png)

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/devgunnu/kirohack.git
cd kirohack

# Install dependencies
uv venv .venv
.venv\Scripts\activate
uv pip install -e .
uv pip install pytest hypothesis cryptography

# Run tests
python -m pytest meshrun/worker/test_protocol.py -v
```

## Project Structure

```
meshrun/
  coordinator/     # Control plane (gRPC) — routing, scheduling, health
  worker/          # Data plane — TCP protocol, encryption, tensor streaming
    protocol.py              # 32-byte binary protocol + AES-256-GCM
    connection_pool.py       # Persistent TCP connection management
    shard_manager.py         # Selective weight download + GPU loading
    layer_engine.py          # Forward pass execution
    resource_monitor.py      # GPU metrics tracking
    layer_registry.py        # Layer assignment storage
    coordinator_client.py    # gRPC client for Coordinator
    node.py                  # Worker node lifecycle
    serving.py               # Request processing pipeline
  security/        # Standalone encryption module + demo
    crypto.py                # AES-256-GCM core functions
    model.py                 # TinyLlama inference via ctransformers
    main.py                  # E2E single-node demo
  app/             # Application entry points
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Control Plane | gRPC + Protobuf | Node registration, health checks, routing |
| Data Plane | Custom TCP binary protocol | Tensor streaming between nodes |
| Encryption | AES-256-GCM (cryptography lib) | Hop-to-hop authenticated encryption |
| Model Format | Safetensors | Selective layer download via HTTP Range |
| Inference | PyTorch | Forward pass execution on GPU |
| Quantization | fp16 / int8 | Reduced memory + transfer size |
| Testing | pytest + Hypothesis | Unit + property-based + integration tests |

## Built With

- [Kiro](https://kiro.dev) — Spec-driven development, agent hooks, powers
- [Context7](https://context7.com) — Up-to-date library documentation for MCP
- [Sequential Thinking](https://github.com/modelcontextprotocol/servers) — Step-by-step reasoning for complex design decisions

## Team — Port 37

- **Aaditya** — Security layer, encryption integration, tensor stability testing
- **Gunbir** — Binary TCP protocol, worker node architecture, connection management
- **Vishal** — Coordinator, routing logic, priority queue, fault tolerance

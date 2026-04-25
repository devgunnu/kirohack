# MeshRun Distributed Testing Guide (Tailscale)

This guide walks you through testing the full distributed inference pipeline across multiple physical machines connected via Tailscale.

You need at least **two machines** (one Coordinator + one Worker), but a realistic test uses **three machines** (Coordinator, two Workers, plus a Client which can be any of them).

---

## 0. Prerequisites (every machine)

1. **Python 3.11+** and **`uv`** (or any Python package manager).
2. **Tailscale** installed and authenticated to the same tailnet on every machine.
   - Install: https://tailscale.com/download
   - On each machine, run `tailscale up` (Linux/macOS) or sign in via the desktop app (Windows).
   - Verify the tailnet by running `tailscale ip --4` on each machine and noting each machine's `100.x.y.z` address.
3. **GPU on at least one Worker.** PyTorch must see CUDA:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
   ```
4. **Identical clone of this repo on every machine**, on the same branch.

### Install MeshRun (every machine)

```bash
cd "Distributed Inference Server"
uv venv .venv
# Activate:
#   Windows (PowerShell):  .venv\Scripts\Activate.ps1
#   Windows (Git Bash):    source .venv/Scripts/activate
#   Linux/macOS:           source .venv/bin/activate
uv pip install -e .
```

This single install picks up the unified `pyproject.toml` at the repo root and exposes the `meshrun` CLI. Verify:

```bash
meshrun --help
```

You should see `coordinator`, `worker`, `submit`, `status`, `nodes`, etc., listed.

---

## 1. Decide your topology

Pick one machine to play each role and note its Tailscale IP:

| Role         | Machine          | Tailscale IP    |
| ------------ | ---------------- | --------------- |
| Coordinator  | (e.g. desktop)   | `100.64.0.1`    |
| Worker 1     | (e.g. GPU box A) | `100.64.0.2`    |
| Worker 2     | (e.g. GPU box B) | `100.64.0.3`    |
| Client       | (any machine)    | `100.64.0.4`    |

Replace these IPs with the ones from your tailnet (`tailscale ip --4`) throughout the rest of this guide.

You also need the model parameters:

| Setting         | Example value                                           |
| --------------- | ------------------------------------------------------- |
| `--model`       | `meta-llama/Llama-3.2-3B`                               |
| `--layers`      | `28` (Llama-3.2-3B has 28 transformer layers)           |
| `--dtype`       | `int8` (smaller memory footprint) or `fp16`             |
| `--model-url`   | direct HTTPS link to a `.safetensors` file              |

The `--model-url` must be a single safetensors file the workers can download via HTTP Range requests (Hugging Face `resolve/main/...` URLs work). Make sure the file is publicly downloadable or that every worker has network credentials cached.

---

## 2. Start the Coordinator

On the Coordinator machine:

```bash
meshrun coordinator \
  --host 0.0.0.0 \
  --port 50051 \
  --model meta-llama/Llama-3.2-3B \
  --layers 28 \
  --dtype int8 \
  --model-url "https://huggingface.co/.../model.safetensors"
```

You should see:

```
Starting MeshRun Coordinator
  gRPC: 0.0.0.0:50051
  Model: meta-llama/Llama-3.2-3B (28 layers, int8)
Coordinator started. Press Ctrl+C to stop.
```

Leave this terminal open. The Coordinator now listens on `100.64.0.1:50051` over Tailscale.

> **Firewall tip (Windows):** if connections fail, allow inbound TCP 50051 in Windows Defender Firewall, or ensure the rule is restricted to the Tailscale interface.

---

## 3. Start Worker 1

On Worker 1 (Tailscale IP `100.64.0.2`):

```bash
meshrun worker \
  --coordinator 100.64.0.1:50051 \
  --host 0.0.0.0 \
  --data-port 9100 \
  --grpc-port 50052 \
  --advertise 100.64.0.2:9100 \
  --gpu-limit 4096 \
  --device 0
```

What happens:

1. Worker calls `startup()` and queries its GPU.
2. Worker starts its own gRPC server on `100.64.0.2:50052` (so the Coordinator can push `AcceptLayerAssignment`).
3. Worker calls `Register` on the Coordinator.
4. **Auto-assignment fires**: the Coordinator computes layer assignments across every healthy node and pushes `AcceptLayerAssignment` back over gRPC.
5. Worker downloads its assigned shard via HTTP Range requests, builds the layer engine, sends `ConfirmReady`, and starts the data-plane TCP listener on port 9100.

You should see roughly this in the worker logs:

```
GPU detected: ... MB total, ... MB free
Registered with coordinator
Worker assignment server started on 0.0.0.0:50052
Received layer assignment: model=meta-llama/Llama-3.2-3B, layers 0-13, is_final=False
Node ... shard loaded successfully ... → VALIDATING
Node ... confirmed ready → READY
Node ... now SERVING on 0.0.0.0:9100
```

If `--advertise` is omitted, the worker tries `tailscale ip --4` and falls back to `socket.gethostbyname(socket.gethostname())`. **Always pass `--advertise` explicitly when running over Tailscale** — it guarantees the address other peers will use to reach this worker.

---

## 4. Start Worker 2 (and any further workers)

On Worker 2 (`100.64.0.3`):

```bash
meshrun worker \
  --coordinator 100.64.0.1:50051 \
  --host 0.0.0.0 \
  --data-port 9100 \
  --grpc-port 50052 \
  --advertise 100.64.0.3:9100 \
  --gpu-limit 4096 \
  --device 0
```

When Worker 2 registers, the Coordinator **recomputes the layer assignments and re-pushes them to BOTH workers**. Watch Worker 1's terminal — it should receive a fresh `AcceptLayerAssignment` (e.g. now layers `0-13` instead of `0-27`) and reload its shard.

> If you are running multiple workers on the *same* machine, give them different ports: `--data-port 9101 --grpc-port 50053` for the second one and matching `--advertise <ip>:9101`.

---

## 5. Verify the network from the Coordinator's view

From the Client machine (or any machine in the tailnet), confirm cluster state:

```bash
meshrun status
# or
meshrun nodes
```

These call `GetNetworkStatus` on the Coordinator and should report **all healthy nodes** and **layers covered = total layers**.

> The first time you run any `meshrun <something>` command on a machine that hasn't been "joined", you may see a `meshrun join` onboarding prompt. This is a CLI UX gate from `meshrun/app/state.py`; run `meshrun join` once to satisfy it locally.

---

## 6. Submit an inference

Set the `MESHRUN_MODEL_URL` env var so the client knows where to fetch embedding/LM-head weights (it streams the embedding tensors directly from the same safetensors file):

```bash
# Linux / macOS
export MESHRUN_COORDINATOR_URL="100.64.0.1:50051"
export MESHRUN_MODEL_URL="https://huggingface.co/.../model.safetensors"

# Windows PowerShell
$env:MESHRUN_COORDINATOR_URL = "100.64.0.1:50051"
$env:MESHRUN_MODEL_URL       = "https://huggingface.co/.../model.safetensors"
```

Then submit:

```bash
meshrun submit "What is the capital of France?"
```

Pipeline behavior end-to-end:

1. CLI builds an `InferenceClient`, loads the tokenizer, downloads the embedding tensor.
2. Client calls `RequestRoute` on the Coordinator → receives an ordered list of nodes plus the AES-256 session key.
3. Client connects via encrypted TCP to Worker 1, sends the `FORWARD` message.
4. Worker 1 runs its layers, forwards encrypted hidden states to Worker 2.
5. Worker 2 runs the final layers, projects to logits, encrypts, sends back.
6. Client decrypts, decodes the logits, detokenizes, prints the text.

---

## 7. Optional: Watch the live dashboard

On any machine that can reach the Coordinator:

```bash
meshrun dashboard
```

This launches a FastAPI app on `http://127.0.0.1:7654` with a websocket that polls `GetNetworkStatus` and streams updates to the browser.

---

## 8. Verification checklist

Tick these off in order. If any item fails, see the troubleshooting section below.

- [ ] `tailscale ping <other-machine>` works in both directions between every pair of machines.
- [ ] `meshrun --help` lists `coordinator` and `worker`.
- [ ] Coordinator logs `Coordinator server started on 0.0.0.0:50051`.
- [ ] Worker 1 logs `Registered with coordinator`.
- [ ] Worker 1 logs `Received layer assignment: ... layers 0-N`.
- [ ] Worker 1 logs `Node ... now SERVING on 0.0.0.0:9100`.
- [ ] Worker 2 registers and Worker 1 receives a **second** `AcceptLayerAssignment` with a smaller range.
- [ ] `meshrun status` from the Client shows both workers as `active` and `covered_layers == total_layers`.
- [ ] `meshrun submit "<prompt>"` returns generated text without raising.
- [ ] (Optional) `meshrun dashboard` opens and the websocket stays connected.

---

## 9. Troubleshooting

**`pydantic-settings` import error on first run** — you skipped `uv pip install -e .` after the new unified `pyproject.toml` landed. Run it.

**Worker exits with `Registration failed: ...`** — the coordinator can't see the worker's tailnet address, OR the worker advertised `0.0.0.0` (which is unusable from the coordinator's side). Always pass `--advertise <tailnet-ip>:<data-port>`.

**Worker registers but never enters SERVING** — the Coordinator pushed `AcceptLayerAssignment` but couldn't reach the worker's gRPC port. Check:
- Worker started with `--grpc-port 50052` (or whichever port).
- Coordinator can reach `<worker-tailnet-ip>:50052` (try `nc -vz` or `Test-NetConnection`).
- Firewall allows inbound TCP on the gRPC port.

**`Insufficient capacity` from the Coordinator** — the workers' `--gpu-limit` total minus framework overhead can't fit the model. Either lower `--layers`, switch `--dtype` to `int8`, or add more workers.

**Shard download stalls or fails** — the `--model-url` must support HTTP Range requests. HF `resolve/main/*.safetensors` URLs work; LFS-pointer URLs do not. Test with `curl -I -H "Range: bytes=0-100" <url>` and look for `HTTP/1.1 206 Partial Content`.

**Inference returns garbled text or errors** — confirm every machine is using the *exact same* model URL/version. A version mismatch between Coordinator/Worker/Client (different layer count or dtype) corrupts the pipeline silently.

**Two workers on the same machine collide** — give each a distinct `--data-port` and `--grpc-port`, and a distinct `--device` if both are using GPUs.

---

## 10. Tearing down

`Ctrl+C` in each terminal stops the corresponding process gracefully:
- Coordinator stops the gRPC server and the health tracker thread.
- Worker stops its data-plane listener, stops the gRPC server, and stops the heartbeat sender.

The Coordinator's registry detects missed heartbeats and transitions dead workers to `DEAD` automatically — but a clean `Ctrl+C` is faster.

---

## 11. Topology quick reference

```
┌─────────────────────────── Tailscale (100.64.0.0/10) ─────────────────────────┐
│                                                                                │
│  ┌──────────────────┐        gRPC :50051         ┌──────────────────────────┐ │
│  │   Coordinator    │ ◄────── Register ────────  │   Worker 1               │ │
│  │   100.64.0.1     │        Heartbeat,          │   100.64.0.2             │ │
│  │                  │        ConfirmReady        │   gRPC :50052            │ │
│  │                  │ ──── AcceptLayerAssignment ►   data :9100  layers 0-13│ │
│  └──────────────────┘                            └──────────┬───────────────┘ │
│           ▲                                                  │ TCP encrypted   │
│           │ RequestRoute                                     ▼                 │
│           │                                       ┌──────────────────────────┐ │
│           │                                       │   Worker 2               │ │
│           │                                       │   100.64.0.3             │ │
│  ┌────────┴─────────┐                             │   data :9100  layers 14-27│ │
│  │     Client       │ ─── encrypted FORWARD ────► │                          │ │
│  │   100.64.0.4     │ ◄── encrypted RESULT ────── │                          │ │
│  └──────────────────┘                             └──────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────┘
```

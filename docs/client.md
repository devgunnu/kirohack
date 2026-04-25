# Inference Client

The Inference Client is the user-facing entry point for MeshRun. It handles the full pipeline from text input to text output: tokenization, local embedding, route acquisition from the Coordinator, encrypted tensor transmission through the worker pipeline, and logits decoding.

Source: `meshrun/client/`

## Components

### ModelTokenizer (`tokenizer.py`)

Loads a HuggingFace AutoTokenizer matching the served model, selectively downloads embedding weights from the safetensors model file, and provides local embedding lookup and logits decoding.

| Method           | Description                                                         |
| ---------------- | ------------------------------------------------------------------- |
| `load_tokenizer` | Load AutoTokenizer via `transformers.AutoTokenizer.from_pretrained` |
| `load_embedding` | Selectively download `embed_tokens` weights via HTTP Range requests |
| `tokenize`       | Convert text to token IDs                                           |
| `detokenize`     | Convert token IDs back to text                                      |
| `embed`          | Embedding lookup → `[1, seq_len, hidden_dim]` fp16 tensor           |
| `decode_logits`  | Greedy argmax on last token position → output token IDs             |

**Tokenizer loading** uses `transformers.AutoTokenizer.from_pretrained(model_name_or_path)`. The model name must match the HuggingFace model ID (e.g., `"meta-llama/Llama-3.2-3B"`).

**Embedding weight loading** reuses the Shard Manager infrastructure (`fetch_safetensors_header`, `download_selected_tensors_cached`) to download only the `embed_tokens` tensor via HTTP Range requests, with local caching.

**Embedding execution**: `hidden_states = embedding_weight[token_ids]` → output shape `[1, seq_len, hidden_dim]` in fp16.

**Decoding**: Greedy argmax on the last position: `argmax(logits[:, -1, :])`.

### SecureTransport (`transport.py`)

Manages encrypted TCP connections to the worker pipeline. Wraps `protocol.py`'s `write_message_secure` and `read_message_secure` with connection management.

| Method           | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `connect`        | Establish TCP connection to a worker node (host:port string) |
| `send_forward`   | Build FORWARD header from tensor, encrypt and send via TCP   |
| `receive_result` | Receive and decrypt RESULT message, validate message type    |
| `close`          | Close all open TCP sockets                                   |

**send_forward** inspects the PyTorch tensor's shape and dtype, builds a protocol `Header`, flattens the tensor to a list, and calls `write_message_secure`. Supports fp16 and int8 tensors with up to 4 dimensions.

**receive_result** calls `read_message_secure`, validates the message type is RESULT (raises `RuntimeError` for ERROR messages or decryption failures), and returns `(header, tensor_data)`.

### InferenceClient (`client.py`)

Orchestrates the end-to-end inference flow. Coordinates between the tokenizer, Coordinator gRPC client, and secure TCP transport.

| Method             | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| `__init__`         | Initialize with coordinator_address, model_name, model_url    |
| `initialize`       | Load tokenizer and embedding weights                          |
| `request_route`    | gRPC call to Coordinator → ExecutionPath with session key     |
| `submit_inference` | Full flow: tokenize → embed → route → encrypt → send → decode |
| `close`            | Release gRPC channel and transport sockets                    |

## Usage

### Basic Inference

```python
from meshrun.client.client import InferenceClient

client = InferenceClient(
    coordinator_address="10.0.0.1:50051",
    model_name="meta-llama/Llama-3.2-3B",
    model_url="https://huggingface.co/model/resolve/main/model.safetensors",
    cache_dir="./cache",
    device="cpu",
)

# Load tokenizer and embedding weights (one-time setup)
client.initialize()

# Run inference
output = client.submit_inference("What is artificial intelligence?")
print(output)

# Cleanup
client.close()
```

### Step-by-Step Flow

```python
from meshrun.client.tokenizer import ModelTokenizer
from meshrun.client.transport import SecureTransport

# 1. Tokenize
tokenizer = ModelTokenizer()
tokenizer.load_tokenizer("meta-llama/Llama-3.2-3B")
token_ids = tokenizer.tokenize("What is AI?")

# 2. Embed locally
tokenizer.load_embedding(
    model_url="https://example.com/model.safetensors",
    cache_dir="./cache",
    device="cpu",
)
hidden_states = tokenizer.embed(token_ids)
# hidden_states shape: [1, seq_len, hidden_dim], dtype: fp16

# 3. Get route from Coordinator (via gRPC)
# execution_path = client.request_route("llama-3b")
# session_key = execution_path.session_key
# first_node = execution_path.nodes[0].address

# 4. Send encrypted hidden states to pipeline
transport = SecureTransport(connect_timeout=5.0)
sock = transport.connect("192.168.1.10:9000")
transport.send_forward(sock, hidden_states, session_key, request_id=1, step_id=0)

# 5. Receive encrypted logits
header, logits_data = transport.receive_result(sock, session_key)

# 6. Decode
output_ids = tokenizer.decode_logits(logits_tensor)
text = tokenizer.detokenize(output_ids)

transport.close()
```

## Data Classes

### RouteNode

```python
@dataclass
class RouteNode:
    node_id: str       # Worker node identifier
    address: str       # TCP host:port
    layer_start: int   # First layer (inclusive)
    layer_end: int     # Last layer (inclusive)
```

### ExecutionPath

```python
@dataclass
class ExecutionPath:
    request_id: int           # Unique request identifier
    session_key: bytes        # 32-byte AES-256 key
    nodes: list[RouteNode]    # Ordered pipeline nodes
    backup_map: dict[str, str] # node_id → backup_address
```

## Error Handling

- **Tokenizer not loaded**: `RuntimeError` if `tokenize()` or `embed()` called before `load_tokenizer()` / `load_embedding()`
- **PyTorch not installed**: `RuntimeError` from embedding and transport operations
- **transformers not installed**: `RuntimeError` from tokenizer loading
- **Connection failure**: `ConnectionError` from `SecureTransport.connect()`
- **Decryption failure**: `RuntimeError` wrapping `InvalidTag` from wrong session key or tampered data
- **Pipeline error**: `RuntimeError` if the pipeline returns an ERROR message type
- **gRPC errors**: Handled gracefully in `request_route()` (unavailable, deadline exceeded)

## Dependencies

- `transformers` — HuggingFace AutoTokenizer
- `torch` — Embedding lookup, tensor operations, logits decoding
- `grpcio` — gRPC client for Coordinator communication
- `cryptography` — AES-256-GCM encryption (via `protocol.py`)
- `meshrun.worker.protocol` — Secure message read/write
- `meshrun.worker.shard_manager` — Selective safetensors download for embedding weights

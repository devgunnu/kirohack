"""
Secure Inference Pipeline — E2E Demo

Stages:
  1. Prompt → torch tensors (tokenize + embed)
  2. Client encrypts tensor  (AES-256-GCM)
  3. Node A decrypts + reconstructs tensor
  4. Node A runs inference
  5. Node A encrypts response
  6. Client decrypts + prints answer

Run: python main.py
"""

import struct
import time
import numpy as np
import torch
import torch.nn as nn
import crypto
import model as mdl

# ── binary protocol ─────────────────────────────────────────────────
FORWARD     = 1
RESULT      = 2
DTYPE_FP16  = 1
HEADER_FMT  = "<B I I I B B I I I I B"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def pack_header(msg_type, req_id, payload_size, dtype, dims):
    padded = list(dims) + [0] * (4 - len(dims))
    return struct.pack(HEADER_FMT,
        msg_type, req_id, 0, payload_size,
        dtype, len(dims),
        padded[0], padded[1], padded[2], padded[3], 0)


def unpack_header(data):
    f = struct.unpack(HEADER_FMT, data)
    return {"msg_type": f[0], "req_id": f[1],
            "payload_size": f[3], "num_dims": f[5], "dims": list(f[6:10])}


# ── one-time setup ───────────────────────────────────────────────────
print("Loading tokenizer...", end=" ", flush=True)
from transformers import AutoTokenizer
_tok = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True)
print("done.")

HIDDEN_DIM = 256
torch.manual_seed(42)
_embed = nn.Embedding(_tok.vocab_size, HIDDEN_DIM)
_embed.weight.data = _embed.weight.data.half()
_embed.eval()

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)


# ── pipeline ─────────────────────────────────────────────────────────
def run(prompt: str):
    req_id = torch.randint(1, 100_000, (1,)).item()
    t_wall = time.perf_counter()

    print()
    print("═" * 68)
    print(f"  PROMPT  »  {prompt}")
    print("═" * 68)

    # ── STAGE 1: Prompt → torch tensors ─────────────────────────────
    t0 = time.perf_counter()

    token_ids: torch.Tensor = _tok.encode(prompt, return_tensors="pt")[0]
    tokens = _tok.convert_ids_to_tokens(token_ids.tolist())

    with torch.no_grad():
        embeddings: torch.Tensor = _embed(token_ids.unsqueeze(0))
        # [1, seq_len, hidden_dim]  torch.float16

    t1 = time.perf_counter()
    print()
    print(f"┌─ STAGE 1 · Prompt → Torch Tensors  ({(t1-t0)*1000:.2f} ms)")
    print(f"│  tokens      : {tokens}")
    print(f"│  token_ids   : {token_ids.tolist()}")
    print(f"│  token tensor: {token_ids}")
    print(f"│               shape={list(token_ids.shape)}  dtype={token_ids.dtype}")
    print(f"│")
    print(f"│  embeddings  : shape={list(embeddings.shape)}  dtype={embeddings.dtype}")
    print(f"│  token[0]    : {embeddings[0, 0]}")
    if embeddings.shape[1] > 1:
        print(f"│  token[1]    : {embeddings[0, 1]}")
    print(f"└──────────────────────────────────────────────────────────────")

    # ── STAGE 2: Client encrypts tensor ─────────────────────────────
    t0 = time.perf_counter()

    payload  = embeddings.contiguous().cpu().numpy().tobytes()
    dims     = list(embeddings.shape)
    header   = pack_header(FORWARD, req_id, len(payload), DTYPE_FP16, dims)
    message  = header + payload
    key      = crypto.generate_session_key()
    wire     = crypto.pack_for_wire(message, key)
    overhead = len(wire) - len(message)

    t1 = time.perf_counter()
    print()
    print(f"┌─ STAGE 2 · Client Encrypts  ({(t1-t0)*1000:.2f} ms)")
    print(f"│  AES-256-GCM key  : {key.hex()}")
    print(f"│  nonce            : {wire[4:16].hex()}")
    print(f"│  plaintext size   : {len(message):,} bytes  (32 hdr + {len(payload):,} fp16 payload)")
    print(f"│  wire size        : {len(wire):,} bytes  (+{overhead} overhead)")
    print(f"│  ciphertext [0:32]: {wire[16:48].hex()}")
    print(f"└──────────────────────────────────────────────────────────────")

    # ── STAGE 3: Node A decrypts + reconstructs tensor ──────────────
    t0 = time.perf_counter()

    dec_msg     = crypto.unpack_from_wire(wire, key)
    hdr         = unpack_header(dec_msg[:HEADER_SIZE])
    dec_payload = dec_msg[HEADER_SIZE:]
    rec_dims    = hdr["dims"][: hdr["num_dims"]]
    rec_np      = np.frombuffer(dec_payload, dtype=np.float16).reshape(rec_dims).copy()
    rec_tensor: torch.Tensor = torch.from_numpy(rec_np)

    bitwise_ok = torch.equal(embeddings.cpu(), rec_tensor)
    max_diff   = (embeddings.float() - rec_tensor.float()).abs().max().item()

    t1 = time.perf_counter()
    print()
    print(f"┌─ STAGE 3 · Node A Decrypts & Reconstructs  ({(t1-t0)*1000:.2f} ms)")
    print(f"│  received         : {len(wire):,} encrypted bytes")
    print(f"│  decrypted to     : {len(dec_msg):,} bytes")
    print(f"│  tensor shape     : {list(rec_tensor.shape)}  dtype={rec_tensor.dtype}")
    print(f"│  token[0]         : {rec_tensor[0, 0]}")
    if rec_tensor.shape[1] > 1:
        print(f"│  token[1]         : {rec_tensor[0, 1]}")
    print(f"│  bitwise identical: {'yes' if bitwise_ok else 'NO — mismatch'}")
    print(f"│  max abs diff     : {max_diff}  (0 = lossless)")
    print(f"└──────────────────────────────────────────────────────────────")

    # ── STAGE 4: Node A inference ────────────────────────────────────
    t0 = time.perf_counter()

    print()
    print(f"┌─ STAGE 4 · Node A Inference  (running...)", flush=True)
    answer = mdl.run_model(prompt)

    t1 = time.perf_counter()
    print(f"│  prompt  : \"{prompt}\"")
    print(f"│  answer  : \"{answer}\"")
    print(f"│  latency : {(t1-t0)*1000:.2f} ms")
    print(f"└──────────────────────────────────────────────────────────────")

    # ── STAGE 5: Node A encrypts response ───────────────────────────
    t0 = time.perf_counter()

    resp_bytes = answer.encode("utf-8")
    resp_hdr   = pack_header(RESULT, req_id, len(resp_bytes), DTYPE_FP16, [len(resp_bytes)])
    resp_msg   = resp_hdr + resp_bytes
    resp_wire  = crypto.pack_for_wire(resp_msg, key)

    t1 = time.perf_counter()
    print()
    print(f"┌─ STAGE 5 · Node A Encrypts Response  ({(t1-t0)*1000:.2f} ms)")
    print(f"│  response text  : \"{answer}\"")
    print(f"│  nonce          : {resp_wire[4:16].hex()}")
    print(f"│  wire size      : {len(resp_wire)} bytes")
    print(f"│  ciphertext[0:32]: {resp_wire[16:48].hex()}")
    print(f"└──────────────────────────────────────────────────────────────")

    # ── STAGE 6: Client decrypts ─────────────────────────────────────
    t0 = time.perf_counter()

    dec_resp = crypto.unpack_from_wire(resp_wire, key)
    resp_hdr_parsed = unpack_header(dec_resp[:HEADER_SIZE])
    final    = dec_resp[HEADER_SIZE:].decode("utf-8")

    t1 = time.perf_counter()
    t_total = time.perf_counter() - t_wall
    print()
    print(f"┌─ STAGE 6 · Client Decrypts  ({(t1-t0)*1000:.2f} ms)")
    print(f"│  msg type  : {resp_hdr_parsed['msg_type']}  (2 = RESULT)")
    print(f"│  req id    : {resp_hdr_parsed['req_id']}")
    print(f"│  answer    : \"{final}\"")
    print(f"└──────────────────────────────────────────────────────────────")

    print()
    print(f"  total wall time : {t_total*1000:.2f} ms")
    print(f"  cipher          : AES-256-GCM")
    print(f"  tensor dtype    : {embeddings.dtype}  (2 bytes/element)")
    print(f"  data loss       : {'none' if bitwise_ok else 'YES'}")
    print()


if __name__ == "__main__":
    while True:
        prompt = input("Enter prompt (or quit): ").strip()
        if not prompt or prompt.lower() == "quit":
            break
        run(prompt)

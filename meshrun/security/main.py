"""
Secure Inference Pipeline — Client → NodeA → Client

Uses PyTorch tensors, real tokenizer embeddings, AES-256-GCM encryption,
and the worker protocol's binary format. Shows every conversion step
and latency at each stage.

Run: python main.py
"""

import sys
import os
import struct
import time

import torch
import numpy as np

# Add parent so we can import the worker protocol
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import crypto
import model as mdl
from meshrun.worker.protocol import Header, MessageType, DType, DTYPE_SIZE

# ─── Helpers ────────────────────────────────────────────────────────

def timer():
    """Returns (elapsed_ms, result) context manager."""
    class T:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *a):
            self.ms = (time.perf_counter() - self.start) * 1000
    return T()


def torch_to_wire(tensor: torch.Tensor, dtype_code: int,
                  req_id: int, step_id: int = 0) -> tuple[bytes, bytes]:
    """Convert a torch tensor to the spec's binary protocol message (header + payload).

    Returns (header_bytes, payload_bytes).
    """
    # Ensure contiguous C-order
    t = tensor.contiguous()

    # Convert to raw bytes via numpy (torch → numpy → bytes)
    if dtype_code == DType.FP16:
        np_arr = t.to(torch.float16).numpy()
    else:
        np_arr = t.to(torch.int8).numpy()

    payload = np_arr.tobytes()
    dims = list(t.shape)

    header = Header(
        message_type=MessageType.FORWARD,
        request_id=req_id,
        step_id=step_id,
        payload_size=len(payload),
        dtype=dtype_code,
        num_dims=len(dims),
        dims=tuple(dims + [0] * (4 - len(dims))),
    )
    header.validate()
    return header.pack(), payload


def wire_to_torch(header_bytes: bytes, payload: bytes) -> tuple[torch.Tensor, Header]:
    """Reconstruct a torch tensor from binary protocol message."""
    hdr = Header.unpack(header_bytes)
    active_dims = list(hdr.dims[:hdr.num_dims])

    if hdr.dtype == DType.FP16:
        np_arr = np.frombuffer(payload, dtype=np.float16).reshape(active_dims)
        tensor = torch.from_numpy(np_arr.copy())
    else:
        np_arr = np.frombuffer(payload, dtype=np.int8).reshape(active_dims)
        tensor = torch.from_numpy(np_arr.copy())

    return tensor, hdr


# ─── Tokenizer (loaded once) ───────────────────────────────────────

print("Loading tokenizer...", end=" ", flush=True)
from transformers import AutoTokenizer
_tok = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True
)
print("done.")


# ─── Run ────────────────────────────────────────────────────────────

def run(prompt: str):
    req_id = int(torch.randint(1, 100000, (1,)).item())
    latencies = {}

    print()
    print("╔" + "═" * 62 + "╗")
    print("║  CLIENT → NODE A → CLIENT  (AES-256-GCM + PyTorch)         ║")
    print("╚" + "═" * 62 + "╝")

    # ── STEP 1: Tokenize ────────────────────────────────────────
    print()
    print("━" * 64)
    print("  STEP 1: TOKENIZE  (str → token IDs)")
    print("━" * 64)

    with timer() as t1:
        input_ids = _tok.encode(prompt, return_tensors="pt")[0]  # torch.LongTensor
    latencies["tokenize"] = t1.ms

    tokens = _tok.convert_ids_to_tokens(input_ids.tolist())
    print(f"  Input:      \"{prompt}\"")
    print(f"  Tokens:     {tokens}")
    print(f"  IDs:        {input_ids.tolist()}")
    print(f"  ID tensor:  {input_ids.shape}  dtype={input_ids.dtype}")
    print(f"  Latency:    {t1.ms:.2f} ms")

    # ── STEP 2: Embed ───────────────────────────────────────────
    print()
    print("━" * 64)
    print("  STEP 2: EMBED  (token IDs → torch.float16 tensor)")
    print("━" * 64)

    HIDDEN_DIM = 256
    torch.manual_seed(42)

    with timer() as t2:
        # Simulate embedding lookup (in real system: model.embed_tokens(input_ids))
        embed_weight = torch.randn(_tok.vocab_size, HIDDEN_DIM, dtype=torch.float16)
        embeddings = embed_weight[input_ids]                    # [seq_len, hidden_dim]
        embeddings = embeddings.unsqueeze(0)                    # [1, seq_len, hidden_dim]
    latencies["embed"] = t2.ms

    print(f"  embed_weight: torch.Size([{_tok.vocab_size}, {HIDDEN_DIM}])  dtype={embed_weight.dtype}")
    print(f"  Lookup:       embed_weight[{input_ids.tolist()}]")
    print(f"  Result:       {embeddings.shape}  dtype={embeddings.dtype}")
    print(f"  Memory:       {embeddings.nelement() * 2:,} bytes ({embeddings.nelement() * 2 / 1024:.1f} KB)")
    print(f"  Stats:        mean={embeddings.float().mean():.4f}  std={embeddings.float().std():.4f}")
    print(f"                min={embeddings.min().item():.4f}  max={embeddings.max().item():.4f}")
    print(f"  Sample [0]:   {embeddings[0, 0, :6].tolist()}")
    print(f"  Latency:      {t2.ms:.2f} ms")

    # ── STEP 3: Tensor → Binary Protocol ────────────────────────
    print()
    print("━" * 64)
    print("  STEP 3: SERIALIZE  (torch.Tensor → header + raw bytes)")
    print("━" * 64)

    with timer() as t3:
        header_bytes, payload = torch_to_wire(embeddings, DType.FP16, req_id)
    latencies["serialize"] = t3.ms

    hdr = Header.unpack(header_bytes)
    print(f"  torch.Tensor {embeddings.shape} fp16")
    print(f"    → .contiguous().numpy().tobytes()")
    print(f"    → {len(payload):,} raw bytes (little-endian IEEE 754 half-precision)")
    print(f"  Header (32 bytes):")
    print(f"    msg_type={hdr.message_type} req_id={hdr.request_id} step_id={hdr.step_id}")
    print(f"    payload_size={hdr.payload_size:,}  dtype={hdr.dtype} (fp16)")
    print(f"    dims={list(hdr.dims[:hdr.num_dims])}")
    print(f"  Message:    {len(header_bytes) + len(payload):,} bytes (32 header + {len(payload):,} payload)")
    print(f"  Latency:    {t3.ms:.2f} ms")

    # ── STEP 4: AES-256-GCM Encrypt ────────────────────────────
    print()
    print("━" * 64)
    print("  STEP 4: ENCRYPT  (header+payload → AES-256-GCM wire)")
    print("━" * 64)

    key = crypto.generate_session_key()
    message = header_bytes + payload

    with timer() as t4:
        wire = crypto.pack_for_wire(message, key)
    latencies["encrypt"] = t4.ms

    overhead = len(wire) - len(message)
    nonce = wire[4:16]
    print(f"  AES-256 key:   {key.hex()}")
    print(f"  Plaintext:     {len(message):,} bytes")
    print(f"  Nonce (96b):   {nonce.hex()}")
    print(f"  Wire format:   [4 len][12 nonce][ciphertext][16 GCM tag]")
    print(f"  Wire size:     {len(wire):,} bytes  (+{overhead} = {overhead/len(message)*100:.1f}% overhead)")
    print(f"  Cipher sample: {wire[16:48].hex()}...")
    print(f"  Latency:       {t4.ms:.2f} ms")

    # ── STEP 5: NodeA Decrypt + Reconstruct ─────────────────────
    print()
    print("━" * 64)
    print("  STEP 5: NODE A — DECRYPT & RECONSTRUCT")
    print("━" * 64)

    print(f"  [NodeA] ← {len(wire):,} encrypted bytes")

    with timer() as t5:
        dec_msg = crypto.unpack_from_wire(wire, key)
        dec_tensor, dec_hdr = wire_to_torch(dec_msg[:32], dec_msg[32:])
    latencies["node_decrypt"] = t5.ms

    print(f"  [NodeA] Decrypted: {len(dec_msg):,} bytes")
    print(f"  [NodeA] Header:    type={dec_hdr.message_type} req={dec_hdr.request_id} dims={list(dec_hdr.dims[:dec_hdr.num_dims])}")
    print(f"  [NodeA] Tensor:    {dec_tensor.shape}  dtype={dec_tensor.dtype}")
    print(f"  [NodeA] Sample:    {dec_tensor[0, 0, :6].tolist()}")
    print(f"  Latency:           {t5.ms:.2f} ms")

    # Numerical check
    print()
    print(f"  ── NUMERICAL STABILITY ──")
    match = torch.equal(embeddings, dec_tensor)
    max_diff = (embeddings.float() - dec_tensor.float()).abs().max().item()
    print(f"  torch.equal():     {'✓ PASS' if match else '✗ FAIL'}")
    print(f"  Max abs diff:      {max_diff}")
    print(f"  Shape preserved:   {'✓' if embeddings.shape == dec_tensor.shape else '✗'}")
    print(f"  Dtype preserved:   {'✓' if embeddings.dtype == dec_tensor.dtype else '✗'}")

    # ── STEP 6: NodeA Inference ─────────────────────────────────
    print()
    print("━" * 64)
    print("  STEP 6: NODE A — INFERENCE (TinyLlama)")
    print("━" * 64)

    print(f"  [NodeA] Prompt: \"{prompt}\"")
    with timer() as t6:
        answer = mdl.run_model(prompt)
    latencies["inference"] = t6.ms

    print(f"  [NodeA] Output: \"{answer}\"")
    print(f"  Latency:        {t6.ms:.2f} ms")

    # ── STEP 7: NodeA Encrypt Response ──────────────────────────
    print()
    print("━" * 64)
    print("  STEP 7: NODE A — ENCRYPT RESPONSE")
    print("━" * 64)

    resp_bytes = answer.encode("utf-8")
    resp_hdr = Header(
        message_type=MessageType.RESULT,
        request_id=req_id,
        step_id=0,
        payload_size=len(resp_bytes),
        dtype=DType.INT8,
        num_dims=1,
        dims=(len(resp_bytes), 0, 0, 0),
    )
    resp_msg = resp_hdr.pack() + resp_bytes

    with timer() as t7:
        resp_wire = crypto.pack_for_wire(resp_msg, key)
    latencies["resp_encrypt"] = t7.ms

    print(f"  [NodeA] Response:  {len(resp_bytes)} bytes")
    print(f"  [NodeA] Nonce:     {resp_wire[4:16].hex()}")
    print(f"  [NodeA] Wire:      {len(resp_wire)} bytes")
    print(f"  Latency:           {t7.ms:.2f} ms")

    # ── STEP 8: Client Decrypt ──────────────────────────────────
    print()
    print("━" * 64)
    print("  STEP 8: CLIENT — DECRYPT RESPONSE")
    print("━" * 64)

    print(f"  [Client] ← {len(resp_wire)} encrypted bytes")

    with timer() as t8:
        dec_resp = crypto.unpack_from_wire(resp_wire, key)
    latencies["resp_decrypt"] = t8.ms

    final = dec_resp[32:].decode("utf-8")
    print(f"  [Client] Answer: \"{final}\"")
    print(f"  [Client] Match:  {'✓' if final == answer else '✗'}")
    print(f"  Latency:         {t8.ms:.2f} ms")

    # ── Summary ─────────────────────────────────────────────────
    total_crypto = latencies["encrypt"] + latencies["node_decrypt"] + latencies["resp_encrypt"] + latencies["resp_decrypt"]
    total_all = sum(latencies.values())

    print()
    print("╔" + "═" * 62 + "╗")
    print("║  LATENCY BREAKDOWN                                         ║")
    print("╠" + "═" * 62 + "╣")
    for name, ms in latencies.items():
        bar = "█" * int(ms / total_all * 30) if total_all > 0 else ""
        print(f"║  {name:<16} {ms:>8.2f} ms  {bar:<30}  ║")
    print("╠" + "═" * 62 + "╣")
    print(f"║  Total crypto:   {total_crypto:>8.2f} ms                                ║")
    print(f"║  Total all:      {total_all:>8.2f} ms                                ║")
    print(f"║  Crypto overhead: {total_crypto/total_all*100 if total_all > 0 else 0:>7.1f}% of total time                      ║")
    print("╚" + "═" * 62 + "╝")


if __name__ == "__main__":
    while True:
        prompt = input("\nEnter prompt (or 'quit'): ").strip()
        if not prompt or prompt.lower() == "quit":
            break
        run(prompt)

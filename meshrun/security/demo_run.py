"""Demo run: full AES-256-GCM pipeline with verbose logging."""

import crypto
import pipeline

prompt = "What is the capital of France?"

print("=" * 60)
print("  SECURE INFERENCE PIPELINE — AES-256-GCM DEMO")
print("=" * 60)
print()

key = crypto.generate_key()
print(f"[Client] AES-256 Session Key: {key.hex()}")
print(f"[Client] Key length: {len(key) * 8} bits")
print()

prompt_bytes = prompt.encode("utf-8")
encrypted_prompt = crypto.encrypt(prompt_bytes, key)
nonce_hex = encrypted_prompt[:12].hex()
print(f"[Client] Original prompt:     {prompt}")
print(f"[Client] Plaintext bytes:     {len(prompt_bytes)} bytes")
print(f"[Client] Encrypted payload:   {len(encrypted_prompt)} bytes")
print(f"[Client] Nonce:               {nonce_hex}")
print(f"[Client] Ciphertext (hex):    {encrypted_prompt[12:].hex()[:80]}...")
print()

encrypted_response = pipeline.run(encrypted_prompt, key)

print("─" * 50)
print("  HOP 4: NodeC → Client (final)")
print("─" * 50)
resp_nonce = encrypted_response[:12].hex()
print(f"[Client] Received {len(encrypted_response)} encrypted bytes (nonce: {resp_nonce})")
print(f"[Client] Decrypting with AES-256-GCM...")

decrypted = crypto.decrypt(encrypted_response, key)
final = decrypted.decode("utf-8")
print(f"[Client] Decrypted output:    {final}")

print()
print("=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Encryption:    AES-256-GCM (authenticated)")
print(f"  Key size:      256 bits")
print(f"  Nonce:         96 bits (random per encryption)")
print(f"  Auth tag:      128 bits (GCM)")
print(f"  Total hops:    4 encrypt/decrypt cycles")
print(f"  Input:         {prompt}")
print(f"  Output:        {final}")
print("=" * 60)

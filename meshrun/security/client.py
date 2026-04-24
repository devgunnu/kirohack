"""Client CLI: Entry point for the secure inference pipeline with AES-256-GCM."""

import crypto
import pipeline


def main() -> None:
    """CLI entry point: prompt user, run pipeline, display results."""
    prompt = input("Enter your prompt: ").strip()
    if not prompt:
        print("[Client] Error: Prompt cannot be empty.")
        return

    print()
    print("=" * 60)
    print("  SECURE INFERENCE PIPELINE — AES-256-GCM")
    print("=" * 60)
    print()

    # Generate 256-bit AES session key
    key = crypto.generate_key()
    print(f"[Client] AES-256 Session Key: {key.hex()}")
    print(f"[Client] Key length: {len(key) * 8} bits")
    print()

    # Encrypt the prompt
    prompt_bytes = prompt.encode("utf-8")
    encrypted_prompt = crypto.encrypt(prompt_bytes, key)
    nonce_hex = encrypted_prompt[:12].hex()
    print(f"[Client] Original prompt:     {prompt}")
    print(f"[Client] Plaintext bytes:     {len(prompt_bytes)} bytes")
    print(f"[Client] Encrypted payload:   {len(encrypted_prompt)} bytes (12 nonce + {len(encrypted_prompt)-12-16} ciphertext + 16 tag)")
    print(f"[Client] Nonce:               {nonce_hex}")
    print(f"[Client] Ciphertext (hex):    {encrypted_prompt[12:].hex()[:80]}...")
    print()

    # Run through pipeline (encrypted at every hop)
    encrypted_response = pipeline.run(encrypted_prompt, key)

    # Decrypt final response
    print("─" * 50)
    print("  HOP 4: NodeC → Client (final)")
    print("─" * 50)
    resp_nonce = encrypted_response[:12].hex()
    print(f"[Client] Received {len(encrypted_response)} encrypted bytes (nonce: {resp_nonce})")
    print(f"[Client] Decrypting with AES-256-GCM...")

    decrypted_response = crypto.decrypt(encrypted_response, key)
    final_output = decrypted_response.decode("utf-8")
    print(f"[Client] Decrypted output:    {final_output}")

    # Summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Encryption:    AES-256-GCM (authenticated)")
    print(f"  Key size:      256 bits")
    print(f"  Nonce:         96 bits (random per encryption)")
    print(f"  Auth tag:      128 bits (GCM)")
    print(f"  Total hops:    4 (Client→A→B→C→Client)")
    print(f"  Encryptions:   4 (one per hop, fresh nonce each)")
    print(f"  Input:         {prompt}")
    print(f"  Output:        {final_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

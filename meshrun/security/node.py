"""Node module: Each node decrypts incoming data, processes it, and re-encrypts before forwarding."""

import crypto
import model


class Node:
    def __init__(self, name: str) -> None:
        self.name = name

    def process(self, encrypted_payload: bytes, key: bytes) -> bytes:
        """
        Simulate a real network hop:
        1. Receive encrypted bytes (as if from TCP socket)
        2. Decrypt with shared AES key
        3. Run inference on plaintext
        4. Re-encrypt result with fresh nonce
        5. Return encrypted bytes (as if sending over TCP to next node)

        Args:
            encrypted_payload: AES-256-GCM encrypted bytes [nonce|ciphertext|tag]
            key: 32-byte shared AES session key

        Returns:
            AES-256-GCM encrypted result bytes
        """
        # --- RECEIVE: decrypt incoming payload ---
        nonce_hex = encrypted_payload[:12].hex()
        print(f"[{self.name}] Received {len(encrypted_payload)} encrypted bytes (nonce: {nonce_hex})")
        print(f"[{self.name}] Decrypting with AES-256-GCM...")

        plaintext_bytes = crypto.decrypt(encrypted_payload, key)
        plaintext = plaintext_bytes.decode("utf-8")
        print(f"[{self.name}] Plaintext IN:  {plaintext}")

        # --- PROCESS: run inference ---
        print(f"[{self.name}] Running inference...")
        result = model.run_model(plaintext)
        print(f"[{self.name}] Plaintext OUT: {result}")

        # --- SEND: re-encrypt with fresh nonce before forwarding ---
        result_bytes = result.encode("utf-8")
        encrypted_out = crypto.encrypt(result_bytes, key)
        out_nonce_hex = encrypted_out[:12].hex()
        print(f"[{self.name}] Re-encrypted → {len(encrypted_out)} bytes (new nonce: {out_nonce_hex})")
        print()

        return encrypted_out

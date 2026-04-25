"""
Crypto module: AES-256-GCM authenticated encryption for the data plane.

Designed to wrap the TCP binary protocol from the main distributed inference spec.
Each hop encrypts the full message (32-byte header + tensor payload) before sending
over TCP, and the receiving node decrypts before processing.

Wire format per encrypted message:
  [4-byte payload_len][12-byte nonce][ciphertext][16-byte GCM auth tag]

The 4-byte length prefix enables framing over TCP streams (receiver knows exactly
how many bytes to read_exact for the encrypted blob).
"""

import os
import struct
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag


# --- Key Management ---

def generate_session_key() -> bytes:
    """Generate a random 256-bit AES session key."""
    return os.urandom(32)


def derive_key_from_password(password: str, salt: bytes = None) -> tuple[bytes, bytes]:
    """Derive AES-256 key from password via PBKDF2-HMAC-SHA256.

    Returns (key, salt). Reuse the same salt to reproduce the key.
    """
    import hashlib
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations=100_000)
    return key, salt


# --- Core AES-256-GCM Operations ---

def encrypt(plaintext: bytes, key: bytes, aad: bytes = None) -> bytes:
    """Encrypt plaintext with AES-256-GCM.

    Args:
        plaintext: Raw bytes to encrypt (e.g., header + tensor payload).
        key: 32-byte AES key.
        aad: Optional additional authenticated data (e.g., request_id for binding).

    Returns:
        Encrypted blob: [12-byte nonce][ciphertext + 16-byte tag]
    """
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext_and_tag = aesgcm.encrypt(nonce, plaintext, aad)
    return nonce + ciphertext_and_tag


def decrypt(encrypted_blob: bytes, key: bytes, aad: bytes = None) -> bytes:
    """Decrypt AES-256-GCM encrypted blob.

    Args:
        encrypted_blob: [12-byte nonce][ciphertext + 16-byte tag]
        key: 32-byte AES key.
        aad: Must match the AAD used during encryption.

    Returns:
        Decrypted plaintext bytes.

    Raises:
        cryptography.exceptions.InvalidTag: Wrong key, tampered data, or AAD mismatch.
    """
    nonce = encrypted_blob[:12]
    ciphertext_and_tag = encrypted_blob[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext_and_tag, aad)


# --- TCP Wire Format Helpers ---

def pack_for_wire(plaintext: bytes, key: bytes, aad: bytes = None) -> bytes:
    """Encrypt and frame for TCP transmission.

    Wire format: [4-byte length][encrypted_blob]
    where encrypted_blob = [12-byte nonce][ciphertext][16-byte tag]

    This mirrors how the main spec's read_exact works — the receiver reads
    4 bytes to get the length, then read_exact(length) for the encrypted payload.
    """
    encrypted_blob = encrypt(plaintext, key, aad)
    length_prefix = struct.pack("!I", len(encrypted_blob))
    return length_prefix + encrypted_blob


def unpack_from_wire(wire_data: bytes, key: bytes, aad: bytes = None) -> bytes:
    """Unframe and decrypt from TCP wire format.

    Expects: [4-byte length][encrypted_blob]
    """
    if len(wire_data) < 4:
        raise ValueError("Wire data too short: missing length prefix")
    length = struct.unpack("!I", wire_data[:4])[0]
    encrypted_blob = wire_data[4:4 + length]
    if len(encrypted_blob) != length:
        raise ValueError(f"Expected {length} bytes, got {len(encrypted_blob)}")
    return decrypt(encrypted_blob, key, aad)

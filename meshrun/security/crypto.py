"""Crypto module: AES-256-GCM authenticated encryption with random nonce per message."""

import os
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def generate_key() -> bytes:
    """Generate a random 256-bit (32-byte) AES key."""
    return os.urandom(32)


def derive_key(password: str, salt: bytes = None) -> tuple[bytes, bytes]:
    """Derive a 256-bit AES key from a password using PBKDF2.

    Returns:
        (key, salt) tuple. Pass the same salt to reproduce the key.
    """
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=100_000)
    return key, salt


def encrypt(data: bytes, key: bytes) -> bytes:
    """Encrypt data using AES-256-GCM.

    Wire format: [12-byte nonce][ciphertext+tag]
    The 16-byte GCM auth tag is appended to the ciphertext by AESGCM.

    Args:
        data: Plaintext bytes to encrypt.
        key: 32-byte AES key.

    Returns:
        Encrypted payload: nonce || ciphertext || tag
    """
    nonce = os.urandom(12)  # 96-bit nonce, standard for GCM
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, data, None)  # ciphertext includes 16-byte tag
    return nonce + ciphertext


def decrypt(data: bytes, key: bytes) -> bytes:
    """Decrypt AES-256-GCM encrypted data.

    Expects wire format: [12-byte nonce][ciphertext+tag]

    Args:
        data: Encrypted payload (nonce || ciphertext || tag).
        key: 32-byte AES key.

    Returns:
        Decrypted plaintext bytes.

    Raises:
        cryptography.exceptions.InvalidTag: If key is wrong or data is tampered.
    """
    nonce = data[:12]
    ciphertext = data[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)

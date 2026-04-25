"""Key Manager for AES-256 session key lifecycle.

Generates, stores, retrieves, rotates, and deletes per-pipeline
AES-256 session keys used for encrypted data-plane traffic.
"""

from __future__ import annotations

import os
import threading


class KeyManager:
    """Thread-safe store of ``model_id → session_key`` mappings.

    All public methods acquire an internal lock so the manager is safe
    to use from the gRPC servicer threads and the health-check thread
    concurrently.
    """

    _KEY_SIZE_BYTES: int = 32  # AES-256

    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_pipeline_key(self, model_id: str) -> bytes:
        """Create a 32-byte AES-256 key for *model_id* and store it.

        Returns the newly generated key.
        """
        key = os.urandom(self._KEY_SIZE_BYTES)
        with self._lock:
            self._keys[model_id] = key
        return key

    def get_pipeline_key(self, model_id: str) -> bytes | None:
        """Return the stored key for *model_id*, or ``None``."""
        with self._lock:
            return self._keys.get(model_id)

    def rotate_key(self, model_id: str) -> bytes:
        """Replace the key for *model_id* with a fresh one.

        Returns the new key.
        """
        key = os.urandom(self._KEY_SIZE_BYTES)
        with self._lock:
            self._keys[model_id] = key
        return key

    def delete_key(self, model_id: str) -> bool:
        """Remove the key for *model_id*.

        Returns ``True`` if a key existed and was deleted.
        """
        with self._lock:
            return self._keys.pop(model_id, None) is not None

"""
Layer Assignment Registry — stores the current layer assignment for a worker node.

Provides lookup for pipeline routing decisions: which layers this node hosts,
the downstream node to forward results to, and the upstream nodes that may
send data to this node.

Validates: Requirements 6.1, 6.2
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ── Enumerations ─────────────────────────────────────────────────────────────

class AssignmentDType(IntEnum):
    """Quantization formats for the assigned shard."""
    FP16 = 1  # 2 bytes per element (IEEE 754 half-precision)
    INT8 = 2  # 1 byte per element  (signed 8-bit integer)


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class LayerAssignment:
    """Immutable snapshot of a layer assignment received from the Coordinator.

    Captures everything a worker node needs to know about its role in the
    inference pipeline: which model and layers it hosts, the quantization
    format, its position in the pipeline topology (downstream / upstream
    neighbours), and whether it is the final node that produces logits.
    """

    node_id: str
    """This node's unique identifier (UUID string)."""

    model_id: str
    """Identifier of the model being served (e.g. ``'llama-3b'``)."""

    model_url: str
    """HTTP URL to the safetensors model file."""

    layer_start: int
    """First assigned layer index (inclusive)."""

    layer_end: int
    """Last assigned layer index (inclusive)."""

    dtype: AssignmentDType
    """Quantization format for this shard (fp16 or int8)."""

    is_final_node: bool
    """Whether this node hosts the last layers and should produce logits."""

    downstream_node: Optional[str] = None
    """TCP ``host:port`` of the next node in the pipeline.

    ``None`` when ``is_final_node`` is ``True``.
    """

    upstream_nodes: tuple[str, ...] = ()
    """TCP ``host:port`` addresses that may send data to this node."""

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be a non-empty string")
        if not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if not self.model_url:
            raise ValueError("model_url must be a non-empty string")
        if self.layer_start < 0:
            raise ValueError(
                f"layer_start must be >= 0, got {self.layer_start}"
            )
        if self.layer_end < self.layer_start:
            raise ValueError(
                f"layer_end ({self.layer_end}) must be >= "
                f"layer_start ({self.layer_start})"
            )
        if not isinstance(self.dtype, AssignmentDType):
            raise ValueError(
                f"dtype must be an AssignmentDType, got {self.dtype!r}"
            )
        if self.is_final_node and self.downstream_node is not None:
            raise ValueError(
                "downstream_node must be None when is_final_node is True"
            )
        if not self.is_final_node and not self.downstream_node:
            raise ValueError(
                "downstream_node is required when is_final_node is False"
            )

    @property
    def layer_count(self) -> int:
        """Number of layers in this assignment."""
        return self.layer_end - self.layer_start + 1


# ── Registry ────────────────────────────────────────────────────────────────

class LayerAssignmentRegistry:
    """Thread-safe registry that stores the current layer assignment for a worker node.

    The Coordinator sends an ``AcceptLayerAssignment`` message containing the
    full assignment details.  The registry stores exactly one assignment at a
    time — accepting a new assignment replaces any previous one.

    All public methods are safe to call from multiple threads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._assignment: Optional[LayerAssignment] = None

    # ── Mutators ────────────────────────────────────────────────────────

    def accept_layer_assignment(self, assignment: LayerAssignment) -> None:
        """Store a new layer assignment received from the Coordinator.

        Parameters
        ----------
        assignment:
            A validated ``LayerAssignment`` instance.  The dataclass
            ``__post_init__`` ensures all invariants hold before storage.

        Raises
        ------
        TypeError
            If *assignment* is not a ``LayerAssignment``.
        """
        if not isinstance(assignment, LayerAssignment):
            raise TypeError(
                f"expected LayerAssignment, got {type(assignment).__name__}"
            )
        with self._lock:
            self._assignment = assignment

    def clear(self) -> None:
        """Remove the current assignment (e.g. on shard unload or shutdown)."""
        with self._lock:
            self._assignment = None

    # ── Queries ─────────────────────────────────────────────────────────

    @property
    def has_assignment(self) -> bool:
        """Return ``True`` if an assignment is currently stored."""
        with self._lock:
            return self._assignment is not None

    @property
    def assignment(self) -> Optional[LayerAssignment]:
        """Return the current assignment snapshot, or ``None``."""
        with self._lock:
            return self._assignment

    # ── Convenience query methods for sub-components ────────────────────

    def get_downstream_address(self) -> Optional[str]:
        """Return the downstream node's ``host:port``, or ``None`` if final node or unassigned."""
        with self._lock:
            if self._assignment is None:
                return None
            return self._assignment.downstream_node

    def get_upstream_addresses(self) -> tuple[str, ...]:
        """Return upstream node addresses. Empty tuple if unassigned."""
        with self._lock:
            if self._assignment is None:
                return ()
            return self._assignment.upstream_nodes

    def get_layer_range(self) -> Optional[tuple[int, int]]:
        """Return ``(layer_start, layer_end)`` inclusive, or ``None`` if unassigned."""
        with self._lock:
            if self._assignment is None:
                return None
            return (self._assignment.layer_start, self._assignment.layer_end)

    def get_dtype(self) -> Optional[AssignmentDType]:
        """Return the assigned quantization dtype, or ``None`` if unassigned."""
        with self._lock:
            if self._assignment is None:
                return None
            return self._assignment.dtype

    def is_final_node(self) -> bool:
        """Return ``True`` if this node is the final node in the pipeline.

        Returns ``False`` if no assignment is stored.
        """
        with self._lock:
            if self._assignment is None:
                return False
            return self._assignment.is_final_node

    def get_model_id(self) -> Optional[str]:
        """Return the model identifier, or ``None`` if unassigned."""
        with self._lock:
            if self._assignment is None:
                return None
            return self._assignment.model_id

    def get_model_url(self) -> Optional[str]:
        """Return the HTTP URL to the safetensors model file, or ``None`` if unassigned."""
        with self._lock:
            if self._assignment is None:
                return None
            return self._assignment.model_url

    def get_node_id(self) -> Optional[str]:
        """Return this node's identifier, or ``None`` if unassigned."""
        with self._lock:
            if self._assignment is None:
                return None
            return self._assignment.node_id

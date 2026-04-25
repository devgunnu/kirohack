"""Scheduler — Layer Assignment Engine, Route Building, and Priority Queue.

Computes static layer assignments at cluster startup, builds execution
routes for inference requests, and manages the priority queue for request
scheduling.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meshrun.coordinator.key_manager import KeyManager
    from meshrun.coordinator.registry import NodeEntry, NodeRegistry, NodeStatus

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

FRAMEWORK_OVERHEAD_MB: int = 800
"""Memory reserved for PyTorch/CUDA framework overhead per node."""

LAYER_MEM_FP16_MB: int = 200
"""Estimated per-layer memory in MB for fp16 (~3B param model)."""

LAYER_MEM_INT8_MB: int = 100
"""Estimated per-layer memory in MB for int8 (~3B param model)."""


# ── Layer Map Entry ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LayerMapEntry:
    """Immutable record mapping a contiguous layer range to primary and backup nodes.

    Each entry represents one segment of the model pipeline.  The primary
    node is responsible for executing the layers during normal operation;
    the backup node (if any) can take over when the primary is unhealthy.
    """

    layer_start: int
    """First layer index (inclusive) in this range."""

    layer_end: int
    """Last layer index (inclusive) in this range."""

    primary_node_id: str
    """Identifier of the primary worker node for this range."""

    primary_address: str
    """TCP ``host:port`` of the primary node's data-plane listener."""

    backup_node_id: str | None = None
    """Identifier of the backup worker node, or ``None`` if no backup."""

    backup_address: str | None = None
    """TCP ``host:port`` of the backup node, or ``None`` if no backup."""


# ── Layer Map ────────────────────────────────────────────────────────────────


class LayerMap:
    """Stores an ordered list of :class:`LayerMapEntry` and provides fast lookups.

    The layer map is the authoritative mapping from layer indices to the
    nodes responsible for executing them.  It is populated after
    :func:`compute_assignments` completes and is consulted during route
    building and failure handling.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: list[LayerMapEntry] | None = None) -> None:
        self._entries: list[LayerMapEntry] = list(entries) if entries else []

    # ── Lookup methods ───────────────────────────────────────────────

    def get_primary_node_for_layer(self, layer_index: int) -> LayerMapEntry | None:
        """Return the entry whose primary node covers *layer_index*, or ``None``."""
        for entry in self._entries:
            if entry.layer_start <= layer_index <= entry.layer_end:
                return entry
        return None

    def get_backup_for_range(
        self, layer_start: int, layer_end: int
    ) -> LayerMapEntry | None:
        """Return the entry matching the exact range that has a backup node.

        Returns ``None`` if no entry matches the range or the matching
        entry has no backup assigned.
        """
        for entry in self._entries:
            if entry.layer_start == layer_start and entry.layer_end == layer_end:
                if entry.backup_node_id is not None:
                    return entry
                return None
        return None

    def get_all_entries(self) -> list[LayerMapEntry]:
        """Return a copy of all layer map entries (ascending by layer_start)."""
        return list(self._entries)

    # ── Mutation ─────────────────────────────────────────────────────

    def set_entries(self, entries: list[LayerMapEntry]) -> None:
        """Replace all entries with *entries*."""
        self._entries = list(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"LayerMap(entries={len(self._entries)})"


# ── Assignment Plan ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AssignmentPlan:
    """Result of a layer assignment computation.

    Contains the ordered list of :class:`LayerMapEntry` objects covering
    every layer in the model, plus the AES-256 session key generated for
    this pipeline.
    """

    assignments: tuple[LayerMapEntry, ...]
    """Ordered layer-range assignments (ascending by ``layer_start``)."""

    session_key: bytes
    """32-byte AES-256-GCM session key for this pipeline."""



# ── Exceptions ───────────────────────────────────────────────────────────────


class InsufficientCapacityError(Exception):
    """Raised when the cluster cannot cover all model layers."""


# ── Layer Assignment Algorithm ───────────────────────────────────────────────


def compute_assignments(
    model_id: str,
    total_layers: int,
    dtype: str,
    nodes: list[NodeEntry],
    key_manager: KeyManager,
) -> AssignmentPlan:
    """Compute greedy contiguous layer assignments across *nodes*.

    The algorithm:
    1. Determine per-layer memory cost from *dtype* (``"fp16"`` or ``"int8"``).
    2. For each node, compute usable memory by subtracting
       :data:`FRAMEWORK_OVERHEAD_MB` from ``memory_limit_mb``.
    3. Sort nodes by usable memory descending (largest capacity first).
    4. Greedily assign contiguous layer blocks: each node receives
       ``min(max_layers_it_can_hold, remaining_unassigned_layers)`` layers.
    5. Validate that every layer ``[0, total_layers)`` is covered.
    6. Generate a pipeline session key via *key_manager*.

    Parameters
    ----------
    model_id:
        Identifier for the model pipeline.
    total_layers:
        Number of transformer layers in the model.
    dtype:
        Weight data type — ``"fp16"`` or ``"int8"``.
    nodes:
        List of :class:`~meshrun.coordinator.registry.NodeEntry` objects
        representing the available worker nodes.  Only nodes with positive
        usable memory after overhead subtraction are considered.
    key_manager:
        :class:`~meshrun.coordinator.key_manager.KeyManager` used to
        generate the pipeline session key.

    Returns
    -------
    AssignmentPlan
        The ordered layer assignments and the generated session key.

    Raises
    ------
    ValueError
        If *total_layers* < 1, *dtype* is unrecognised, or *nodes* is empty.
    InsufficientCapacityError
        If the combined node capacity cannot cover all layers.
    """
    # ── Input validation ─────────────────────────────────────────────
    if total_layers < 1:
        raise ValueError(f"total_layers must be >= 1, got {total_layers}")

    dtype_lower = dtype.lower()
    if dtype_lower == "fp16":
        layer_mem_mb = LAYER_MEM_FP16_MB
    elif dtype_lower == "int8":
        layer_mem_mb = LAYER_MEM_INT8_MB
    else:
        raise ValueError(f"Unsupported dtype '{dtype}'; expected 'fp16' or 'int8'")

    if not nodes:
        raise ValueError("No nodes provided for layer assignment")

    # ── Compute usable memory per node ───────────────────────────────
    node_capacities: list[tuple[NodeEntry, int]] = []
    for node in nodes:
        usable = node.memory_limit_mb - FRAMEWORK_OVERHEAD_MB
        if usable > 0:
            node_capacities.append((node, usable))

    if not node_capacities:
        raise InsufficientCapacityError(
            "No nodes have usable memory after subtracting "
            f"{FRAMEWORK_OVERHEAD_MB}MB framework overhead"
        )

    # Sort by usable memory descending (largest first)
    node_capacities.sort(key=lambda pair: pair[1], reverse=True)

    # ── Greedy contiguous assignment ─────────────────────────────────
    assignments: list[LayerMapEntry] = []
    next_layer = 0

    for node, usable in node_capacities:
        if next_layer >= total_layers:
            break

        max_layers = math.floor(usable / layer_mem_mb)
        if max_layers < 1:
            continue

        layers_to_assign = min(max_layers, total_layers - next_layer)
        layer_start = next_layer
        layer_end = next_layer + layers_to_assign - 1  # inclusive

        assignments.append(
            LayerMapEntry(
                layer_start=layer_start,
                layer_end=layer_end,
                primary_node_id=node.node_id,
                primary_address=node.address,
            )
        )

        logger.info(
            "Assigned layers %d–%d to node %s (%dMB usable, %d layers)",
            layer_start,
            layer_end,
            node.node_id,
            usable,
            layers_to_assign,
        )

        next_layer = layer_end + 1

    # ── Validate full coverage ───────────────────────────────────────
    if next_layer < total_layers:
        raise InsufficientCapacityError(
            f"Could only assign layers 0–{next_layer - 1} of {total_layers}; "
            f"cluster capacity is insufficient "
            f"({total_layers - next_layer} layers unassigned)"
        )

    # ── Backup node assignment ───────────────────────────────────────
    # Track how many layers each node is already committed to (primary).
    primary_layer_counts: dict[str, int] = {}
    for entry in assignments:
        count = entry.layer_end - entry.layer_start + 1
        primary_layer_counts[entry.primary_node_id] = (
            primary_layer_counts.get(entry.primary_node_id, 0) + count
        )

    # Build a lookup of node_id → (NodeEntry, usable_memory) for all
    # nodes that passed the usable-memory filter.
    node_lookup: dict[str, tuple[NodeEntry, int]] = {
        node.node_id: (node, usable) for node, usable in node_capacities
    }

    # Track additional backup layers assigned to each node so we don't
    # over-commit a single node as backup for multiple ranges.
    backup_layer_counts: dict[str, int] = {}

    assignments_with_backups: list[LayerMapEntry] = []
    for entry in assignments:
        range_size = entry.layer_end - entry.layer_start + 1
        range_mem_needed = range_size * layer_mem_mb

        backup_node_id: str | None = None
        backup_address: str | None = None

        for candidate_id, (candidate, usable) in node_lookup.items():
            if candidate_id == entry.primary_node_id:
                continue

            # Spare capacity = usable memory minus primary commitment
            # minus any backup layers already assigned to this candidate.
            primary_committed = (
                primary_layer_counts.get(candidate_id, 0) * layer_mem_mb
            )
            backup_committed = (
                backup_layer_counts.get(candidate_id, 0) * layer_mem_mb
            )
            spare = usable - primary_committed - backup_committed

            if spare >= range_mem_needed:
                backup_node_id = candidate_id
                backup_address = candidate.address
                backup_layer_counts[candidate_id] = (
                    backup_layer_counts.get(candidate_id, 0) + range_size
                )
                logger.info(
                    "Backup for layers %d–%d: node %s (%dMB spare)",
                    entry.layer_start,
                    entry.layer_end,
                    candidate_id,
                    spare,
                )
                break

        if backup_node_id is None:
            logger.warning(
                "No backup node available for layers %d–%d (primary: %s)",
                entry.layer_start,
                entry.layer_end,
                entry.primary_node_id,
            )

        assignments_with_backups.append(
            LayerMapEntry(
                layer_start=entry.layer_start,
                layer_end=entry.layer_end,
                primary_node_id=entry.primary_node_id,
                primary_address=entry.primary_address,
                backup_node_id=backup_node_id,
                backup_address=backup_address,
            )
        )

    # ── Generate session key ─────────────────────────────────────────
    session_key = key_manager.generate_pipeline_key(model_id)

    plan = AssignmentPlan(
        assignments=tuple(assignments_with_backups),
        session_key=session_key,
    )

    logger.info(
        "Assignment plan for model '%s': %d segments covering %d layers",
        model_id,
        len(assignments_with_backups),
        total_layers,
    )

    return plan


# ── Route Building ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RouteNode:
    """A single node in an execution route.

    Mirrors the protobuf ``RouteNode`` message for use in pure-Python
    scheduler logic.
    """

    node_id: str
    """Identifier of the worker node."""

    address: str
    """TCP ``host:port`` of the node's data-plane listener."""

    layer_start: int
    """First layer index (inclusive) handled by this node."""

    layer_end: int
    """Last layer index (inclusive) handled by this node."""


@dataclass(frozen=True, slots=True)
class ExecutionPath:
    """An ordered execution route through the pipeline.

    Returned by :func:`build_route` and consumed by the Inference Client
    and the gRPC ``RequestRoute`` handler.
    """

    request_id: str
    """Unique identifier for this inference request."""

    model_id: str
    """Model pipeline this route was built for."""

    session_key: bytes
    """32-byte AES-256-GCM session key for encrypting data-plane traffic."""

    nodes: tuple[RouteNode, ...]
    """Ordered list of pipeline nodes (ascending by layer index)."""

    backup_map: dict[str, str]
    """Mapping of ``primary_node_id → backup_address`` for rerouting."""


@dataclass(frozen=True, slots=True)
class RerouteInfo:
    """Backup routing information for a failed node.

    Returned by :func:`handle_failure` and translated to the protobuf
    ``RerouteInfo`` message by the gRPC servicer.
    """

    backup_addr: str | None = None
    """TCP ``host:port`` of the backup node, or ``None`` if no backup available."""

    message: str = ""
    """Human-readable description of the reroute outcome."""


class RouteError(Exception):
    """Raised when a valid execution route cannot be constructed."""


def build_route(
    model_id: str,
    layer_map: LayerMap,
    registry: NodeRegistry,
    key_manager: KeyManager,
) -> ExecutionPath:
    """Construct an :class:`ExecutionPath` for *model_id*.

    Iterates the layer map entries in order.  For each entry the primary
    node is selected when it is HEALTHY; otherwise the backup node is
    used.  If neither is available the function raises :class:`RouteError`.

    Parameters
    ----------
    model_id:
        Identifier of the model pipeline to route.
    layer_map:
        The :class:`LayerMap` produced by :func:`compute_assignments`.
    registry:
        The live :class:`~meshrun.coordinator.registry.NodeRegistry` used
        to check node health.
    key_manager:
        The :class:`~meshrun.coordinator.key_manager.KeyManager` holding
        the pipeline session key.

    Returns
    -------
    ExecutionPath
        The ordered route with session key and backup map.

    Raises
    ------
    RouteError
        If the layer map is empty, the session key is missing, or any
        layer range has no healthy primary or backup node.
    """
    from meshrun.coordinator.registry import NodeStatus

    entries = layer_map.get_all_entries()
    if not entries:
        raise RouteError(
            f"Layer map for model '{model_id}' is empty — "
            "cannot build a route with no layer assignments"
        )

    session_key = key_manager.get_pipeline_key(model_id)
    if session_key is None:
        raise RouteError(
            f"No session key found for model '{model_id}' — "
            "has compute_assignments been run?"
        )

    route_nodes: list[RouteNode] = []
    backup_map: dict[str, str] = {}

    for entry in entries:
        # Try primary node first
        primary = registry.get_node(entry.primary_node_id)
        if primary is not None and primary.status == NodeStatus.HEALTHY:
            route_nodes.append(
                RouteNode(
                    node_id=entry.primary_node_id,
                    address=entry.primary_address,
                    layer_start=entry.layer_start,
                    layer_end=entry.layer_end,
                )
            )
            # Record backup in the map if one exists
            if entry.backup_node_id is not None and entry.backup_address is not None:
                backup_map[entry.primary_node_id] = entry.backup_address
            continue

        # Primary unhealthy — try backup
        if entry.backup_node_id is not None and entry.backup_address is not None:
            backup = registry.get_node(entry.backup_node_id)
            if backup is not None and backup.status == NodeStatus.HEALTHY:
                route_nodes.append(
                    RouteNode(
                        node_id=entry.backup_node_id,
                        address=entry.backup_address,
                        layer_start=entry.layer_start,
                        layer_end=entry.layer_end,
                    )
                )
                logger.warning(
                    "Using backup node %s for layers %d–%d "
                    "(primary %s is unavailable)",
                    entry.backup_node_id,
                    entry.layer_start,
                    entry.layer_end,
                    entry.primary_node_id,
                )
                continue

        # Neither primary nor backup is healthy
        raise RouteError(
            f"No healthy node available for layers {entry.layer_start}–"
            f"{entry.layer_end} (primary={entry.primary_node_id}, "
            f"backup={entry.backup_node_id})"
        )

    request_id = uuid.uuid4().hex

    path = ExecutionPath(
        request_id=request_id,
        model_id=model_id,
        session_key=session_key,
        nodes=tuple(route_nodes),
        backup_map=backup_map,
    )

    logger.info(
        "Built route for model '%s': request_id=%s, %d nodes",
        model_id,
        request_id,
        len(route_nodes),
    )

    return path


# ── Failure Handling ─────────────────────────────────────────────────────────


def handle_failure(
    request_id: str,
    failed_node_id: str,
    layer_map: LayerMap,
    registry: NodeRegistry,
) -> RerouteInfo:
    """Look up a backup node for *failed_node_id* and return reroute info.

    Searches the layer map for entries where *failed_node_id* is the
    primary node.  If a backup node exists for that range and is HEALTHY,
    returns a :class:`RerouteInfo` with the backup address.  Otherwise
    returns a ``RerouteInfo`` with ``backup_addr=None``.

    Parameters
    ----------
    request_id:
        Identifier of the inference request that encountered the failure.
    failed_node_id:
        The ``node_id`` of the node that failed.
    layer_map:
        The :class:`LayerMap` produced by :func:`compute_assignments`.
    registry:
        The live :class:`~meshrun.coordinator.registry.NodeRegistry` used
        to check backup node health.

    Returns
    -------
    RerouteInfo
        Contains the backup node's TCP address if available and healthy,
        or ``backup_addr=None`` if no viable backup exists.
    """
    from meshrun.coordinator.registry import NodeStatus

    entries = layer_map.get_all_entries()

    # Find the layer map entry where the failed node is the primary.
    failed_entry: LayerMapEntry | None = None
    for entry in entries:
        if entry.primary_node_id == failed_node_id:
            failed_entry = entry
            break

    if failed_entry is None:
        logger.warning(
            "handle_failure: node %s not found as primary in layer map "
            "(request_id=%s)",
            failed_node_id,
            request_id,
        )
        return RerouteInfo(
            backup_addr=None,
            message=f"Node {failed_node_id} not found in layer map",
        )

    # Check if a backup exists for this range.
    if failed_entry.backup_node_id is None or failed_entry.backup_address is None:
        logger.warning(
            "handle_failure: no backup assigned for layers %d–%d "
            "(primary=%s, request_id=%s)",
            failed_entry.layer_start,
            failed_entry.layer_end,
            failed_node_id,
            request_id,
        )
        return RerouteInfo(
            backup_addr=None,
            message=(
                f"No backup node assigned for layers "
                f"{failed_entry.layer_start}–{failed_entry.layer_end}"
            ),
        )

    # Verify the backup node is healthy.
    backup = registry.get_node(failed_entry.backup_node_id)
    if backup is None or backup.status != NodeStatus.HEALTHY:
        status_desc = backup.status.name if backup is not None else "NOT_FOUND"
        logger.warning(
            "handle_failure: backup node %s is %s for layers %d–%d "
            "(request_id=%s)",
            failed_entry.backup_node_id,
            status_desc,
            failed_entry.layer_start,
            failed_entry.layer_end,
            request_id,
        )
        return RerouteInfo(
            backup_addr=None,
            message=(
                f"Backup node {failed_entry.backup_node_id} is {status_desc} "
                f"for layers {failed_entry.layer_start}–{failed_entry.layer_end}"
            ),
        )

    logger.info(
        "handle_failure: rerouting layers %d–%d to backup node %s at %s "
        "(request_id=%s)",
        failed_entry.layer_start,
        failed_entry.layer_end,
        failed_entry.backup_node_id,
        failed_entry.backup_address,
        request_id,
    )

    return RerouteInfo(
        backup_addr=failed_entry.backup_address,
        message=(
            f"Rerouted layers {failed_entry.layer_start}–{failed_entry.layer_end} "
            f"to backup node {failed_entry.backup_node_id}"
        ),
    )


# ── Priority Queue ───────────────────────────────────────────────────────────

DEFAULT_ALPHA: float = 0.7
"""Weight for compute_contributed in the priority scoring function."""

DEFAULT_BETA: float = 0.3
"""Weight for wait_time (seconds) in the priority scoring function."""

DEFAULT_MAX_DEPTH: int = 100
"""Maximum number of entries the priority queue will hold."""


@dataclass(slots=True)
class QueueEntry:
    """A single entry in the inference request priority queue.

    Attributes are mutable so the scheduler can update the computed
    ``priority`` score at dequeue time without creating new objects.
    """

    request_id: str
    """Unique identifier for the inference request."""

    client_id: str
    """Identifier of the client that submitted the request."""

    model_id: str
    """Model pipeline the request targets."""

    compute_contributed: float
    """Opaque measure of how much compute the client has contributed."""

    enqueued_at: float
    """Monotonic timestamp (``time.monotonic()``) when the entry was enqueued."""

    priority: float = 0.0
    """Last-computed priority score (higher = more urgent)."""


class QueueFullError(Exception):
    """Raised when the priority queue has reached its maximum depth."""


class PriorityQueue:
    """Priority queue for inference request scheduling.

    Entries are scored at dequeue time using::

        priority = α × compute_contributed + β × wait_time

    where *wait_time* is ``now − enqueued_at`` in seconds.  Re-scoring at
    dequeue ensures that entries that have been waiting longer naturally
    rise in priority.

    Parameters
    ----------
    max_depth:
        Maximum number of entries.  :meth:`enqueue` raises
        :class:`QueueFullError` when the limit is reached.
    alpha:
        Weight for ``compute_contributed`` in the scoring function.
    beta:
        Weight for ``wait_time`` in the scoring function.
    """

    __slots__ = ("_entries", "_max_depth", "_alpha", "_beta")

    def __init__(
        self,
        max_depth: int = DEFAULT_MAX_DEPTH,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
    ) -> None:
        self._entries: list[QueueEntry] = []
        self._max_depth = max_depth
        self._alpha = alpha
        self._beta = beta

    # ── Public API ───────────────────────────────────────────────────

    def enqueue(
        self,
        request_id: str,
        client_id: str,
        model_id: str,
        compute_contributed: float,
    ) -> QueueEntry:
        """Add a new inference request to the queue.

        Parameters
        ----------
        request_id:
            Unique request identifier.
        client_id:
            Identifier of the submitting client.
        model_id:
            Target model pipeline.
        compute_contributed:
            Client's compute contribution metric.

        Returns
        -------
        QueueEntry
            The newly created queue entry.

        Raises
        ------
        QueueFullError
            If the queue has reached *max_depth*.
        """
        import time

        if len(self._entries) >= self._max_depth:
            raise QueueFullError(
                f"Priority queue is full ({self._max_depth} entries); "
                "request rejected"
            )

        entry = QueueEntry(
            request_id=request_id,
            client_id=client_id,
            model_id=model_id,
            compute_contributed=compute_contributed,
            enqueued_at=time.monotonic(),
        )
        self._entries.append(entry)

        logger.debug(
            "Enqueued request %s from client %s (model=%s, compute=%.2f)",
            request_id,
            client_id,
            model_id,
            compute_contributed,
        )

        return entry

    def dequeue(self) -> QueueEntry | None:
        """Remove and return the highest-priority entry, or ``None`` if empty.

        All entries are re-scored before selection so that accumulated
        wait time is reflected in the priority.
        """
        import time

        if not self._entries:
            return None

        now = time.monotonic()

        # Re-score every entry
        for entry in self._entries:
            wait_time = now - entry.enqueued_at
            entry.priority = (
                self._alpha * entry.compute_contributed + self._beta * wait_time
            )

        # Find the entry with the highest priority
        best_idx = 0
        for i in range(1, len(self._entries)):
            if self._entries[i].priority > self._entries[best_idx].priority:
                best_idx = i

        best = self._entries.pop(best_idx)

        logger.debug(
            "Dequeued request %s (priority=%.4f, waited=%.2fs)",
            best.request_id,
            best.priority,
            now - best.enqueued_at,
        )

        return best

    # ── Introspection ────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of entries currently in the queue."""
        return len(self._entries)

    @property
    def is_full(self) -> bool:
        """Return ``True`` if the queue has reached its maximum depth."""
        return len(self._entries) >= self._max_depth

    def __repr__(self) -> str:
        return (
            f"PriorityQueue(size={len(self._entries)}, "
            f"max_depth={self._max_depth})"
        )

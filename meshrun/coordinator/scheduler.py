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

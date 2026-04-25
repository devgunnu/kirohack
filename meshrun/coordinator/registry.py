"""Node Registry and Health Tracker for the Coordinator control plane.

Maintains the live registry of all worker nodes and tracks their health
status via heartbeat monitoring.  This is the single source of truth for
which nodes are available and healthy.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum, auto

logger = logging.getLogger(__name__)


# ── Node Status ──────────────────────────────────────────────────────────────


class NodeStatus(IntEnum):
    """Lifecycle status of a worker node in the registry."""

    REGISTERED = auto()
    HEALTHY = auto()
    UNHEALTHY = auto()
    DEAD = auto()
    DEREGISTERED = auto()


# ── Valid Transitions ────────────────────────────────────────────────────────

_VALID_TRANSITIONS: dict[NodeStatus, frozenset[NodeStatus]] = {
    NodeStatus.REGISTERED: frozenset({NodeStatus.HEALTHY}),
    NodeStatus.HEALTHY: frozenset(
        {NodeStatus.UNHEALTHY, NodeStatus.DEREGISTERED}
    ),
    NodeStatus.UNHEALTHY: frozenset({NodeStatus.HEALTHY, NodeStatus.DEAD}),
    NodeStatus.DEAD: frozenset(),
    NodeStatus.DEREGISTERED: frozenset(),
}


# ── Node Entry ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class NodeEntry:
    """Mutable record for a single worker node in the registry.

    Capacity fields mirror :class:`~meshrun.worker.coordinator_client.CapacityInfo`
    so the registry is self-contained and does not depend on the worker package.
    """

    # Identity
    node_id: str
    address: str
    """TCP ``host:port`` for the data-plane listener."""
    grpc_address: str
    """gRPC ``host:port`` for control-plane callbacks."""

    # Capacity (set at registration time)
    gpu_memory_total_mb: int
    gpu_memory_free_mb: int
    memory_limit_mb: int
    gpu_utilization: float

    # Status
    status: NodeStatus = NodeStatus.REGISTERED

    # Layer assignment (populated after compute_assignments)
    layer_start: int | None = None
    layer_end: int | None = None

    # Timestamps
    last_seen: float = field(default_factory=time.monotonic)
    registered_at: float = field(default_factory=time.monotonic)

    # Dynamic metrics (updated via heartbeats)
    memory_used_mb: int = 0
    active_requests: int = 0


# ── Node Registry ────────────────────────────────────────────────────────────


class DuplicateNodeError(Exception):
    """Raised when a node with the same ``node_id`` is already registered."""


class InvalidTransitionError(Exception):
    """Raised when a node status transition is not allowed."""


class NodeRegistry:
    """Thread-safe registry of all worker nodes known to the Coordinator.

    Every mutation acquires ``_lock`` so the registry is safe for concurrent
    access from the gRPC servicer threads and the health-check background
    thread.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NodeEntry] = {}
        self._lock = threading.Lock()

    # ── Registration ─────────────────────────────────────────────────────

    def register_node(
        self,
        node_id: str,
        address: str,
        grpc_address: str,
        gpu_memory_total_mb: int,
        gpu_memory_free_mb: int,
        memory_limit_mb: int,
        gpu_utilization: float,
    ) -> NodeEntry:
        """Add a worker node to the registry with status REGISTERED.

        Parameters
        ----------
        node_id:
            Unique identifier for the worker node.
        address:
            TCP ``host:port`` for the data-plane listener.
        grpc_address:
            gRPC ``host:port`` for control-plane callbacks.
        gpu_memory_total_mb, gpu_memory_free_mb, memory_limit_mb:
            GPU capacity fields reported by the worker.
        gpu_utilization:
            Current GPU utilisation percentage (0.0–1.0).

        Returns
        -------
        NodeEntry
            The newly created registry entry.

        Raises
        ------
        DuplicateNodeError
            If a node with the same *node_id* is already present.
        """
        with self._lock:
            if node_id in self._nodes:
                raise DuplicateNodeError(
                    f"Node '{node_id}' is already registered"
                )

            now = time.monotonic()
            entry = NodeEntry(
                node_id=node_id,
                address=address,
                grpc_address=grpc_address,
                gpu_memory_total_mb=gpu_memory_total_mb,
                gpu_memory_free_mb=gpu_memory_free_mb,
                memory_limit_mb=memory_limit_mb,
                gpu_utilization=gpu_utilization,
                status=NodeStatus.REGISTERED,
                last_seen=now,
                registered_at=now,
            )
            self._nodes[node_id] = entry
            logger.info(
                "Registered node '%s' at %s (status=REGISTERED)",
                node_id,
                address,
            )
            return entry

    # ── Deregistration ───────────────────────────────────────────────────

    def deregister_node(self, node_id: str) -> bool:
        """Remove a worker node from the registry.

        If the node is currently HEALTHY it is first transitioned to
        DEREGISTERED before removal so the status lifecycle is respected.

        Parameters
        ----------
        node_id:
            Identifier of the node to remove.

        Returns
        -------
        bool
            ``True`` if the node existed and was removed, ``False`` otherwise.
        """
        with self._lock:
            entry = self._nodes.get(node_id)
            if entry is None:
                logger.warning(
                    "Deregister requested for unknown node '%s'", node_id
                )
                return False

            # Respect the status lifecycle — only HEALTHY nodes may
            # transition to DEREGISTERED.  DEAD / already-DEREGISTERED
            # nodes are simply removed.
            if entry.status == NodeStatus.HEALTHY:
                entry.status = NodeStatus.DEREGISTERED
                logger.info(
                    "Node '%s' transitioned HEALTHY → DEREGISTERED",
                    node_id,
                )

            del self._nodes[node_id]
            logger.info("Removed node '%s' from registry", node_id)
            return True

    # ── Heartbeat ────────────────────────────────────────────────────────

    def update_heartbeat(
        self,
        node_id: str,
        gpu_utilization: float,
        memory_used_mb: int,
        active_requests: int,
    ) -> bool:
        """Process a heartbeat from a worker node.

        Updates the node's ``last_seen`` timestamp and dynamic metrics.

        Parameters
        ----------
        node_id:
            Identifier of the reporting node.
        gpu_utilization:
            Current GPU utilisation percentage (0.0–1.0).
        memory_used_mb:
            Current GPU memory usage in megabytes.
        active_requests:
            Number of in-flight inference requests on the node.

        Returns
        -------
        bool
            ``True`` if the node was found and updated, ``False`` if the
            node is not in the registry.
        """
        with self._lock:
            entry = self._nodes.get(node_id)
            if entry is None:
                logger.warning(
                    "Heartbeat from unknown node '%s'", node_id
                )
                return False

            entry.last_seen = time.monotonic()
            entry.gpu_utilization = gpu_utilization
            entry.memory_used_mb = memory_used_mb
            entry.active_requests = active_requests
            logger.debug(
                "Heartbeat from '%s': gpu=%.1f%%, mem=%dMB, reqs=%d",
                node_id,
                gpu_utilization * 100,
                memory_used_mb,
                active_requests,
            )
            return True

    # ── Lookup ───────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> NodeEntry | None:
        """Return the registry entry for *node_id*, or ``None`` if not found.

        Parameters
        ----------
        node_id:
            Identifier of the node to look up.
        """
        with self._lock:
            return self._nodes.get(node_id)

    # ── Status Transitions ───────────────────────────────────────────────

    def mark_node_healthy(self, node_id: str) -> None:
        """Transition a REGISTERED node to HEALTHY (called after ConfirmReady).

        Parameters
        ----------
        node_id:
            Identifier of the node to mark healthy.

        Raises
        ------
        KeyError
            If *node_id* is not in the registry.
        InvalidTransitionError
            If the node is not in REGISTERED status.
        """
        with self._lock:
            entry = self._nodes.get(node_id)
            if entry is None:
                raise KeyError(f"Node '{node_id}' not found in registry")

            if NodeStatus.HEALTHY not in _VALID_TRANSITIONS.get(
                entry.status, frozenset()
            ):
                raise InvalidTransitionError(
                    f"Cannot transition node '{node_id}' from "
                    f"{entry.status.name} to HEALTHY"
                )

            old_status = entry.status
            entry.status = NodeStatus.HEALTHY
            entry.last_seen = time.monotonic()
            logger.info(
                "Node '%s' transitioned %s → HEALTHY",
                node_id,
                old_status.name,
            )

    # ── Layer Assignment ─────────────────────────────────────────────────

    def update_node_assignment(
        self,
        node_id: str,
        layer_start: int,
        layer_end: int,
    ) -> None:
        """Store a layer assignment on the node entry.

        Called by the scheduler after ``compute_assignments`` to record
        which contiguous layer range a node is responsible for.

        Parameters
        ----------
        node_id:
            Identifier of the node receiving the assignment.
        layer_start:
            First layer index (inclusive) assigned to this node.
        layer_end:
            Last layer index (inclusive) assigned to this node.

        Raises
        ------
        KeyError
            If *node_id* is not in the registry.
        ValueError
            If *layer_start* > *layer_end* or either is negative.
        """
        if layer_start < 0 or layer_end < 0:
            raise ValueError(
                f"Layer indices must be non-negative, got "
                f"layer_start={layer_start}, layer_end={layer_end}"
            )
        if layer_start > layer_end:
            raise ValueError(
                f"layer_start ({layer_start}) must be <= "
                f"layer_end ({layer_end})"
            )

        with self._lock:
            entry = self._nodes.get(node_id)
            if entry is None:
                raise KeyError(f"Node '{node_id}' not found in registry")

            entry.layer_start = layer_start
            entry.layer_end = layer_end
            logger.info(
                "Assigned layers %d–%d to node '%s'",
                layer_start,
                layer_end,
                node_id,
            )

    # ── Filtered Queries ─────────────────────────────────────────────────

    def get_all_healthy_nodes(self) -> list[NodeEntry]:
        """Return all nodes with status HEALTHY.

        Returns
        -------
        list[NodeEntry]
            Snapshot of healthy node entries (order is not guaranteed).
        """
        with self._lock:
            return [
                entry
                for entry in self._nodes.values()
                if entry.status == NodeStatus.HEALTHY
            ]

    def get_all_nodes(self) -> list[NodeEntry]:
        """Return all nodes currently in the registry.

        Returns
        -------
        list[NodeEntry]
            Snapshot of all node entries (order is not guaranteed).
        """
        with self._lock:
            return list(self._nodes.values())

    def _transition_node(self, node_id: str, new_status: NodeStatus) -> bool:
        """Transition a node to *new_status* if the transition is valid.

        Must be called while ``_lock`` is held.

        Returns
        -------
        bool
            ``True`` if the transition was applied, ``False`` if invalid.
        """
        entry = self._nodes.get(node_id)
        if entry is None:
            return False

        allowed = _VALID_TRANSITIONS.get(entry.status, frozenset())
        if new_status not in allowed:
            logger.debug(
                "Invalid transition for '%s': %s → %s",
                node_id,
                entry.status.name,
                new_status.name,
            )
            return False

        old_status = entry.status
        entry.status = new_status
        logger.info(
            "Node '%s' transitioned %s → %s",
            node_id,
            old_status.name,
            new_status.name,
        )
        return True


# ── Health Tracker ───────────────────────────────────────────────────────────


class HealthTracker:
    """Background daemon that monitors node health via heartbeat freshness.

    Runs a periodic check every *heartbeat_interval_s* seconds.  For each
    node it compares ``now - last_seen`` against configurable thresholds:

    * **missed_threshold** (default 3): if a HEALTHY node has not sent a
      heartbeat for ``missed_threshold * heartbeat_interval_s`` seconds it
      is transitioned to UNHEALTHY.
    * **dead_threshold** (default 5): if an UNHEALTHY node has not sent a
      heartbeat for ``dead_threshold * heartbeat_interval_s`` seconds it
      is transitioned to DEAD.

    Nodes that are UNHEALTHY but have resumed heartbeats (i.e. their
    ``last_seen`` is within the missed window) are transitioned back to
    HEALTHY.

    Parameters
    ----------
    registry:
        The :class:`NodeRegistry` to monitor.
    heartbeat_interval_s:
        Seconds between health-check sweeps (default 5).
    missed_threshold:
        Number of missed heartbeat intervals before HEALTHY → UNHEALTHY
        (default 3).
    dead_threshold:
        Number of missed heartbeat intervals before UNHEALTHY → DEAD
        (default 5).
    """

    def __init__(
        self,
        registry: NodeRegistry,
        heartbeat_interval_s: float = 5.0,
        missed_threshold: int = 3,
        dead_threshold: int = 5,
    ) -> None:
        self._registry = registry
        self._heartbeat_interval_s = heartbeat_interval_s
        self._missed_threshold = missed_threshold
        self._dead_threshold = dead_threshold

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background health-check thread.

        The thread is a daemon so it will not prevent interpreter shutdown.
        Calling ``start()`` when already running is a no-op.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.debug("HealthTracker already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="health-tracker",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "HealthTracker started (interval=%.1fs, missed=%d, dead=%d)",
            self._heartbeat_interval_s,
            self._missed_threshold,
            self._dead_threshold,
        )

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to finish.

        Calling ``stop()`` when not running is a no-op.
        """
        if self._thread is None or not self._thread.is_alive():
            logger.debug("HealthTracker not running")
            return

        self._stop_event.set()
        self._thread.join(timeout=self._heartbeat_interval_s * 2)
        if self._thread.is_alive():
            logger.warning("HealthTracker thread did not stop in time")
        else:
            logger.info("HealthTracker stopped")
        self._thread = None

    # ── Internal ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main loop executed by the background thread."""
        while not self._stop_event.is_set():
            self._check_health()
            self._stop_event.wait(timeout=self._heartbeat_interval_s)

    def _check_health(self) -> None:
        """Run a single health-check sweep across all nodes."""
        now = time.monotonic()
        missed_deadline = self._missed_threshold * self._heartbeat_interval_s
        dead_deadline = self._dead_threshold * self._heartbeat_interval_s

        with self._registry._lock:
            for entry in self._registry._nodes.values():
                elapsed = now - entry.last_seen

                if entry.status == NodeStatus.HEALTHY:
                    if elapsed > missed_deadline:
                        self._registry._transition_node(
                            entry.node_id, NodeStatus.UNHEALTHY
                        )

                elif entry.status == NodeStatus.UNHEALTHY:
                    if elapsed > dead_deadline:
                        self._registry._transition_node(
                            entry.node_id, NodeStatus.DEAD
                        )
                    elif elapsed <= missed_deadline:
                        # Heartbeat resumed — recover the node.
                        self._registry._transition_node(
                            entry.node_id, NodeStatus.HEALTHY
                        )

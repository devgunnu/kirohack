"""
Resource Monitor — GPU memory and compute utilization tracking.

Tracks GPU memory, compute utilization, and active request count on the
local worker node.  Feeds data to heartbeat reports and alerts when
actual GPU usage exceeds the user-configured memory limit.

The Resource Monitor is observe-only: it does NOT decide how much memory
to use or autonomously adjust allocations.  Memory limits are
user-configured and enforced externally.

Validates: Requirements 5.1, 5.2, 5.3
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_POLL_INTERVAL_S: float = 1.0
"""Default polling interval in seconds."""

_MIN_POLL_INTERVAL_S: float = 0.1
"""Minimum allowed polling interval to avoid busy-looping."""

_BYTES_PER_MB: int = 1024 * 1024
"""Bytes in one mebibyte."""


# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GpuMetrics:
    """Snapshot of GPU resource metrics at a single point in time.

    All memory values are in megabytes (MB).  ``gpu_utilization`` is a
    fraction in [0.0, 1.0].
    """

    gpu_memory_total_mb: int
    gpu_memory_used_mb: int
    gpu_memory_free_mb: int
    gpu_utilization: float

    def __post_init__(self) -> None:
        if self.gpu_memory_total_mb < 0:
            raise ValueError(
                f"gpu_memory_total_mb must be >= 0, got {self.gpu_memory_total_mb}"
            )
        if self.gpu_memory_used_mb < 0:
            raise ValueError(
                f"gpu_memory_used_mb must be >= 0, got {self.gpu_memory_used_mb}"
            )
        if self.gpu_memory_free_mb < 0:
            raise ValueError(
                f"gpu_memory_free_mb must be >= 0, got {self.gpu_memory_free_mb}"
            )
        if not (0.0 <= self.gpu_utilization <= 1.0):
            raise ValueError(
                f"gpu_utilization must be in [0.0, 1.0], got {self.gpu_utilization}"
            )


@dataclass(frozen=True, slots=True)
class HeartbeatSnapshot:
    """Lightweight resource snapshot for inclusion in heartbeat messages.

    Contains only the fields the Coordinator needs for health monitoring
    and scheduling decisions.
    """

    gpu_utilization: float
    memory_used_mb: int
    active_requests: int


# ── GPU Polling ─────────────────────────────────────────────────────────────


def _poll_gpu_metrics_torch(device_index: int = 0) -> GpuMetrics:
    """Query GPU metrics using PyTorch CUDA APIs.

    Parameters
    ----------
    device_index:
        CUDA device ordinal to query.

    Returns
    -------
    GpuMetrics:
        Current GPU memory and utilization snapshot.

    Raises
    ------
    RuntimeError:
        If CUDA is not available or the device index is invalid.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed — cannot poll GPU metrics")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid CUDA device index {device_index}; "
            f"available devices: {torch.cuda.device_count()}"
        )

    mem_total = torch.cuda.get_device_properties(device_index).total_mem
    mem_allocated = torch.cuda.memory_allocated(device_index)

    # Use allocated as "used" — this reflects actual tensor allocations.
    total_mb = mem_total // _BYTES_PER_MB
    used_mb = mem_allocated // _BYTES_PER_MB
    free_mb = (mem_total - mem_allocated) // _BYTES_PER_MB

    # PyTorch doesn't expose SM utilization directly.  torch.cuda.utilization()
    # is available in recent builds and returns an int percentage (0–100).
    try:
        util_pct: int = torch.cuda.utilization(device_index)  # type: ignore[attr-defined]
        utilization = max(0.0, min(1.0, util_pct / 100.0))
    except (AttributeError, RuntimeError):
        # Fallback: utilization unavailable — report 0.0.
        utilization = 0.0

    return GpuMetrics(
        gpu_memory_total_mb=total_mb,
        gpu_memory_used_mb=used_mb,
        gpu_memory_free_mb=free_mb,
        gpu_utilization=utilization,
    )


# ── Resource Monitor ────────────────────────────────────────────────────────


class ResourceMonitor:
    """Tracks GPU memory, utilization, and active request count.

    Polls GPU metrics at a configurable interval on a background daemon
    thread.  Provides a heartbeat snapshot and alerts when GPU memory
    usage exceeds the user-configured limit.

    Parameters
    ----------
    gpu_memory_limit_mb:
        User-configured maximum GPU memory this node may use (MB).
    poll_interval_s:
        How often to poll GPU metrics, in seconds.  Defaults to 1.0.
    device_index:
        CUDA device ordinal to monitor.  Defaults to 0.
    poll_fn:
        Optional override for the GPU polling function.  Useful for
        testing without a real GPU.  Must return a ``GpuMetrics``.
    on_limit_exceeded:
        Optional callback invoked when ``gpu_memory_used_mb`` exceeds
        ``gpu_memory_limit_mb``.  Receives the current ``GpuMetrics``.
    """

    def __init__(
        self,
        gpu_memory_limit_mb: int,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        device_index: int = 0,
        poll_fn: Optional[Callable[[int], GpuMetrics]] = None,
        on_limit_exceeded: Optional[Callable[[GpuMetrics], None]] = None,
    ) -> None:
        if gpu_memory_limit_mb < 0:
            raise ValueError(
                f"gpu_memory_limit_mb must be >= 0, got {gpu_memory_limit_mb}"
            )
        if poll_interval_s < _MIN_POLL_INTERVAL_S:
            raise ValueError(
                f"poll_interval_s must be >= {_MIN_POLL_INTERVAL_S}, "
                f"got {poll_interval_s}"
            )

        self._gpu_memory_limit_mb = gpu_memory_limit_mb
        self._poll_interval_s = poll_interval_s
        self._device_index = device_index
        self._poll_fn = poll_fn or _poll_gpu_metrics_torch
        self._on_limit_exceeded = on_limit_exceeded

        # Mutable state protected by a lock.
        self._lock = threading.Lock()
        self._latest_metrics: Optional[GpuMetrics] = None
        self._active_requests: int = 0
        self._shard_memory_mb: int = 0
        self._activation_memory_mb: int = 0

        # Background polling thread control.
        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def gpu_memory_limit_mb(self) -> int:
        """User-configured GPU memory limit in MB."""
        return self._gpu_memory_limit_mb

    @gpu_memory_limit_mb.setter
    def gpu_memory_limit_mb(self, value: int) -> None:
        """Update the user-configured GPU memory limit at runtime.

        Parameters
        ----------
        value:
            New memory limit in megabytes.  Must be >= 0.

        Raises
        ------
        ValueError:
            If *value* is negative.
        """
        if value < 0:
            raise ValueError(f"gpu_memory_limit_mb must be >= 0, got {value}")
        prev = self._gpu_memory_limit_mb
        self._gpu_memory_limit_mb = value
        if prev != value:
            logger.info("GPU memory limit updated: %d MB → %d MB", prev, value)

    @property
    def poll_interval_s(self) -> float:
        """Current polling interval in seconds."""
        return self._poll_interval_s

    @property
    def is_polling(self) -> bool:
        """Whether the background polling thread is running."""
        return self._poll_thread is not None and self._poll_thread.is_alive()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background GPU metrics polling thread.

        Does nothing if polling is already active.
        """
        if self.is_polling:
            logger.debug("ResourceMonitor polling already active — skipping start")
            return

        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="resource-monitor-poll",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info(
            "ResourceMonitor started (interval=%.2fs, device=%d, limit=%dMB)",
            self._poll_interval_s,
            self._device_index,
            self._gpu_memory_limit_mb,
        )

    def stop(self) -> None:
        """Stop the background polling thread.

        Blocks until the thread exits (up to one poll interval + 1s).
        """
        if not self.is_polling:
            return

        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=self._poll_interval_s + 1.0)
            self._poll_thread = None
        logger.info("ResourceMonitor stopped")

    # ── Polling Loop ─────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background loop that periodically queries GPU metrics."""
        while not self._stop_event.is_set():
            try:
                metrics = self._poll_fn(self._device_index)
                with self._lock:
                    self._latest_metrics = metrics

                self._check_memory_limit(metrics)

            except Exception:
                logger.exception("GPU metrics poll failed")

            self._stop_event.wait(timeout=self._poll_interval_s)

    def _check_memory_limit(self, metrics: GpuMetrics) -> None:
        """Compare polled usage against the user-configured limit.

        Logs a warning and fires the ``on_limit_exceeded`` callback when
        ``gpu_memory_used_mb`` exceeds ``gpu_memory_limit_mb``.
        """
        if metrics.gpu_memory_used_mb > self._gpu_memory_limit_mb:
            logger.warning(
                "GPU memory usage (%d MB) exceeds limit (%d MB)",
                metrics.gpu_memory_used_mb,
                self._gpu_memory_limit_mb,
            )
            if self._on_limit_exceeded is not None:
                try:
                    self._on_limit_exceeded(metrics)
                except Exception:
                    logger.exception("on_limit_exceeded callback failed")

    # ── Metrics Access ───────────────────────────────────────────────────

    def get_latest_metrics(self) -> Optional[GpuMetrics]:
        """Return the most recently polled GPU metrics, or ``None``."""
        with self._lock:
            return self._latest_metrics

    def poll_once(self) -> GpuMetrics:
        """Perform a single synchronous GPU poll and update internal state.

        Useful during initialization or when an immediate reading is
        needed without waiting for the background thread.  Also runs the
        memory-limit comparison so callers get the same alerting
        behaviour as the background polling loop.

        Returns
        -------
        GpuMetrics:
            The freshly polled metrics.
        """
        metrics = self._poll_fn(self._device_index)
        with self._lock:
            self._latest_metrics = metrics

        self._check_memory_limit(metrics)
        return metrics

    # ── Active Request Tracking ──────────────────────────────────────────

    def increment_active_requests(self) -> int:
        """Increment the in-flight request counter.

        Called when a new forward pass begins processing on this node.

        Returns
        -------
        int:
            The new active request count after incrementing.
        """
        with self._lock:
            self._active_requests += 1
            count = self._active_requests
        logger.debug("Active requests incremented to %d", count)
        return count

    def decrement_active_requests(self) -> int:
        """Decrement the in-flight request counter.

        Called when a forward pass completes (or fails) on this node.
        The counter is clamped to zero — it will never go negative.

        Returns
        -------
        int:
            The new active request count after decrementing.
        """
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            count = self._active_requests
        logger.debug("Active requests decremented to %d", count)
        return count

    @property
    def active_requests(self) -> int:
        """Current number of in-flight forward passes."""
        with self._lock:
            return self._active_requests

    # ── Shard / Activation Memory Tracking ───────────────────────────────

    def set_shard_memory_mb(self, mb: int) -> None:
        """Record the memory consumed by the loaded model shard.

        Called by the Shard Manager after loading or unloading a shard.
        Negative values are clamped to zero.

        Parameters
        ----------
        mb:
            Memory consumed by the model shard in megabytes.
        """
        clamped = max(0, mb)
        with self._lock:
            prev = self._shard_memory_mb
            self._shard_memory_mb = clamped
        if prev != clamped:
            logger.info("Shard memory updated: %d MB → %d MB", prev, clamped)

    def set_activation_memory_mb(self, mb: int) -> None:
        """Record the estimated activation memory for in-flight requests.

        Called by the Layer Engine to report current activation memory
        usage based on active forward passes.  Negative values are
        clamped to zero.

        Parameters
        ----------
        mb:
            Estimated activation memory in megabytes.
        """
        clamped = max(0, mb)
        with self._lock:
            prev = self._activation_memory_mb
            self._activation_memory_mb = clamped
        if prev != clamped:
            logger.debug("Activation memory updated: %d MB → %d MB", prev, clamped)

    @property
    def shard_memory_mb(self) -> int:
        """Memory consumed by the loaded model shard in MB."""
        with self._lock:
            return self._shard_memory_mb

    @property
    def activation_memory_mb(self) -> int:
        """Estimated activation memory for in-flight requests in MB."""
        with self._lock:
            return self._activation_memory_mb

    # ── Heartbeat Snapshot ───────────────────────────────────────────────

    def get_heartbeat_snapshot(self) -> HeartbeatSnapshot:
        """Build a lightweight snapshot for heartbeat messages.

        Returns
        -------
        HeartbeatSnapshot:
            Contains ``gpu_utilization``, ``memory_used_mb``, and
            ``active_requests`` reflecting the latest polled state.
        """
        with self._lock:
            metrics = self._latest_metrics
            active = self._active_requests

        if metrics is None:
            return HeartbeatSnapshot(
                gpu_utilization=0.0,
                memory_used_mb=0,
                active_requests=active,
            )

        return HeartbeatSnapshot(
            gpu_utilization=metrics.gpu_utilization,
            memory_used_mb=metrics.gpu_memory_used_mb,
            active_requests=active,
        )

    # ── Limit Check ──────────────────────────────────────────────────────

    def is_over_limit(self) -> bool:
        """Check whether current GPU usage exceeds the configured limit.

        Returns ``False`` if no metrics have been polled yet.
        """
        with self._lock:
            if self._latest_metrics is None:
                return False
            return self._latest_metrics.gpu_memory_used_mb > self._gpu_memory_limit_mb

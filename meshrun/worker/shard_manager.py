"""
Shard Manager — selective download and lifecycle management of model shards.

The Shard Manager uses the safetensors format with HTTP Range requests to
selectively download only the assigned layer weights from a model hosted on
any HTTP-compatible server (HuggingFace Hub, S3, etc.). Each worker node
fetches only the byte ranges corresponding to its assigned layers, avoiding
full model downloads.

Weight download strategy:
1. Fetch safetensors header via HTTP Range request (first 8 bytes → header
   size, then header JSON)
2. Parse tensor metadata to identify tensors belonging to assigned layers
3. Download only those tensor byte ranges using HTTP Range requests
4. Cache downloaded weights locally to avoid re-downloading on restart
5. Load cached weights into GPU memory

Validates: Requirements 3.1, 3.2, 3.3, 3.4
"""

from __future__ import annotations

import json
import logging
import struct
import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Enumerations ─────────────────────────────────────────────────────────────

class ShardDType(IntEnum):
    """Quantization formats supported for model shards."""
    FP16 = 1  # 2 bytes per element (IEEE 754 half-precision)
    INT8 = 2  # 1 byte per element  (signed 8-bit integer)


class LoadStatus(IntEnum):
    """Lifecycle status of a model shard on a worker node."""
    UNLOADED = 0
    DOWNLOADING = 1
    LOADING = 2
    READY = 3
    ERROR = 4


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class LayerRange:
    """Contiguous range of transformer layers assigned to a node.

    Both ``start`` and ``end`` are inclusive indices.
    """

    start: int
    """First assigned layer index (inclusive)."""

    end: int
    """Last assigned layer index (inclusive)."""

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"layer start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"layer end ({self.end}) must be >= start ({self.start})"
            )

    @property
    def count(self) -> int:
        """Number of layers in this range."""
        return self.end - self.start + 1


@dataclass(slots=True)
class ShardMetadata:
    """Metadata describing a model shard hosted on a worker node.

    Tracks the model identity, assigned layer range, quantization format,
    memory footprint, download/load status, and local cache directory.
    This is the central state object for the Shard Manager lifecycle.
    """

    model_id: str
    """Identifier of the model being served (e.g. ``'llama-3b'``)."""

    model_url: str
    """HTTP URL to the safetensors model file."""

    layer_range: LayerRange
    """Contiguous range of transformer layers assigned to this node."""

    dtype: ShardDType
    """Quantization format for this shard (fp16 or int8)."""

    cache_dir: Path
    """Local directory for cached weight files."""

    memory_footprint_mb: float = 0.0
    """GPU memory consumed by the loaded shard, in megabytes."""

    load_status: LoadStatus = LoadStatus.UNLOADED
    """Current lifecycle status of the shard."""

    bytes_downloaded: int = 0
    """Number of bytes downloaded so far (for progress tracking)."""

    bytes_total: int = 0
    """Total bytes to download for assigned layers."""

    error_message: Optional[str] = None
    """Error description if ``load_status`` is ``ERROR``."""

    loaded_tensors: dict[str, object] = field(default_factory=dict)
    """Mapping of tensor name → torch.Tensor on GPU (populated after load)."""

    @property
    def download_progress(self) -> float:
        """Fraction of bytes downloaded, in range [0.0, 1.0]."""
        if self.bytes_total <= 0:
            return 0.0
        return min(self.bytes_downloaded / self.bytes_total, 1.0)


@dataclass(frozen=True, slots=True)
class ShardInfo:
    """Immutable snapshot of shard metadata for external consumers.

    Returned by :func:`get_shard_info` to provide a read-only view of the
    current shard state, including download progress.
    """

    model_id: str
    """Identifier of the model being served."""

    model_url: str
    """HTTP URL to the safetensors model file."""

    layer_start: int
    """First assigned layer (inclusive)."""

    layer_end: int
    """Last assigned layer (inclusive)."""

    dtype: ShardDType
    """Quantization format for this shard."""

    memory_footprint_mb: float
    """GPU memory consumed by the loaded shard, in megabytes."""

    load_status: LoadStatus
    """Current lifecycle status of the shard."""

    bytes_downloaded: int
    """Number of bytes downloaded so far."""

    bytes_total: int
    """Total bytes to download for assigned layers."""

    download_progress: float
    """Fraction of bytes downloaded, in range [0.0, 1.0]."""


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """Metadata for a single tensor parsed from a safetensors header.

    Captures the tensor name, dtype string, shape, and the byte offset
    range within the safetensors file's data section.
    """

    name: str
    """Tensor name (e.g. ``'model.layers.5.self_attn.q_proj.weight'``)."""

    dtype: str
    """Data type string from safetensors (e.g. ``'F16'``, ``'I8'``, ``'F32'``)."""

    shape: tuple[int, ...]
    """Tensor shape as a tuple of dimension sizes."""

    data_offset_start: int
    """Start byte offset within the data section (relative to data start)."""

    data_offset_end: int
    """End byte offset within the data section (exclusive)."""

    @property
    def byte_size(self) -> int:
        """Number of bytes this tensor occupies in the data section."""
        return self.data_offset_end - self.data_offset_start


# ── Safetensors Header Fetcher ──────────────────────────────────────────────

# Safetensors file layout:
#   [0:8]   — uint64 LE: header size in bytes (N)
#   [8:8+N] — JSON header: dict of tensor_name → {dtype, shape, data_offsets}
#   [8+N:]  — raw tensor data (contiguous)

HEADER_SIZE_BYTES = 8
"""Number of bytes at the start of a safetensors file encoding the header size."""

_MAX_HEADER_SIZE = 100 * 1024 * 1024  # 100 MB safety cap
"""Maximum allowed header size to prevent unbounded memory allocation."""

_DEFAULT_TIMEOUT = 30  # seconds
"""Default HTTP request timeout."""


class SafetensorsHeaderError(Exception):
    """Raised when the safetensors header cannot be fetched or parsed."""


def _http_range_request(url: str, start: int, end: int, *, timeout: int = _DEFAULT_TIMEOUT) -> bytes:
    """Fetch a byte range from *url* using an HTTP Range request.

    Parameters
    ----------
    url:
        HTTP(S) URL to fetch from.
    start:
        First byte offset (inclusive).
    end:
        Last byte offset (inclusive).
    timeout:
        Request timeout in seconds.

    Returns
    -------
    bytes
        The requested byte range.

    Raises
    ------
    SafetensorsHeaderError
        If the HTTP request fails or the server does not support Range
        requests.
    """
    req = Request(url, headers={"Range": f"bytes={start}-{end}"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as exc:
        raise SafetensorsHeaderError(
            f"HTTP {exc.code} fetching bytes {start}-{end} from {url}: {exc.reason}"
        ) from exc
    except URLError as exc:
        raise SafetensorsHeaderError(
            f"URL error fetching {url}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise SafetensorsHeaderError(
            f"Timeout fetching bytes {start}-{end} from {url}"
        ) from exc


def fetch_safetensors_header(url: str, *, timeout: int = _DEFAULT_TIMEOUT) -> tuple[int, dict[str, TensorInfo]]:
    """Fetch and parse the safetensors header from a remote HTTP URL.

    Performs two HTTP Range requests:
    1. Bytes 0–7 to read the 8-byte little-endian header size.
    2. Bytes 8–(8 + header_size - 1) to read the full JSON header.

    Parameters
    ----------
    url:
        HTTP(S) URL pointing to a ``.safetensors`` file.
    timeout:
        Per-request timeout in seconds.

    Returns
    -------
    tuple[int, dict[str, TensorInfo]]
        A 2-tuple of ``(header_size, tensors)`` where *header_size* is the
        byte length of the JSON header and *tensors* is a dict mapping
        tensor names to their :class:`TensorInfo` metadata.

    Raises
    ------
    SafetensorsHeaderError
        If the header cannot be fetched, is malformed, or exceeds the
        safety size cap.
    """
    # ── Step 1: read the 8-byte header size ──────────────────────────────
    size_bytes = _http_range_request(url, 0, HEADER_SIZE_BYTES - 1, timeout=timeout)
    if len(size_bytes) != HEADER_SIZE_BYTES:
        raise SafetensorsHeaderError(
            f"Expected {HEADER_SIZE_BYTES} bytes for header size, got {len(size_bytes)}"
        )

    header_size = struct.unpack("<Q", size_bytes)[0]

    if header_size == 0:
        raise SafetensorsHeaderError("Safetensors header size is 0")
    if header_size > _MAX_HEADER_SIZE:
        raise SafetensorsHeaderError(
            f"Header size {header_size} exceeds safety cap of {_MAX_HEADER_SIZE} bytes"
        )

    # ── Step 2: read the JSON header ─────────────────────────────────────
    header_bytes = _http_range_request(
        url,
        HEADER_SIZE_BYTES,
        HEADER_SIZE_BYTES + header_size - 1,
        timeout=timeout,
    )
    if len(header_bytes) != header_size:
        raise SafetensorsHeaderError(
            f"Expected {header_size} bytes for JSON header, got {len(header_bytes)}"
        )

    # ── Step 3: parse JSON and extract tensor metadata ───────────────────
    try:
        header_json: dict = json.loads(header_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SafetensorsHeaderError(f"Failed to parse header JSON: {exc}") from exc

    return header_size, _parse_tensor_metadata(header_json)


def _parse_tensor_metadata(header_json: dict) -> dict[str, TensorInfo]:
    """Parse the safetensors JSON header into a dict of :class:`TensorInfo`.

    The safetensors header is a JSON object where each key is a tensor name
    and each value is an object with ``dtype``, ``shape``, and
    ``data_offsets`` fields.  The special key ``__metadata__`` (if present)
    is skipped.

    Parameters
    ----------
    header_json:
        Parsed JSON header dict from a safetensors file.

    Returns
    -------
    dict[str, TensorInfo]
        Mapping of tensor name → metadata.

    Raises
    ------
    SafetensorsHeaderError
        If any tensor entry is missing required fields or has invalid
        offset values.
    """
    tensors: dict[str, TensorInfo] = {}

    for name, meta in header_json.items():
        # Skip the optional metadata key
        if name == "__metadata__":
            continue

        if not isinstance(meta, dict):
            raise SafetensorsHeaderError(
                f"Tensor '{name}': expected dict metadata, got {type(meta).__name__}"
            )

        # ── Validate required fields ────────────────────────────────────
        missing = [k for k in ("dtype", "shape", "data_offsets") if k not in meta]
        if missing:
            raise SafetensorsHeaderError(
                f"Tensor '{name}': missing required fields: {missing}"
            )

        dtype_str = meta["dtype"]
        shape = meta["shape"]
        offsets = meta["data_offsets"]

        if not isinstance(dtype_str, str):
            raise SafetensorsHeaderError(
                f"Tensor '{name}': dtype must be a string, got {type(dtype_str).__name__}"
            )
        if not isinstance(shape, list) or not all(isinstance(d, int) for d in shape):
            raise SafetensorsHeaderError(
                f"Tensor '{name}': shape must be a list of ints, got {shape!r}"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) for o in offsets)
        ):
            raise SafetensorsHeaderError(
                f"Tensor '{name}': data_offsets must be [start, end], got {offsets!r}"
            )

        offset_start, offset_end = offsets
        if offset_start < 0 or offset_end < offset_start:
            raise SafetensorsHeaderError(
                f"Tensor '{name}': invalid data_offsets [{offset_start}, {offset_end}]"
            )

        tensors[name] = TensorInfo(
            name=name,
            dtype=dtype_str,
            shape=tuple(shape),
            data_offset_start=offset_start,
            data_offset_end=offset_end,
        )

    return tensors


# ── Layer-to-Tensor Mapping ─────────────────────────────────────────────────

# Pattern matching transformer layer tensors: "model.layers.{i}.anything"
_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")

# Tensor name fragments that identify embedding and LM head tensors.
_EMBEDDING_FRAGMENTS = ("embed_tokens",)
_LM_HEAD_FRAGMENTS = ("lm_head",)


def filter_tensors_for_assignment(
    tensors: dict[str, TensorInfo],
    layer_range: LayerRange,
    *,
    is_first_node: bool = False,
    is_final_node: bool = False,
) -> dict[str, TensorInfo]:
    """Filter a safetensors tensor map to only those needed by this node.

    Selects tensors whose names match ``model.layers.{i}.*`` for every
    layer index *i* in *layer_range* (inclusive on both ends).  Additionally:

    * If *is_first_node* is ``True``, embedding tensors (names containing
      ``embed_tokens``) are included.
    * If *is_final_node* is ``True``, LM head tensors (names containing
      ``lm_head``) are included.

    Parameters
    ----------
    tensors:
        Full tensor metadata dict as returned by
        :func:`fetch_safetensors_header`.
    layer_range:
        The contiguous layer range assigned to this node.
    is_first_node:
        Whether this node hosts the first layers in the model and
        therefore needs the embedding weights.
    is_final_node:
        Whether this node hosts the last layers in the model and
        therefore needs the LM head weights.

    Returns
    -------
    dict[str, TensorInfo]
        Subset of *tensors* relevant to this node's assignment.
    """
    assigned_indices = set(range(layer_range.start, layer_range.end + 1))
    result: dict[str, TensorInfo] = {}

    for name, info in tensors.items():
        # Check if this tensor belongs to an assigned transformer layer
        m = _LAYER_PATTERN.match(name)
        if m is not None:
            layer_idx = int(m.group(1))
            if layer_idx in assigned_indices:
                result[name] = info
            continue

        # Embedding tensors — needed by the first node only
        if is_first_node and any(frag in name for frag in _EMBEDDING_FRAGMENTS):
            result[name] = info
            continue

        # LM head tensors — needed by the final node only
        if is_final_node and any(frag in name for frag in _LM_HEAD_FRAGMENTS):
            result[name] = info
            continue

    return result


# ── Selective Weight Download ───────────────────────────────────────────────

def _compute_absolute_offset(header_size: int, data_offset: int) -> int:
    """Compute the absolute byte offset within a safetensors file.

    The safetensors file layout is::

        [0:8]           — uint64 LE header size (N)
        [8:8+N]         — JSON header
        [8+N:]          — raw tensor data

    So a tensor whose ``data_offset`` is relative to the data section
    starts at absolute position ``8 + header_size + data_offset``.

    Parameters
    ----------
    header_size:
        Byte length of the JSON header (the value read from bytes 0–7).
    data_offset:
        Byte offset within the data section.

    Returns
    -------
    int
        Absolute byte offset from the start of the file.
    """
    return HEADER_SIZE_BYTES + header_size + data_offset


def download_tensor_bytes(
    url: str,
    header_size: int,
    tensor: TensorInfo,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
) -> bytes:
    """Download the raw bytes for a single tensor via an HTTP Range request.

    Computes the absolute byte range as
    ``[8 + header_size + data_offset_start, 8 + header_size + data_offset_end)``
    and fetches exactly those bytes.

    Parameters
    ----------
    url:
        HTTP(S) URL to the ``.safetensors`` file.
    header_size:
        Byte length of the JSON header (returned by
        :func:`fetch_safetensors_header`).
    tensor:
        Tensor metadata describing the byte range to fetch.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    bytes
        Raw tensor data (length == ``tensor.byte_size``).

    Raises
    ------
    SafetensorsHeaderError
        If the HTTP request fails or the returned byte count does not
        match the expected tensor size.
    """
    if tensor.byte_size <= 0:
        return b""

    abs_start = _compute_absolute_offset(header_size, tensor.data_offset_start)
    # HTTP Range end is inclusive, and data_offset_end is exclusive,
    # so subtract 1 for the Range header.
    abs_end = _compute_absolute_offset(header_size, tensor.data_offset_end) - 1

    data = _http_range_request(url, abs_start, abs_end, timeout=timeout)

    if len(data) != tensor.byte_size:
        raise SafetensorsHeaderError(
            f"Tensor '{tensor.name}': expected {tensor.byte_size} bytes, "
            f"got {len(data)}"
        )
    return data


def download_selected_tensors(
    url: str,
    header_size: int,
    tensors: dict[str, TensorInfo],
    *,
    timeout: int = _DEFAULT_TIMEOUT,
    progress_callback: Optional[object] = None,
) -> dict[str, bytes]:
    """Download raw bytes for a set of tensors via HTTP Range requests.

    Each tensor is fetched individually using its byte offset range from
    the safetensors header.  An optional *progress_callback* is invoked
    after each tensor download with ``(tensor_name, bytes_downloaded_so_far,
    bytes_total)`` to support progress tracking.

    Parameters
    ----------
    url:
        HTTP(S) URL to the ``.safetensors`` file.
    header_size:
        Byte length of the JSON header (returned by
        :func:`fetch_safetensors_header`).
    tensors:
        Dict of tensor name → :class:`TensorInfo` for the tensors to
        download (typically the output of
        :func:`filter_tensors_for_assignment`).
    timeout:
        Per-request HTTP timeout in seconds.
    progress_callback:
        Optional callable ``(name: str, downloaded: int, total: int) -> None``
        invoked after each tensor is fetched.

    Returns
    -------
    dict[str, bytes]
        Mapping of tensor name → raw bytes for every tensor in *tensors*.

    Raises
    ------
    SafetensorsHeaderError
        If any individual tensor download fails or returns an unexpected
        byte count.
    """
    total_bytes = sum(t.byte_size for t in tensors.values())
    downloaded_bytes = 0
    result: dict[str, bytes] = {}

    for name, info in tensors.items():
        data = download_tensor_bytes(url, header_size, info, timeout=timeout)
        result[name] = data
        downloaded_bytes += len(data)

        if progress_callback is not None:
            progress_callback(name, downloaded_bytes, total_bytes)  # type: ignore[operator]

    return result


# ── Local Weight Cache ──────────────────────────────────────────────────────

def _tensor_name_to_filename(tensor_name: str) -> str:
    """Convert a tensor name to a safe filesystem filename.

    Replaces dots with ``--`` to avoid path separator issues, producing
    filenames like ``model--layers--5--self_attn--q_proj--weight.bin``.

    Parameters
    ----------
    tensor_name:
        Safetensors tensor name (e.g.
        ``'model.layers.5.self_attn.q_proj.weight'``).

    Returns
    -------
    str
        Filesystem-safe filename with ``.bin`` extension.
    """
    safe = tensor_name.replace(".", "--")
    return f"{safe}.bin"


def _get_cache_dir_for_model(cache_dir: Path, model_id: str) -> Path:
    """Return the cache subdirectory for a specific model.

    Parameters
    ----------
    cache_dir:
        Root cache directory.
    model_id:
        Model identifier (e.g. ``'llama-3b'``).

    Returns
    -------
    Path
        ``cache_dir / model_id`` path (not yet created on disk).
    """
    return cache_dir / model_id


def save_tensor_to_cache(
    cache_dir: Path,
    model_id: str,
    tensor_name: str,
    data: bytes,
) -> Path:
    """Save raw tensor bytes to the local weight cache.

    Creates the directory ``cache_dir/{model_id}/`` if it does not exist
    and writes *data* to a ``.bin`` file named after the tensor.

    Parameters
    ----------
    cache_dir:
        Root cache directory.
    model_id:
        Model identifier.
    tensor_name:
        Safetensors tensor name.
    data:
        Raw tensor bytes to persist.

    Returns
    -------
    Path
        Absolute path to the written cache file.
    """
    model_cache = _get_cache_dir_for_model(cache_dir, model_id)
    model_cache.mkdir(parents=True, exist_ok=True)
    filename = _tensor_name_to_filename(tensor_name)
    filepath = model_cache / filename
    filepath.write_bytes(data)
    return filepath


def is_tensor_cached(
    cache_dir: Path,
    model_id: str,
    tensor_name: str,
    expected_size: int,
) -> bool:
    """Check whether a tensor is present in the local cache with the correct size.

    Validates the cached file by comparing its size on disk against the
    expected byte count from the safetensors header.

    Parameters
    ----------
    cache_dir:
        Root cache directory.
    model_id:
        Model identifier.
    tensor_name:
        Safetensors tensor name.
    expected_size:
        Expected file size in bytes (from :attr:`TensorInfo.byte_size`).

    Returns
    -------
    bool
        ``True`` if the cache file exists and its size matches
        *expected_size*; ``False`` otherwise.
    """
    model_cache = _get_cache_dir_for_model(cache_dir, model_id)
    filepath = model_cache / _tensor_name_to_filename(tensor_name)
    if not filepath.is_file():
        return False
    return filepath.stat().st_size == expected_size


def load_tensor_from_cache(
    cache_dir: Path,
    model_id: str,
    tensor_name: str,
    expected_size: int,
) -> bytes | None:
    """Load raw tensor bytes from the local cache if valid.

    Returns the cached bytes only when the file exists and its size
    matches *expected_size*.  Returns ``None`` otherwise, signalling
    that the tensor must be re-downloaded.

    Parameters
    ----------
    cache_dir:
        Root cache directory.
    model_id:
        Model identifier.
    tensor_name:
        Safetensors tensor name.
    expected_size:
        Expected file size in bytes.

    Returns
    -------
    bytes | None
        Cached tensor data, or ``None`` if the cache entry is missing
        or invalid.
    """
    model_cache = _get_cache_dir_for_model(cache_dir, model_id)
    filepath = model_cache / _tensor_name_to_filename(tensor_name)
    if not filepath.is_file():
        return None
    if filepath.stat().st_size != expected_size:
        logger.warning(
            "Cache size mismatch for '%s': expected %d bytes, got %d — re-downloading",
            tensor_name,
            expected_size,
            filepath.stat().st_size,
        )
        return None
    return filepath.read_bytes()


def download_selected_tensors_cached(
    url: str,
    header_size: int,
    tensors: dict[str, TensorInfo],
    cache_dir: Path,
    model_id: str,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
    progress_callback: Optional[object] = None,
) -> dict[str, bytes]:
    """Download tensors with local caching — cache-first, download on miss.

    For each tensor in *tensors*:

    1. Check the local cache (``cache_dir/{model_id}/{safe_name}.bin``).
       If the file exists and its size matches the expected byte count,
       load from cache.
    2. Otherwise, download via HTTP Range request and save to cache.

    An optional *progress_callback* is invoked after each tensor is
    resolved (from cache or network) with
    ``(tensor_name, bytes_resolved_so_far, bytes_total)``.

    Parameters
    ----------
    url:
        HTTP(S) URL to the ``.safetensors`` file.
    header_size:
        Byte length of the JSON header.
    tensors:
        Dict of tensor name → :class:`TensorInfo` to resolve.
    cache_dir:
        Root local cache directory.
    model_id:
        Model identifier (used as cache subdirectory name).
    timeout:
        Per-request HTTP timeout in seconds.
    progress_callback:
        Optional callable ``(name, downloaded, total) -> None``.

    Returns
    -------
    dict[str, bytes]
        Mapping of tensor name → raw bytes for every tensor in *tensors*.

    Raises
    ------
    SafetensorsHeaderError
        If any tensor download fails.
    """
    total_bytes = sum(t.byte_size for t in tensors.values())
    resolved_bytes = 0
    result: dict[str, bytes] = {}

    for name, info in tensors.items():
        # ── Try cache first ──────────────────────────────────────────
        cached = load_tensor_from_cache(cache_dir, model_id, name, info.byte_size)
        if cached is not None:
            logger.debug("Cache hit for tensor '%s' (%d bytes)", name, info.byte_size)
            result[name] = cached
        else:
            # ── Download and persist ─────────────────────────────────
            logger.debug("Cache miss for tensor '%s' — downloading", name)
            data = download_tensor_bytes(url, header_size, info, timeout=timeout)
            save_tensor_to_cache(cache_dir, model_id, name, data)
            result[name] = data

        resolved_bytes += info.byte_size
        if progress_callback is not None:
            progress_callback(name, resolved_bytes, total_bytes)  # type: ignore[operator]

    return result


# ── Safetensors dtype → torch dtype mapping ─────────────────────────────────

# Safetensors uses string dtype identifiers.  We map the ones relevant to
# this project (fp16 / int8) plus a few common extras so that the
# reconstruction logic is not artificially narrow.
_SAFETENSORS_DTYPE_TO_TORCH: dict[str, str] = {
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
}

_DTYPE_ELEMENT_SIZE: dict[str, int] = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
}


def _reconstruct_tensor(
    raw_bytes: bytes,
    tensor_info: TensorInfo,
    *,
    device: str = "cpu",
) -> object:
    """Reconstruct a torch tensor from raw safetensors bytes.

    The raw bytes are interpreted according to the dtype and shape stored
    in *tensor_info*.  The resulting tensor is placed on *device*.

    Parameters
    ----------
    raw_bytes:
        Raw contiguous tensor data (row-major / C-contiguous).
    tensor_info:
        Metadata describing the tensor's dtype and shape.
    device:
        Target torch device (e.g. ``'cpu'``, ``'cuda'``, ``'cuda:0'``).

    Returns
    -------
    torch.Tensor
        Reconstructed tensor on the requested device.

    Raises
    ------
    RuntimeError
        If PyTorch is not installed.
    ValueError
        If the safetensors dtype is not supported or the byte count does
        not match the expected size for the given shape and dtype.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for tensor reconstruction but is not installed"
        )

    torch_dtype_name = _SAFETENSORS_DTYPE_TO_TORCH.get(tensor_info.dtype)
    if torch_dtype_name is None:
        raise ValueError(
            f"Unsupported safetensors dtype '{tensor_info.dtype}' for tensor "
            f"'{tensor_info.name}'"
        )

    torch_dtype = getattr(torch, torch_dtype_name)
    element_size = _DTYPE_ELEMENT_SIZE[tensor_info.dtype]

    # Validate byte count
    num_elements = 1
    for d in tensor_info.shape:
        num_elements *= d
    expected_bytes = num_elements * element_size

    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Tensor '{tensor_info.name}': expected {expected_bytes} bytes "
            f"for shape {tensor_info.shape} with dtype {tensor_info.dtype}, "
            f"got {len(raw_bytes)}"
        )

    # Reconstruct from raw buffer — frombuffer gives a 1-D view, then reshape
    tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch_dtype)
    tensor = tensor.reshape(tensor_info.shape)

    if device != "cpu":
        tensor = tensor.to(device)

    return tensor


# ── LoadShard Orchestrator ──────────────────────────────────────────────────

def load_shard(
    metadata: ShardMetadata,
    *,
    is_first_node: bool = False,
    is_final_node: bool = False,
    device: str = "cuda",
    timeout: int = _DEFAULT_TIMEOUT,
) -> ShardMetadata:
    """Orchestrate the full shard loading flow.

    Executes the complete lifecycle:

    1. **Check status** — only ``UNLOADED`` or ``ERROR`` shards can be loaded.
    2. **Transition → DOWNLOADING** — fetch the safetensors header, filter
       tensors for the assigned layer range, and download (or load from
       cache) the required byte ranges.
    3. **Transition → LOADING** — reconstruct each tensor from raw bytes
       and move it to the target GPU device.
    4. **Transition → READY** — all tensors are on-device and the shard is
       ready for inference.

    If any step fails the status transitions to ``ERROR`` with a
    descriptive message.

    Parameters
    ----------
    metadata:
        Mutable :class:`ShardMetadata` describing the shard to load.
        Updated in-place throughout the flow.
    is_first_node:
        Include embedding tensors (needed by the first pipeline node).
    is_final_node:
        Include LM head tensors (needed by the last pipeline node).
    device:
        Target torch device (e.g. ``'cuda'``, ``'cuda:0'``, ``'cpu'``).
    timeout:
        Per-request HTTP timeout in seconds.

    Returns
    -------
    ShardMetadata
        The same *metadata* object, updated with final status, loaded
        tensors, and memory footprint.
    """
    # ── Guard: only load from UNLOADED or ERROR ──────────────────────────
    if metadata.load_status not in (LoadStatus.UNLOADED, LoadStatus.ERROR):
        logger.warning(
            "Cannot load shard in status %s — must be UNLOADED or ERROR",
            metadata.load_status.name,
        )
        return metadata

    # ── Phase 1: DOWNLOADING ─────────────────────────────────────────────
    metadata.load_status = LoadStatus.DOWNLOADING
    metadata.error_message = None
    logger.info(
        "Loading shard for model '%s', layers %d–%d (%s) from %s",
        metadata.model_id,
        metadata.layer_range.start,
        metadata.layer_range.end,
        metadata.dtype.name,
        metadata.model_url,
    )

    try:
        # Fetch safetensors header
        header_size, all_tensors = fetch_safetensors_header(
            metadata.model_url, timeout=timeout,
        )
        logger.info(
            "Fetched safetensors header: %d tensors, header_size=%d bytes",
            len(all_tensors),
            header_size,
        )

        # Filter to only the tensors this node needs
        assigned_tensors = filter_tensors_for_assignment(
            all_tensors,
            metadata.layer_range,
            is_first_node=is_first_node,
            is_final_node=is_final_node,
        )
        if not assigned_tensors:
            raise SafetensorsHeaderError(
                f"No tensors found for layers {metadata.layer_range.start}"
                f"–{metadata.layer_range.end} in safetensors header"
            )
        logger.info(
            "Filtered %d tensors for layers %d–%d",
            len(assigned_tensors),
            metadata.layer_range.start,
            metadata.layer_range.end,
        )

        # Compute total bytes for progress tracking
        metadata.bytes_total = sum(t.byte_size for t in assigned_tensors.values())
        metadata.bytes_downloaded = 0

        def _progress(name: str, resolved: int, total: int) -> None:
            metadata.bytes_downloaded = resolved

        # Download with cache support
        raw_data = download_selected_tensors_cached(
            metadata.model_url,
            header_size,
            assigned_tensors,
            metadata.cache_dir,
            metadata.model_id,
            timeout=timeout,
            progress_callback=_progress,
        )

    except (SafetensorsHeaderError, OSError) as exc:
        metadata.load_status = LoadStatus.ERROR
        metadata.error_message = f"Download failed: {exc}"
        logger.error("Shard download failed: %s", exc)
        return metadata

    # ── Phase 2: LOADING (deserialize + move to device) ──────────────────
    metadata.load_status = LoadStatus.LOADING
    logger.info("Deserializing %d tensors → device '%s'", len(raw_data), device)

    try:
        loaded: dict[str, object] = {}
        total_bytes_on_device = 0

        for name, raw_bytes in raw_data.items():
            tensor_info = assigned_tensors[name]
            tensor = _reconstruct_tensor(
                raw_bytes, tensor_info, device=device,
            )
            loaded[name] = tensor
            total_bytes_on_device += len(raw_bytes)

        metadata.loaded_tensors = loaded
        metadata.memory_footprint_mb = total_bytes_on_device / (1024 * 1024)

    except (RuntimeError, ValueError, OSError) as exc:
        metadata.load_status = LoadStatus.ERROR
        metadata.error_message = f"Tensor loading failed: {exc}"
        metadata.loaded_tensors = {}
        logger.error("Tensor loading failed: %s", exc)
        return metadata

    # ── Phase 3: READY ───────────────────────────────────────────────────
    metadata.load_status = LoadStatus.READY
    logger.info(
        "Shard ready: %d tensors, %.1f MB on '%s'",
        len(metadata.loaded_tensors),
        metadata.memory_footprint_mb,
        device,
    )
    return metadata


# ── Shard Validation ────────────────────────────────────────────────────────

# Maps ShardDType enum values to the safetensors dtype strings used in
# TensorInfo.  Used by validate_shard to compare the expected dtype
# against what was actually loaded.
_SHARD_DTYPE_TO_SAFETENSORS: dict[ShardDType, str] = {
    ShardDType.FP16: "F16",
    ShardDType.INT8: "I8",
}


class ShardValidationError(Exception):
    """Raised when a loaded shard fails validation checks."""


def validate_shard(
    metadata: ShardMetadata,
    assigned_tensors: dict[str, TensorInfo],
) -> ShardMetadata:
    """Validate that a loaded shard matches the expected configuration.

    Performs three checks:

    1. **Layer count** — the number of unique transformer layer indices
       found in *assigned_tensors* must equal
       ``metadata.layer_range.count`` (i.e. ``layer_end - layer_start + 1``).
    2. **Dtype match** — every layer tensor's safetensors dtype must match
       the expected dtype derived from ``metadata.dtype``.
    3. **Hidden dimension consistency** — the first dimension (``shape[0]``)
       of every weight tensor within each layer must be identical across
       all layers, ensuring the hidden size is uniform.

    On success the shard status remains ``READY``.  On failure the status
    transitions to ``ERROR`` with a descriptive message.

    Parameters
    ----------
    metadata:
        Mutable :class:`ShardMetadata` for the shard to validate.
        Must be in ``READY`` status with ``loaded_tensors`` populated.
    assigned_tensors:
        Tensor metadata dict (as returned by
        :func:`filter_tensors_for_assignment`) describing the tensors
        that were loaded.

    Returns
    -------
    ShardMetadata
        The same *metadata* object, potentially with ``load_status``
        set to ``ERROR`` if validation failed.

    Raises
    ------
    ShardValidationError
        If the shard is not in ``READY`` status when validation is
        attempted.
    """
    if metadata.load_status != LoadStatus.READY:
        raise ShardValidationError(
            f"Cannot validate shard in status {metadata.load_status.name} "
            f"— must be READY"
        )

    expected_layer_count = metadata.layer_range.count
    expected_safetensors_dtype = _SHARD_DTYPE_TO_SAFETENSORS.get(metadata.dtype)

    # ── 1. Layer count check ─────────────────────────────────────────────
    layer_indices: set[int] = set()
    for name in assigned_tensors:
        m = _LAYER_PATTERN.match(name)
        if m is not None:
            layer_indices.add(int(m.group(1)))

    actual_layer_count = len(layer_indices)
    if actual_layer_count != expected_layer_count:
        msg = (
            f"Layer count mismatch: expected {expected_layer_count} layers "
            f"(range {metadata.layer_range.start}–{metadata.layer_range.end}), "
            f"but found {actual_layer_count} unique layer indices "
            f"({sorted(layer_indices)})"
        )
        logger.error("Shard validation failed: %s", msg)
        metadata.load_status = LoadStatus.ERROR
        metadata.error_message = msg
        return metadata

    # ── 2. Dtype match ───────────────────────────────────────────────────
    if expected_safetensors_dtype is not None:
        mismatched: list[str] = []
        for name, info in assigned_tensors.items():
            # Only check layer tensors — embedding/lm_head may use different dtypes
            if _LAYER_PATTERN.match(name) is None:
                continue
            if info.dtype != expected_safetensors_dtype:
                mismatched.append(
                    f"  {name}: expected {expected_safetensors_dtype}, "
                    f"got {info.dtype}"
                )

        if mismatched:
            details = "\n".join(mismatched)
            msg = (
                f"Dtype mismatch for {len(mismatched)} tensor(s) "
                f"(expected {expected_safetensors_dtype} / "
                f"{metadata.dtype.name}):\n{details}"
            )
            logger.error("Shard validation failed: %s", msg)
            metadata.load_status = LoadStatus.ERROR
            metadata.error_message = msg
            return metadata

    # ── 3. Hidden dimension consistency ──────────────────────────────────
    # Collect the first shape dimension of every layer weight tensor.
    # We group by layer index and check that all layers share the same
    # hidden dimension.
    hidden_dims_by_layer: dict[int, set[int]] = {}
    for name, info in assigned_tensors.items():
        m = _LAYER_PATTERN.match(name)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        if len(info.shape) < 1:
            continue
        dim = info.shape[0]
        hidden_dims_by_layer.setdefault(layer_idx, set()).add(dim)

    # Gather the union of all hidden dims seen across all layers
    all_hidden_dims: set[int] = set()
    for dims in hidden_dims_by_layer.values():
        all_hidden_dims.update(dims)

    if len(all_hidden_dims) > 1:
        # Build a readable summary of which layers have which dims
        per_layer_summary: list[str] = []
        for layer_idx in sorted(hidden_dims_by_layer):
            dims = hidden_dims_by_layer[layer_idx]
            per_layer_summary.append(
                f"  layer {layer_idx}: {sorted(dims)}"
            )
        details = "\n".join(per_layer_summary)
        msg = (
            f"Inconsistent hidden dimensions across layers — "
            f"found {len(all_hidden_dims)} distinct values "
            f"({sorted(all_hidden_dims)}):\n{details}"
        )
        logger.error("Shard validation failed: %s", msg)
        metadata.load_status = LoadStatus.ERROR
        metadata.error_message = msg
        return metadata

    logger.info(
        "Shard validation passed: %d layers, dtype=%s, hidden_dims=%s",
        actual_layer_count,
        metadata.dtype.name,
        sorted(all_hidden_dims) if all_hidden_dims else "N/A",
    )
    return metadata


# ── Shard Unloading ─────────────────────────────────────────────────────────


def unload_shard(
    metadata: ShardMetadata,
    *,
    memory_freed_callback: Optional[object] = None,
) -> ShardMetadata:
    """Unload a shard by releasing GPU memory and resetting state.

    Deletes all loaded tensors from GPU memory (via ``del`` + optional
    ``torch.cuda.empty_cache()``), resets the metadata to ``UNLOADED``,
    and reports the freed memory through an optional callback intended
    for the Resource Monitor.

    Parameters
    ----------
    metadata:
        Mutable :class:`ShardMetadata` for the shard to unload.
    memory_freed_callback:
        Optional callable ``(freed_mb: float) -> None`` invoked after
        tensors are released, reporting how much GPU memory was freed.
        Designed for integration with the Resource Monitor.

    Returns
    -------
    ShardMetadata
        The same *metadata* object, updated to ``UNLOADED`` status with
        cleared tensors and zeroed memory footprint.
    """
    if metadata.load_status == LoadStatus.UNLOADED and not metadata.loaded_tensors:
        logger.debug("Shard already unloaded — nothing to do")
        return metadata

    freed_mb = metadata.memory_footprint_mb
    tensor_count = len(metadata.loaded_tensors)

    logger.info(
        "Unloading shard for model '%s', layers %d–%d: "
        "%d tensors, %.1f MB to free",
        metadata.model_id,
        metadata.layer_range.start,
        metadata.layer_range.end,
        tensor_count,
        freed_mb,
    )

    # ── Release tensor references ────────────────────────────────────────
    # Clearing the dict drops all Python references to the GPU tensors,
    # allowing PyTorch's memory allocator to reclaim the device memory.
    metadata.loaded_tensors.clear()

    # Trigger CUDA cache cleanup if torch is available
    if _TORCH_AVAILABLE:
        try:
            torch.cuda.empty_cache()
        except Exception:
            # Not critical — cache cleanup is best-effort (e.g. CPU-only env)
            pass

    # ── Reset metadata ───────────────────────────────────────────────────
    metadata.load_status = LoadStatus.UNLOADED
    metadata.memory_footprint_mb = 0.0
    metadata.bytes_downloaded = 0
    metadata.bytes_total = 0
    metadata.error_message = None

    # ── Report freed memory to Resource Monitor ──────────────────────────
    if memory_freed_callback is not None and freed_mb > 0:
        try:
            memory_freed_callback(freed_mb)  # type: ignore[operator]
        except Exception as exc:
            logger.warning("memory_freed_callback failed: %s", exc)

    logger.info(
        "Shard unloaded: freed %.1f MB from %d tensors",
        freed_mb,
        tensor_count,
    )
    return metadata


# ── Shard Info ───────────────────────────────────────────────────────────────


def get_shard_info(metadata: ShardMetadata) -> ShardInfo:
    """Return an immutable snapshot of the current shard metadata.

    Creates a frozen :class:`ShardInfo` from the mutable
    :class:`ShardMetadata`, capturing model identity, layer range,
    dtype, memory footprint, load status, and download progress
    (bytes downloaded / total bytes for assigned layers).

    Parameters
    ----------
    metadata:
        The mutable :class:`ShardMetadata` to snapshot.

    Returns
    -------
    ShardInfo
        A frozen, read-only view of the shard's current state.
    """
    return ShardInfo(
        model_id=metadata.model_id,
        model_url=metadata.model_url,
        layer_start=metadata.layer_range.start,
        layer_end=metadata.layer_range.end,
        dtype=metadata.dtype,
        memory_footprint_mb=metadata.memory_footprint_mb,
        load_status=metadata.load_status,
        bytes_downloaded=metadata.bytes_downloaded,
        bytes_total=metadata.bytes_total,
        download_progress=metadata.download_progress,
    )

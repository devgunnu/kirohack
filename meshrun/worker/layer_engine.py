"""
Layer Engine — sequential forward pass through hosted transformer layers.

Executes inference on the locally hosted contiguous block of transformer
layers.  Takes input hidden states, runs them through each layer
sequentially, and produces output hidden states (or logits if this is the
final node in the pipeline).

The engine operates on raw weight tensors loaded by the Shard Manager.
It groups per-layer tensors (attention projections, MLP weights, norms)
into ordered ``TransformerLayer`` blocks and applies them in sequence.

Validates: Requirements 4.1, 4.2
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

_LAYER_INDEX_PATTERN = re.compile(r"model\.layers\.(\d+)\.")
"""Regex to extract the layer index from a safetensors tensor name."""


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass(slots=True)
class TransformerLayer:
    """Weight tensors for a single transformer layer.

    Groups the individual weight tensors (attention Q/K/V/O projections,
    MLP gate/up/down projections, and layer norms) that belong to one
    transformer block.  All tensors are expected to reside on the same
    device.
    """

    index: int
    """Layer index within the full model (e.g. 5 for ``model.layers.5``)."""


    # ── Attention weights ────────────────────────────────────────────────
    q_proj: Optional[object] = None
    """Query projection weight ``[hidden_dim, hidden_dim]``."""

    k_proj: Optional[object] = None
    """Key projection weight ``[hidden_dim, hidden_dim]``."""

    v_proj: Optional[object] = None
    """Value projection weight ``[hidden_dim, hidden_dim]``."""

    o_proj: Optional[object] = None
    """Output projection weight ``[hidden_dim, hidden_dim]``."""

    # ── MLP weights ──────────────────────────────────────────────────────
    gate_proj: Optional[object] = None
    """MLP gate projection weight ``[intermediate_dim, hidden_dim]``."""

    up_proj: Optional[object] = None
    """MLP up projection weight ``[intermediate_dim, hidden_dim]``."""

    down_proj: Optional[object] = None
    """MLP down projection weight ``[hidden_dim, intermediate_dim]``."""

    # ── Layer norms ──────────────────────────────────────────────────────
    input_layernorm: Optional[object] = None
    """RMSNorm weight before attention ``[hidden_dim]``."""

    post_attention_layernorm: Optional[object] = None
    """RMSNorm weight after attention / before MLP ``[hidden_dim]``."""


@dataclass(slots=True)
class LayerEngine:
    """Manages sequential forward pass through hosted transformer layers.

    After construction via :func:`build_layer_engine`, call
    :meth:`forward` to run inference on input hidden states.

    Attributes
    ----------
    layers : list[TransformerLayer]
        Ordered list of transformer layer weight groups.
    layer_start : int
        First layer index (inclusive) hosted by this engine.
    layer_end : int
        Last layer index (inclusive) hosted by this engine.
    is_final_node : bool
        If ``True``, apply the LM head after the last layer to produce
        logits instead of hidden states.
    lm_head_weight : optional tensor
        LM head projection weight (only present when *is_final_node*).
    model_norm_weight : optional tensor
        Final RMSNorm weight applied before the LM head.
    num_heads : int
        Number of attention heads (inferred from weight shapes).
    head_dim : int
        Dimension per attention head (inferred from weight shapes).
    rms_norm_eps : float
        Epsilon for RMSNorm numerical stability.
    """

    layers: list[TransformerLayer] = field(default_factory=list)
    layer_start: int = 0
    layer_end: int = 0
    is_final_node: bool = False
    lm_head_weight: Optional[object] = None
    model_norm_weight: Optional[object] = None
    num_heads: int = 0
    head_dim: int = 0
    rms_norm_eps: float = 1e-5


# ── Helper Functions ─────────────────────────────────────────────────────────

def _rms_norm(
    hidden_states: object,
    weight: object,
    eps: float = 1e-5,
) -> object:
    """Apply RMSNorm: ``x * weight / rms(x)``.

    Parameters
    ----------
    hidden_states:
        Input tensor of shape ``[..., hidden_dim]``.
    weight:
        Norm weight of shape ``[hidden_dim]``.
    eps:
        Small constant for numerical stability.

    Returns
    -------
    torch.Tensor
        Normalised tensor, same shape as input.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(input_dtype)


def _apply_attention(
    hidden_states: object,
    layer: TransformerLayer,
    num_heads: int,
    head_dim: int,
) -> object:
    """Simplified multi-head self-attention (no KV-cache, no causal mask).

    For the hackathon POC this implements a basic attention pass:
    Q/K/V projections → scaled dot-product attention → output projection.
    No positional encoding or causal masking is applied (stateless
    single-pass inference).

    Parameters
    ----------
    hidden_states:
        Input tensor ``[batch, seq_len, hidden_dim]``.
    layer:
        Transformer layer containing Q/K/V/O projection weights.
    num_heads:
        Number of attention heads.
    head_dim:
        Dimension per attention head.

    Returns
    -------
    torch.Tensor
        Attention output ``[batch, seq_len, hidden_dim]``.
    """
    batch, seq_len, hidden_dim = hidden_states.shape

    q = F.linear(hidden_states, layer.q_proj)
    k = F.linear(hidden_states, layer.k_proj)
    v = F.linear(hidden_states, layer.v_proj)

    # Reshape to [batch, num_heads, seq_len, head_dim]
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v)

    # Reshape back to [batch, seq_len, hidden_dim]
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        batch, seq_len, hidden_dim,
    )

    return F.linear(attn_output, layer.o_proj)


def _apply_mlp(hidden_states: object, layer: TransformerLayer) -> object:
    """SwiGLU MLP: ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Parameters
    ----------
    hidden_states:
        Input tensor ``[batch, seq_len, hidden_dim]``.
    layer:
        Transformer layer containing gate/up/down projection weights.

    Returns
    -------
    torch.Tensor
        MLP output ``[batch, seq_len, hidden_dim]``.
    """
    gate = F.linear(hidden_states, layer.gate_proj)
    up = F.linear(hidden_states, layer.up_proj)
    return F.linear(F.silu(gate) * up, layer.down_proj)


def _apply_transformer_layer(
    hidden_states: object,
    layer: TransformerLayer,
    num_heads: int,
    head_dim: int,
    rms_norm_eps: float,
) -> object:
    """Apply a single transformer layer (pre-norm architecture).

    Follows the LLaMA / Mistral pattern::

        residual = x
        x = rms_norm(x, input_layernorm)
        x = attention(x)
        x = residual + x
        residual = x
        x = rms_norm(x, post_attention_layernorm)
        x = mlp(x)
        x = residual + x

    Parameters
    ----------
    hidden_states:
        Input tensor ``[batch, seq_len, hidden_dim]``.
    layer:
        Transformer layer weight group.
    num_heads:
        Number of attention heads.
    head_dim:
        Dimension per attention head.
    rms_norm_eps:
        Epsilon for RMSNorm.

    Returns
    -------
    torch.Tensor
        Output hidden states ``[batch, seq_len, hidden_dim]``.
    """
    # ── Self-attention block ─────────────────────────────────────────────
    residual = hidden_states
    hidden_states = _rms_norm(hidden_states, layer.input_layernorm, rms_norm_eps)
    hidden_states = _apply_attention(hidden_states, layer, num_heads, head_dim)
    hidden_states = residual + hidden_states

    # ── MLP block ────────────────────────────────────────────────────────
    residual = hidden_states
    hidden_states = _rms_norm(
        hidden_states, layer.post_attention_layernorm, rms_norm_eps,
    )
    hidden_states = _apply_mlp(hidden_states, layer)
    hidden_states = residual + hidden_states

    return hidden_states


# ── Engine Construction ──────────────────────────────────────────────────────

# Mapping from safetensors tensor name suffixes to TransformerLayer fields.
_WEIGHT_SUFFIX_MAP: dict[str, str] = {
    "self_attn.q_proj.weight": "q_proj",
    "self_attn.k_proj.weight": "k_proj",
    "self_attn.v_proj.weight": "v_proj",
    "self_attn.o_proj.weight": "o_proj",
    "mlp.gate_proj.weight": "gate_proj",
    "mlp.up_proj.weight": "up_proj",
    "mlp.down_proj.weight": "down_proj",
    "input_layernorm.weight": "input_layernorm",
    "post_attention_layernorm.weight": "post_attention_layernorm",
}


def _extract_layer_index(tensor_name: str) -> Optional[int]:
    """Extract the layer index from a tensor name like ``model.layers.5.xxx``.

    Returns ``None`` if the name does not match the expected pattern.
    """
    m = _LAYER_INDEX_PATTERN.search(tensor_name)
    if m is None:
        return None
    return int(m.group(1))


def _infer_head_config(layers: list[TransformerLayer]) -> tuple[int, int]:
    """Infer ``(num_heads, head_dim)`` from the Q-projection weight shape.

    The Q-projection weight has shape ``[hidden_dim, hidden_dim]`` for
    standard multi-head attention.  We assume ``head_dim = hidden_dim //
    num_heads`` and try common head dimensions (128, 64, 96, 80) to find
    one that divides evenly.

    Returns
    -------
    tuple[int, int]
        ``(num_heads, head_dim)``

    Raises
    ------
    ValueError
        If no Q-projection weight is found or the hidden dimension
        cannot be factored into a supported head configuration.
    """
    for layer in layers:
        if layer.q_proj is not None:
            hidden_dim = layer.q_proj.shape[0]
            # Try common head dimensions in order of likelihood
            for candidate_head_dim in (128, 64, 96, 80, 32):
                if hidden_dim % candidate_head_dim == 0:
                    num_heads = hidden_dim // candidate_head_dim
                    return num_heads, candidate_head_dim
            raise ValueError(
                f"Cannot infer head config: hidden_dim={hidden_dim} "
                f"is not divisible by any common head_dim"
            )
    raise ValueError("No Q-projection weight found — cannot infer head config")


def build_layer_engine(
    loaded_tensors: dict[str, object],
    layer_start: int,
    layer_end: int,
    *,
    is_final_node: bool = False,
    rms_norm_eps: float = 1e-5,
) -> LayerEngine:
    """Build a :class:`LayerEngine` from the Shard Manager's loaded tensors.

    Groups the flat ``tensor_name → tensor`` mapping into ordered
    :class:`TransformerLayer` blocks and infers the attention head
    configuration from weight shapes.

    Parameters
    ----------
    loaded_tensors:
        Mapping of safetensors tensor names to ``torch.Tensor`` objects
        on the target device (as produced by
        :func:`~meshrun.worker.shard_manager.load_shard`).
    layer_start:
        First layer index (inclusive) hosted by this node.
    layer_end:
        Last layer index (inclusive) hosted by this node.
    is_final_node:
        Whether this node is the last in the pipeline (applies LM head).
    rms_norm_eps:
        Epsilon for RMSNorm.

    Returns
    -------
    LayerEngine
        Fully initialised engine ready for :meth:`LayerEngine.forward`.

    Raises
    ------
    RuntimeError
        If PyTorch is not available.
    ValueError
        If required layer weights are missing or head config cannot be
        inferred.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for the Layer Engine but is not installed"
        )

    # ── Group tensors by layer index ─────────────────────────────────────
    layer_map: dict[int, TransformerLayer] = {}
    lm_head_weight: Optional[object] = None
    model_norm_weight: Optional[object] = None

    for tensor_name, tensor in loaded_tensors.items():
        # Check for final-node special tensors
        if tensor_name == "lm_head.weight":
            lm_head_weight = tensor
            continue
        if tensor_name == "model.norm.weight":
            model_norm_weight = tensor
            continue

        # Extract layer index
        layer_idx = _extract_layer_index(tensor_name)
        if layer_idx is None:
            logger.debug("Skipping non-layer tensor: %s", tensor_name)
            continue

        # Only include tensors within our assigned range
        if layer_idx < layer_start or layer_idx > layer_end:
            logger.debug(
                "Skipping out-of-range tensor: %s (layer %d not in %d–%d)",
                tensor_name, layer_idx, layer_start, layer_end,
            )
            continue

        # Get or create the TransformerLayer for this index
        if layer_idx not in layer_map:
            layer_map[layer_idx] = TransformerLayer(index=layer_idx)

        layer = layer_map[layer_idx]

        # Match tensor name suffix to the appropriate field
        for suffix, attr_name in _WEIGHT_SUFFIX_MAP.items():
            if tensor_name.endswith(suffix):
                setattr(layer, attr_name, tensor)
                break

    # ── Sort layers by index ─────────────────────────────────────────────
    sorted_layers = [
        layer_map[i]
        for i in sorted(layer_map.keys())
    ]

    if not sorted_layers:
        raise ValueError(
            f"No transformer layers found for range {layer_start}–{layer_end}"
        )

    # ── Validate contiguity ──────────────────────────────────────────────
    expected_indices = list(range(layer_start, layer_end + 1))
    actual_indices = [layer.index for layer in sorted_layers]
    if actual_indices != expected_indices:
        missing = set(expected_indices) - set(actual_indices)
        raise ValueError(
            f"Missing layers in range {layer_start}–{layer_end}: {sorted(missing)}"
        )

    # ── Infer head configuration ─────────────────────────────────────────
    num_heads, head_dim = _infer_head_config(sorted_layers)
    logger.info(
        "Layer engine built: layers %d–%d, %d heads × %d dim, "
        "is_final=%s",
        layer_start, layer_end, num_heads, head_dim, is_final_node,
    )

    return LayerEngine(
        layers=sorted_layers,
        layer_start=layer_start,
        layer_end=layer_end,
        is_final_node=is_final_node,
        lm_head_weight=lm_head_weight,
        model_norm_weight=model_norm_weight,
        num_heads=num_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
    )


# ── Output Validation ────────────────────────────────────────────────────────

def _validate_output(
    engine: LayerEngine,
    output: object,
    step_id: int,
) -> None:
    """Validate forward pass output tensor shape and numeric integrity.

    Checks that the output has the expected dimensionality and shape for
    the engine's role (intermediate node vs final node), and that no
    NaN or Inf values are present.

    Parameters
    ----------
    engine:
        The layer engine that produced the output.
    output:
        Output tensor from the forward pass.
    step_id:
        Token generation step (for error messages).

    Raises
    ------
    ValueError
        If the output shape is incorrect or contains NaN/Inf values.
    """
    if output.ndim != 3:
        raise ValueError(
            f"Output validation failed (step_id={step_id}): expected 3-D "
            f"tensor, got {output.ndim}-D with shape {tuple(output.shape)}"
        )

    batch, seq_len, last_dim = output.shape

    if engine.is_final_node:
        # Final node produces logits: [batch, seq_len, vocab_size]
        if engine.lm_head_weight is not None:
            expected_vocab = engine.lm_head_weight.shape[0]
            if last_dim != expected_vocab:
                raise ValueError(
                    f"Output validation failed (step_id={step_id}): final node "
                    f"expected vocab_size={expected_vocab}, got {last_dim}"
                )
    else:
        # Intermediate node produces hidden states: [batch, seq_len, hidden_dim]
        expected_hidden = engine.num_heads * engine.head_dim
        if last_dim != expected_hidden:
            raise ValueError(
                f"Output validation failed (step_id={step_id}): expected "
                f"hidden_dim={expected_hidden}, got {last_dim}"
            )

    if torch.isnan(output).any():
        raise ValueError(
            f"Output validation failed (step_id={step_id}): output tensor "
            f"contains NaN values"
        )

    if torch.isinf(output).any():
        raise ValueError(
            f"Output validation failed (step_id={step_id}): output tensor "
            f"contains Inf values"
        )


# ── Forward Pass ─────────────────────────────────────────────────────────────

def forward(
    engine: LayerEngine,
    hidden_states: object,
    step_id: int,
) -> object:
    """Run a sequential forward pass through all hosted layers.

    Applies each transformer layer in order to the input hidden states.
    If the engine is configured as the final node, the model's final
    RMSNorm and LM head projection are applied to produce logits.

    Parameters
    ----------
    engine:
        A :class:`LayerEngine` built via :func:`build_layer_engine`.
    hidden_states:
        Input tensor of shape ``[batch, seq_len, hidden_dim]``.
        Must reside on the same device as the engine's weights.
    step_id:
        Token generation step (0 for prefill, increments for
        autoregressive decoding).  Currently used for logging only.

    Returns
    -------
    torch.Tensor
        If *engine.is_final_node* is ``False``:
            Hidden states ``[batch, seq_len, hidden_dim]`` for the next
            node in the pipeline.
        If *engine.is_final_node* is ``True``:
            Logits ``[batch, seq_len, vocab_size]`` for decoding.

    Raises
    ------
    RuntimeError
        If PyTorch is not available.
    ValueError
        If *hidden_states* has an unexpected number of dimensions or if
        the final node is missing the LM head weight.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for forward pass but is not installed"
        )

    if hidden_states.ndim != 3:
        raise ValueError(
            f"Expected 3-D hidden_states [batch, seq_len, hidden_dim], "
            f"got {hidden_states.ndim}-D tensor with shape {tuple(hidden_states.shape)}"
        )

    logger.debug(
        "Forward pass: step_id=%d, layers %d–%d, input shape=%s",
        step_id,
        engine.layer_start,
        engine.layer_end,
        tuple(hidden_states.shape),
    )

    # ── Sequential layer execution ───────────────────────────────────────
    for layer in engine.layers:
        hidden_states = _apply_transformer_layer(
            hidden_states,
            layer,
            engine.num_heads,
            engine.head_dim,
            engine.rms_norm_eps,
        )

    # ── Final node: apply model norm + LM head ──────────────────────────
    if engine.is_final_node:
        if engine.lm_head_weight is None:
            raise ValueError(
                "Final node requires lm_head_weight but none was loaded"
            )
        # Apply final RMSNorm if available
        if engine.model_norm_weight is not None:
            hidden_states = _rms_norm(
                hidden_states, engine.model_norm_weight, engine.rms_norm_eps,
            )
        # Project to vocabulary logits
        hidden_states = F.linear(hidden_states, engine.lm_head_weight)

    # ── Output validation ───────────────────────────────────────────────
    _validate_output(engine, hidden_states, step_id)

    logger.debug(
        "Forward pass complete: step_id=%d, output shape=%s",
        step_id,
        tuple(hidden_states.shape),
    )

    return hidden_states


# ── Warm-Up ──────────────────────────────────────────────────────────────────

def warm_up(
    engine: LayerEngine,
    *,
    batch_size: int = 1,
    seq_len: int = 1,
) -> None:
    """Run a dummy forward pass to compile GPU kernels and pre-allocate memory.

    Creates a zero-filled tensor matching the engine's hidden dimension
    and runs it through :func:`forward`.  This forces CUDA kernel
    compilation and activation memory allocation so the first real
    inference request doesn't pay the warm-up cost.

    Parameters
    ----------
    engine:
        A :class:`LayerEngine` built via :func:`build_layer_engine`.
    batch_size:
        Batch size for the dummy tensor (default 1).
    seq_len:
        Sequence length for the dummy tensor (default 1).

    Raises
    ------
    RuntimeError
        If PyTorch is not available or if the engine has no layers with
        weight tensors to infer the hidden dimension from.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for warm-up but is not installed"
        )

    # Infer hidden_dim and device from the first layer's Q-projection
    hidden_dim: Optional[int] = None
    device = None
    dtype = None
    for layer in engine.layers:
        if layer.q_proj is not None:
            hidden_dim = layer.q_proj.shape[0]
            device = layer.q_proj.device
            dtype = layer.q_proj.dtype
            break

    if hidden_dim is None:
        raise RuntimeError(
            "Cannot warm up: no Q-projection weight found to infer "
            "hidden dimension"
        )

    logger.info(
        "Warming up layer engine: layers %d–%d, dummy shape=[%d, %d, %d], "
        "device=%s, dtype=%s",
        engine.layer_start, engine.layer_end,
        batch_size, seq_len, hidden_dim,
        device, dtype,
    )

    dummy = torch.zeros(
        batch_size, seq_len, hidden_dim,
        device=device, dtype=dtype,
    )

    with torch.no_grad():
        _ = forward(engine, dummy, step_id=0)

    logger.info("Warm-up complete for layers %d–%d", engine.layer_start, engine.layer_end)

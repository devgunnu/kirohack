"""Tokenizer and embedding engine for the MeshRun inference client.

Loads a HuggingFace AutoTokenizer matching the served model, selectively
downloads embedding weights from a safetensors model file, and provides
local embedding lookup and logits decoding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy torch import — only required when embedding operations are used.
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

# Lazy transformers import — only required for tokenizer loading.
try:
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False


_EMBEDDING_FRAGMENTS = ("embed_tokens",)
"""Tensor name fragments that identify embedding weight tensors."""


class ModelTokenizer:
    """Tokenizer and embedding engine for a single model.

    Provides tokenization (text → token IDs), embedding lookup
    (token IDs → hidden states tensor), detokenization (token IDs → text),
    and greedy logits decoding.
    """

    def __init__(self) -> None:
        self._tokenizer: Optional[object] = None
        self._embedding_weight: Optional[object] = None  # torch.Tensor when loaded


    # ── Tokenizer loading ───────────────────────────────────────────────

    def load_tokenizer(self, model_name_or_path: str) -> None:
        """Load a HuggingFace AutoTokenizer for the given model.

        Parameters
        ----------
        model_name_or_path:
            HuggingFace model ID (e.g. ``'meta-llama/Llama-3.2-3B'``)
            or local path to a tokenizer directory.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "The 'transformers' package is required for tokenizer loading "
                "but is not installed"
            )
        logger.info("Loading tokenizer for '%s'", model_name_or_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        logger.info("Tokenizer loaded successfully")

    # ── Tokenize / Detokenize ───────────────────────────────────────────

    def tokenize(self, text: str) -> list[int]:
        """Convert raw text to a list of token IDs.

        Parameters
        ----------
        text:
            Input text string.

        Returns
        -------
        list[int]
            Token IDs produced by the loaded tokenizer.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded — call load_tokenizer() first")
        return self._tokenizer.encode(text)  # type: ignore[union-attr]

    def detokenize(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text.

        Parameters
        ----------
        token_ids:
            List of integer token IDs.

        Returns
        -------
        str
            Decoded text string.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded — call load_tokenizer() first")
        return self._tokenizer.decode(token_ids)  # type: ignore[union-attr]

    # ── Embedding weight loading ────────────────────────────────────────

    def load_embedding(
        self,
        model_url: str,
        cache_dir: str | Path,
        device: str = "cpu",
    ) -> None:
        """Selectively download and load the embedding weight tensor.

        Uses the shard manager infrastructure to fetch only the
        ``embed_tokens`` tensor(s) from the safetensors model file via
        HTTP Range requests, with local caching.

        Parameters
        ----------
        model_url:
            HTTP(S) URL to the ``.safetensors`` model file.
        cache_dir:
            Local directory for caching downloaded tensor data.
        device:
            Target torch device (e.g. ``'cpu'``, ``'cuda'``).
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for embedding loading but is not installed"
            )

        from meshrun.worker.shard_manager import (
            TensorInfo,
            _reconstruct_tensor,
            download_selected_tensors_cached,
            fetch_safetensors_header,
        )

        cache_path = Path(cache_dir)

        # 1. Fetch the safetensors header to get tensor metadata
        logger.info("Fetching safetensors header from '%s'", model_url)
        header_size, all_tensors = fetch_safetensors_header(model_url)

        # 2. Filter for embed_tokens tensors
        embed_tensors: dict[str, TensorInfo] = {}
        for name, info in all_tensors.items():
            if any(frag in name for frag in _EMBEDDING_FRAGMENTS):
                embed_tensors[name] = info

        if not embed_tensors:
            raise ValueError(
                f"No embedding tensors found in model at '{model_url}'. "
                f"Looked for tensor names containing: {_EMBEDDING_FRAGMENTS}"
            )

        logger.info(
            "Found %d embedding tensor(s): %s",
            len(embed_tensors),
            list(embed_tensors.keys()),
        )

        # 3. Download (or load from cache) the embedding bytes
        # Derive a model_id from the URL for cache directory naming
        model_id = model_url.rstrip("/").rsplit("/", 1)[-1].replace(".safetensors", "")
        raw_data = download_selected_tensors_cached(
            url=model_url,
            header_size=header_size,
            tensors=embed_tensors,
            cache_dir=cache_path,
            model_id=model_id,
        )

        # 4. Reconstruct the embedding weight tensor
        # Typically there is one embed_tokens tensor — take the first one
        tensor_name = next(iter(embed_tensors))
        tensor_info = embed_tensors[tensor_name]
        tensor_bytes = raw_data[tensor_name]

        self._embedding_weight = _reconstruct_tensor(
            tensor_bytes, tensor_info, device=device
        )
        logger.info(
            "Embedding weight loaded: shape=%s, dtype=%s, device=%s",
            tensor_info.shape,
            tensor_info.dtype,
            device,
        )

    # ── Embedding execution ─────────────────────────────────────────────

    def embed(self, token_ids: list[int]) -> object:
        """Perform embedding lookup to produce hidden states.

        Parameters
        ----------
        token_ids:
            List of integer token IDs from :meth:`tokenize`.

        Returns
        -------
        torch.Tensor
            Hidden states tensor of shape ``[1, seq_len, hidden_dim]``
            with dtype matching the embedding weight (typically fp16).
        """
        if self._embedding_weight is None:
            raise RuntimeError(
                "Embedding weight not loaded — call load_embedding() first"
            )
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for embedding but is not installed"
            )

        ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        # Direct indexing: embedding_weight[token_ids] → [seq_len, hidden_dim]
        hidden_states = self._embedding_weight[ids_tensor]  # type: ignore[index]
        # Ensure fp16 output as required by the pipeline
        hidden_states = hidden_states.to(dtype=torch.float16)
        # Add batch dimension → [1, seq_len, hidden_dim]
        return hidden_states.unsqueeze(0)

    # ── Logits decoding ─────────────────────────────────────────────────

    @staticmethod
    def decode_logits(logits_tensor: object) -> list[int]:
        """Decode logits via greedy argmax on the last token position.

        Parameters
        ----------
        logits_tensor:
            Logits tensor of shape ``[batch, seq_len, vocab_size]``.

        Returns
        -------
        list[int]
            Output token IDs (one per batch element).
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for logits decoding but is not installed"
            )
        # Greedy argmax on last position: logits[:, -1, :]
        last_logits = logits_tensor[:, -1, :]  # type: ignore[index]
        output_ids = torch.argmax(last_logits, dim=-1)  # type: ignore[attr-defined]
        return output_ids.tolist()

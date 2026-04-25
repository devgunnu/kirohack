"""Main Inference Client for the MeshRun distributed inference pipeline.

Orchestrates the end-to-end inference flow: tokenize → embed → request
route from Coordinator → encrypt & send to pipeline → receive & decrypt
result → decode logits → detokenize → return text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import grpc
import torch

from meshrun.client.tokenizer import ModelTokenizer
from meshrun.client.transport import SecureTransport
from meshrun.coordinator.proto import coordinator_pb2, coordinator_pb2_grpc
from meshrun.worker.protocol import DType, DTYPE_SIZE

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RouteNode:
    """A single node in an execution route (client-side representation)."""

    node_id: str
    address: str
    layer_start: int
    layer_end: int


@dataclass(frozen=True, slots=True)
class ExecutionPath:
    """Execution route returned by the Coordinator."""

    request_id: int
    model_id: str
    session_key: bytes
    nodes: tuple[RouteNode, ...]
    backup_map: dict[str, str]


class InferenceClient:
    """User-facing client for MeshRun distributed inference.

    Connects to the Coordinator via gRPC to obtain execution routes and
    session keys, then communicates with the worker pipeline over
    encrypted TCP to perform inference.

    Parameters
    ----------
    coordinator_address:
        Coordinator gRPC endpoint as ``"host:port"``.
    model_name:
        HuggingFace model ID for tokenizer loading
        (e.g. ``"meta-llama/Llama-3.2-3B"``).
    model_url:
        HTTP(S) URL to the ``.safetensors`` model file for embedding
        weight download.
    cache_dir:
        Local directory for caching downloaded tensor data.
    device:
        Target torch device (e.g. ``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        coordinator_address: str,
        model_name: str,
        model_url: str,
        cache_dir: str = ".cache",
        device: str = "cpu",
    ) -> None:
        self._coordinator_address = coordinator_address
        self._model_name = model_name
        self._model_url = model_url
        self._cache_dir = Path(cache_dir)
        self._device = device

        # gRPC channel and stub (created eagerly, connection is lazy)
        self._channel: grpc.Channel = grpc.insecure_channel(coordinator_address)
        self._stub = coordinator_pb2_grpc.CoordinatorServiceStub(self._channel)

        # Sub-components
        self._tokenizer = ModelTokenizer()
        self._transport = SecureTransport()

        self._initialized = False

        logger.info(
            "InferenceClient created: coordinator=%s, model=%s, device=%s",
            coordinator_address,
            model_name,
            device,
        )

    def initialize(self) -> None:
        """Prepare the client for inference by loading the tokenizer and embedding weights.

        Loads the HuggingFace tokenizer matching ``model_name`` and
        selectively downloads the embedding layer weights from the
        safetensors model file at ``model_url``.

        Raises
        ------
        RuntimeError
            If required dependencies (transformers, torch) are missing
            or if the model/embedding cannot be loaded.
        """
        logger.info("Initializing InferenceClient for model '%s'", self._model_name)
        self._tokenizer.load_tokenizer(self._model_name)
        self._tokenizer.load_embedding(
            model_url=self._model_url,
            cache_dir=str(self._cache_dir),
            device=self._device,
        )
        self._initialized = True
        logger.info("InferenceClient initialized successfully")

    def request_route(self, model_id: str) -> ExecutionPath:
        """Request an execution route from the Coordinator.

        Calls the ``RequestRoute`` gRPC RPC and translates the protobuf
        response into a client-side :class:`ExecutionPath` dataclass.

        Parameters
        ----------
        model_id:
            Identifier of the model pipeline to route through.

        Returns
        -------
        ExecutionPath
            The execution route containing ordered nodes, session key,
            and backup map.

        Raises
        ------
        grpc.RpcError
            If the gRPC call fails (e.g. Coordinator unavailable).
        """
        logger.info("Requesting route for model '%s'", model_id)

        request = coordinator_pb2.RequestRouteRequest(
            model_id=model_id,
            client_id="",
            priority_token=0.0,
        )

        try:
            response = self._stub.RequestRoute(request)
        except grpc.RpcError as exc:
            logger.error("RequestRoute RPC failed: %s", exc)
            raise

        nodes = tuple(
            RouteNode(
                node_id=n.node_id,
                address=n.address,
                layer_start=n.layer_start,
                layer_end=n.layer_end,
            )
            for n in response.nodes
        )

        backup_map: dict[str, str] = {
            entry.node_id: entry.backup_address
            for entry in response.backup_map
        }

        path = ExecutionPath(
            request_id=response.request_id,
            model_id=model_id,
            session_key=bytes(response.session_key),
            nodes=nodes,
            backup_map=backup_map,
        )

        logger.info(
            "Route acquired: request_id=%s, %d node(s)",
            path.request_id,
            len(path.nodes),
        )
        return path

    def submit_inference(self, prompt_text: str) -> str:
        """Run end-to-end inference: text in → text out.

        Orchestrates the full pipeline: tokenize → embed → request route
        → connect to first node → send encrypted FORWARD → receive
        encrypted RESULT → decode logits → detokenize → return text.

        Parameters
        ----------
        prompt_text:
            Raw text prompt to run inference on.

        Returns
        -------
        str
            Generated text output.

        Raises
        ------
        RuntimeError
            If the client has not been initialized, or if any stage of
            the inference pipeline fails.
        """
        if not self._initialized:
            raise RuntimeError(
                "Client not initialized — call initialize() before submit_inference()"
            )

        # 1. Tokenize
        try:
            token_ids = self._tokenizer.tokenize(prompt_text)
        except Exception as exc:
            raise RuntimeError(f"Tokenization failed: {exc}") from exc

        logger.info("Tokenized prompt into %d token(s)", len(token_ids))

        # 2. Embed
        try:
            hidden_states = self._tokenizer.embed(token_ids)
        except Exception as exc:
            raise RuntimeError(f"Embedding failed: {exc}") from exc

        logger.info("Embedded tokens → shape %s", hidden_states.shape)

        # 3. Request route from Coordinator
        # Derive model_id from model_name (use the full name as the id)
        model_id = self._model_name
        try:
            route = self.request_route(model_id)
        except grpc.RpcError as exc:
            raise RuntimeError(
                f"Route acquisition failed: {exc.code().name}: {exc.details()}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Route acquisition failed: {exc}") from exc

        if not route.nodes:
            raise RuntimeError("Route has no nodes — cannot run inference")

        # 4. Connect to the first node in the execution path
        first_node = route.nodes[0]
        try:
            sock = self._transport.connect(first_node.address)
        except (ConnectionError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to connect to first node '{first_node.address}': {exc}"
            ) from exc

        logger.info(
            "Connected to first node %s (%s)",
            first_node.node_id,
            first_node.address,
        )

        try:
            # 5. Send encrypted FORWARD message
            try:
                self._transport.send_forward(
                    sock=sock,
                    hidden_states=hidden_states,
                    session_key=route.session_key,
                    request_id=route.request_id,
                    step_id=0,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to send FORWARD to {first_node.address}: {exc}"
                ) from exc

            logger.info("Sent encrypted FORWARD (request_id=%d)", route.request_id)

            # 6. Receive encrypted RESULT
            try:
                header, tensor_data = self._transport.receive_result(
                    sock=sock,
                    session_key=route.session_key,
                )
            except RuntimeError:
                # Already a RuntimeError from transport (decryption / ERROR msg)
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to receive RESULT: {exc}"
                ) from exc

            logger.info(
                "Received RESULT (request_id=%d, dims=%s)",
                header.request_id,
                header.dims[: header.num_dims],
            )
        finally:
            # Always close the socket after the request completes
            try:
                sock.close()
            except OSError:
                pass

        # 7. Reconstruct logits tensor from flat data
        try:
            active_dims = header.dims[: header.num_dims]
            proto_dtype = DType(header.dtype)
            _DTYPE_TORCH_MAP = {
                DType.FP16: torch.float16,
                DType.INT8: torch.int8,
            }
            torch_dtype = _DTYPE_TORCH_MAP[proto_dtype]
            logits_tensor = torch.tensor(tensor_data, dtype=torch_dtype).reshape(
                active_dims
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to reconstruct logits tensor: {exc}"
            ) from exc

        # 8. Decode logits → token IDs
        try:
            output_token_ids = self._tokenizer.decode_logits(logits_tensor)
        except Exception as exc:
            raise RuntimeError(f"Logits decoding failed: {exc}") from exc

        logger.info("Decoded %d output token(s)", len(output_token_ids))

        # 9. Detokenize → text
        try:
            output_text = self._tokenizer.detokenize(output_token_ids)
        except Exception as exc:
            raise RuntimeError(f"Detokenization failed: {exc}") from exc

        logger.info("Inference complete: %d chars output", len(output_text))
        return output_text

    def close(self) -> None:
        """Release all resources held by this client.

        Closes the gRPC channel to the Coordinator, all TCP sockets
        managed by the secure transport, and resets internal state.
        Safe to call multiple times.
        """
        # Close transport sockets first (data plane)
        try:
            self._transport.close()
        except Exception:
            logger.debug("Error closing transport", exc_info=True)

        # Close gRPC channel (control plane)
        try:
            self._channel.close()
        except Exception:
            logger.debug("Error closing gRPC channel", exc_info=True)

        self._initialized = False
        logger.info("InferenceClient closed")

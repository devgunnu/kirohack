"""
Real inference client for MeshRun CLI.

Wires the CLI commands to the actual backend components.
"""

from __future__ import annotations

import logging
from typing import Any

import grpc

from meshrun.coordinator.proto import coordinator_pb2, coordinator_pb2_grpc
from meshrun.app.config import settings

logger = logging.getLogger(__name__)

# Global client instances (lazy-initialized)
_inference_client: Any = None
_coordinator_channel: grpc.Channel | None = None
_coordinator_stub: coordinator_pb2_grpc.CoordinatorServiceStub | None = None


def _coordinator_address() -> str:
    """Return the coordinator gRPC address with any http(s):// prefix stripped."""
    addr = settings.coordinator_url
    if addr.startswith("http://"):
        addr = addr[len("http://"):]
    elif addr.startswith("https://"):
        addr = addr[len("https://"):]
    return addr


def _get_coordinator_stub() -> coordinator_pb2_grpc.CoordinatorServiceStub:
    """Get or create the gRPC stub for the Coordinator."""
    global _coordinator_channel, _coordinator_stub
    if _coordinator_stub is None:
        addr = _coordinator_address()
        _coordinator_channel = grpc.insecure_channel(addr)
        _coordinator_stub = coordinator_pb2_grpc.CoordinatorServiceStub(_coordinator_channel)
    return _coordinator_stub


def _get_inference_client(model_url: str = "") -> Any:
    """Get or create the InferenceClient.

    Imports the real ``InferenceClient`` lazily so that this module can be
    imported even if the heavy ML dependencies fail to import. The
    ``model_url`` is required for ``initialize()`` (which downloads embedding
    weights) — callers should pass a non-empty URL.
    """
    global _inference_client
    if _inference_client is None:
        from meshrun.client.client import InferenceClient

        addr = _coordinator_address()
        _inference_client = InferenceClient(
            coordinator_address=addr,
            model_name=settings.default_model,
            model_url=model_url,
        )
        _inference_client.initialize()
    return _inference_client


def submit_inference_job(prompt: str, priority: str = "normal") -> dict:
    """Submit a synchronous inference job and return the result."""
    try:
        # The CLI does not currently know the model URL; the InferenceClient
        # requires one for embedding weights. If no model URL is configured
        # we surface a friendly error rather than crashing the CLI.
        import os

        model_url = os.environ.get("MESHRUN_MODEL_URL", "")
        if not model_url and not _inference_client:
            raise RuntimeError(
                "No model URL configured. Set MESHRUN_MODEL_URL to the "
                "safetensors URL before submitting inference."
            )

        client = _get_inference_client(model_url=model_url)
        output = client.submit_inference(prompt)
        return {
            "job_id": "sync",
            "status": "completed",
            "output": output,
            "tokens_generated": 0,  # Not tracked in current impl
            "total_latency": 0.0,
            "hop_latencies": {},
            "cost_saved_usd": 0.0,
            "co2_avoided_g": 0.0,
        }
    except Exception as exc:
        logger.exception("Inference failed")
        return {
            "job_id": "error",
            "status": "failed",
            "output": f"Error: {exc}",
            "tokens_generated": 0,
            "total_latency": 0.0,
            "hop_latencies": {},
            "cost_saved_usd": 0.0,
            "co2_avoided_g": 0.0,
        }


def submit_async_job(prompt: str, priority: str = "normal") -> dict:
    """Submit an async inference job and return queue info."""
    # TODO: Implement async job submission via priority queue
    return {
        "job_id": "not-implemented",
        "status": "error",
        "queue_position": 0,
        "estimated_wait": 0.0,
    }


def get_job_result(job_id: str) -> dict:
    """Retrieve the result of a previously submitted async job."""
    # TODO: Implement job result retrieval
    return {
        "job_id": job_id,
        "status": "not-implemented",
        "output": "",
        "tokens_generated": 0,
        "total_latency": 0.0,
        "hop_latencies": {},
        "cost_saved_usd": 0.0,
        "co2_avoided_g": 0.0,
    }


def register_node(node_id: str, layers: str, vram_gb: float) -> dict:
    """Register a machine as a worker node.

    The actual registration is performed by ``meshrun worker``; this CLI
    helper just informs the user.
    """
    return {
        "success": False,
        "node_id": node_id,
        "layers_assigned": layers,
        "message": "Use 'meshrun worker' to register a node",
    }


def deregister_node(node_id: str) -> dict:
    """Gracefully deregister a worker node via the Coordinator gRPC API."""
    try:
        stub = _get_coordinator_stub()
        request = coordinator_pb2.DeregisterRequest(node_id=node_id)
        response = stub.Deregister(request)
        return {
            "success": getattr(response, "acknowledged", False),
            "message": getattr(response, "message", ""),
        }
    except Exception as exc:
        logger.exception("Deregister failed")
        return {"success": False, "message": str(exc)}


def get_network_status() -> dict:
    """Fetch full network status from the Coordinator via GetNetworkStatus."""
    try:
        stub = _get_coordinator_stub()
        request = coordinator_pb2.GetNetworkStatusRequest()
        response = stub.GetNetworkStatus(request, timeout=5.0)

        nodes = []
        for ni in response.nodes:
            nodes.append({
                "node_id": ni.node_id,
                "address": ni.address,
                "layers": f"{ni.layer_start}-{ni.layer_end}",
                "status": ni.status,
                "credits": ni.credits_earned,
                "latency": "--",
            })

        queue = []
        for qe in response.queue:
            queue.append({
                "job_id": qe.job_id,
                "prompt": qe.prompt_preview,
                "priority": qe.priority_score,
                "wait_time": "--",
                "position": qe.position,
            })

        return {
            "active_nodes": response.active_nodes,
            "total_layers": response.total_layers,
            "covered_layers": response.covered_layers,
            "model": response.model_id or settings.default_model,
            "queue_depth": response.queue_depth,
            "nodes": nodes,
            "queue": queue,
        }
    except Exception as exc:
        logger.warning("Failed to get network status: %s", exc)
        return {
            "active_nodes": 0,
            "total_layers": 0,
            "covered_layers": 0,
            "model": settings.default_model,
            "queue_depth": 0,
            "nodes": [],
            "queue": [],
        }


def get_credits(node_id: str) -> dict:
    """Fetch credit balance and history for a node."""
    # TODO: Implement credits tracking
    return {
        "node_id": node_id,
        "balance": 0.0,
        "compute_contributed_hours": 0.0,
        "priority_score": 0.0,
        "history": [],
    }


def detect_local_hardware() -> dict:
    """Detect local GPU and RAM for node registration."""
    # Try to read RAM via psutil if available, otherwise fall back.
    try:
        import psutil  # type: ignore

        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        ram_gb = 16.0

    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            total_memory_mb = props.total_memory // (1024 * 1024)

            return {
                "vram_gb": round(total_memory_mb / 1024, 1),
                "ram_gb": ram_gb,
                "gpu_name": props.name,
                "suggested_layers": f"0-{min(10, total_memory_mb // 200)}",
                "can_run_model": total_memory_mb >= 6000,
            }
        else:
            return {
                "vram_gb": 0,
                "ram_gb": ram_gb,
                "gpu_name": "No CUDA GPU",
                "suggested_layers": "",
                "can_run_model": False,
            }
    except ImportError:
        return {
            "vram_gb": 0,
            "ram_gb": ram_gb,
            "gpu_name": "PyTorch not installed",
            "suggested_layers": "",
            "can_run_model": False,
        }


def get_earning_rate(compute_pct: int, vram_gb: float) -> float:
    """Calculate credits earned per forward pass based on compute allocation."""
    # TODO: replace with real coordinator formula
    base = 1.2
    return round(base * (compute_pct / 25), 1)

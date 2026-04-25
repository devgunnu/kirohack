"""Kiro extension sidebar integration.

This module provides the API that the Kiro IDE extension calls to display
real-time MeshRun status in the sidebar.
"""

from __future__ import annotations

import logging
from typing import Any

import grpc

from meshrun.coordinator.proto import coordinator_pb2, coordinator_pb2_grpc
from meshrun.app.config import settings

logger = logging.getLogger(__name__)


def _coordinator_address() -> str:
    addr = settings.coordinator_url
    if addr.startswith("http://"):
        addr = addr[len("http://"):]
    elif addr.startswith("https://"):
        addr = addr[len("https://"):]
    return addr


def _classify_health(active_nodes: int, total_layers: int, covered_layers: int) -> str:
    if active_nodes == 0:
        return "down"
    if total_layers > 0 and covered_layers < total_layers:
        return "degraded"
    return "healthy"


def get_sidebar_status() -> dict[str, Any]:
    """Get the current MeshRun status for the Kiro sidebar.

    Returns a dict with:
    - cluster_health: "healthy" | "degraded" | "down"
    - active_nodes: int
    - total_layers: int
    - covered_layers: int
    - queue_depth: int
    - model: str
    - nodes: list of node info dicts
    """
    addr = _coordinator_address()

    try:
        channel = grpc.insecure_channel(addr)
        stub = coordinator_pb2_grpc.CoordinatorServiceStub(channel)
        try:
            request = coordinator_pb2.GetNetworkStatusRequest()
            response = stub.GetNetworkStatus(request, timeout=5.0)
        finally:
            channel.close()

        nodes = [
            {
                "node_id": ni.node_id,
                "address": ni.address,
                "layers": f"{ni.layer_start}-{ni.layer_end}",
                "status": ni.status,
                "gpu_utilization": ni.gpu_utilization,
                "memory_used_mb": ni.memory_used_mb,
                "memory_total_mb": ni.memory_total_mb,
                "credits_earned": ni.credits_earned,
            }
            for ni in response.nodes
        ]

        cluster_health = _classify_health(
            response.active_nodes,
            response.total_layers,
            response.covered_layers,
        )

        return {
            "cluster_health": cluster_health,
            "active_nodes": response.active_nodes,
            "total_layers": response.total_layers,
            "covered_layers": response.covered_layers,
            "queue_depth": response.queue_depth,
            "model": response.model_id or settings.default_model,
            "nodes": nodes,
        }
    except Exception as exc:
        logger.warning("Failed to get sidebar status: %s", exc)
        return {
            "cluster_health": "down",
            "active_nodes": 0,
            "total_layers": 0,
            "covered_layers": 0,
            "queue_depth": 0,
            "model": settings.default_model,
            "nodes": [],
            "error": str(exc),
        }


def get_node_details(node_id: str) -> dict[str, Any]:
    """Get detailed info for a specific node."""
    addr = _coordinator_address()
    try:
        channel = grpc.insecure_channel(addr)
        stub = coordinator_pb2_grpc.CoordinatorServiceStub(channel)
        try:
            request = coordinator_pb2.GetNetworkStatusRequest()
            response = stub.GetNetworkStatus(request, timeout=5.0)
        finally:
            channel.close()

        for ni in response.nodes:
            if ni.node_id == node_id:
                return {
                    "node_id": ni.node_id,
                    "address": ni.address,
                    "grpc_address": ni.grpc_address,
                    "layers": f"{ni.layer_start}-{ni.layer_end}",
                    "status": ni.status,
                    "gpu_utilization": ni.gpu_utilization,
                    "memory_used_mb": ni.memory_used_mb,
                    "memory_total_mb": ni.memory_total_mb,
                    "requests_served": ni.requests_served,
                    "credits_earned": ni.credits_earned,
                    "last_heartbeat_ms": ni.last_heartbeat_ms,
                }
    except Exception as exc:
        logger.warning("Failed to get node details for %s: %s", node_id, exc)
        return {
            "node_id": node_id,
            "status": "unknown",
            "error": str(exc),
        }

    return {
        "node_id": node_id,
        "status": "not_found",
    }

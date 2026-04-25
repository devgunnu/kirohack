"""Dashboard event streaming - connects to real Coordinator events."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

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


async def stream_events() -> AsyncGenerator[dict, None]:
    """Stream real-time events from the Coordinator.

    Yields event dictionaries with type and payload. Until a dedicated
    SubscribeEvents RPC exists on the Coordinator we poll
    GetNetworkStatus periodically and emit derived events.
    """
    addr = _coordinator_address()
    channel = grpc.insecure_channel(addr)
    stub = coordinator_pb2_grpc.CoordinatorServiceStub(channel)

    try:
        while True:
            try:
                request = coordinator_pb2.GetNetworkStatusRequest()
                # gRPC sync stub call run in thread to avoid blocking the loop.
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: stub.GetNetworkStatus(request, timeout=5.0)
                )

                yield {
                    "type": "stats_update",
                    "active_nodes": response.active_nodes,
                    "total_layers": response.total_layers,
                    "covered_layers": response.covered_layers,
                    "queue_depth": response.queue_depth,
                    "model": response.model_id,
                    "timestamp": loop.time(),
                }

                for ni in response.nodes:
                    yield {
                        "type": "node_status",
                        "node_id": ni.node_id,
                        "address": ni.address,
                        "status": ni.status,
                        "layers": f"{ni.layer_start}-{ni.layer_end}",
                        "timestamp": loop.time(),
                    }

                await asyncio.sleep(5.0)
            except Exception as exc:
                logger.warning("Event stream error: %s", exc)
                yield {
                    "type": "error",
                    "payload": {"message": str(exc)},
                }
                await asyncio.sleep(2.0)
    finally:
        channel.close()


async def get_live_nodes() -> list[dict]:
    """Fetch current node list from the Coordinator."""
    addr = _coordinator_address()
    channel = grpc.insecure_channel(addr)
    stub = coordinator_pb2_grpc.CoordinatorServiceStub(channel)

    try:
        loop = asyncio.get_event_loop()
        request = coordinator_pb2.GetNetworkStatusRequest()
        response = await loop.run_in_executor(
            None, lambda: stub.GetNetworkStatus(request, timeout=5.0)
        )
        return [
            {
                "node_id": ni.node_id,
                "address": ni.address,
                "layers": f"{ni.layer_start}-{ni.layer_end}",
                "status": ni.status,
                "credits": ni.credits_earned,
            }
            for ni in response.nodes
        ]
    except Exception as exc:
        logger.warning("Failed to fetch live nodes: %s", exc)
        return []
    finally:
        channel.close()


# Backward-compatible alias for any caller still importing the old name.
mock_event_stream = stream_events

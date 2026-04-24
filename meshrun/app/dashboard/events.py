"""
Mock event stream for the MeshRun dashboard.

This module provides a mock event generator that simulates real coordinator events.
The dashboard server imports from here and streams events to connected clients.
"""

import asyncio
import random
import time
from typing import AsyncGenerator

# TODO: replace with real coordinator websocket subscription

NODE_IDS = ["node-a", "node-b", "node-c", "node-d"]


async def mock_event_stream() -> AsyncGenerator[dict, None]:
    """
    Yields mock events simulating real coordinator events.
    Each event has a type and payload.
    Event types: node_status, job_started, job_hop, job_complete, queue_update, stats_update
    """
    job_counter = 1
    while True:
        await asyncio.sleep(random.uniform(0.8, 2.0))

        event_type = random.choice([
            "job_started", "job_hop", "job_complete",
            "node_status", "queue_update", "stats_update"
        ])

        if event_type == "job_started":
            yield {
                "type": "job_started",
                "job_id": f"job-{job_counter:04d}",
                "prompt_preview": random.choice([
                    "Summarize this research paper...",
                    "Explain transformer architecture...",
                    "Write unit tests for...",
                    "Translate this document...",
                    "Debug this Python function..."
                ]),
                "route": random.sample(NODE_IDS[:3], 3),
                "timestamp": time.time()
            }
            job_counter += 1

        elif event_type == "job_hop":
            yield {
                "type": "job_hop",
                "job_id": f"job-{max(1, job_counter-1):04d}",
                "from_node": random.choice(NODE_IDS[:2]),
                "to_node": random.choice(NODE_IDS[1:3]),
                "latency_ms": round(random.uniform(30, 60), 1),
                "timestamp": time.time()
            }

        elif event_type == "job_complete":
            yield {
                "type": "job_complete",
                "job_id": f"job-{max(1, job_counter-1):04d}",
                "tokens": random.randint(20, 80),
                "total_latency": round(random.uniform(0.8, 2.5), 2),
                "cost_saved_usd": round(random.uniform(0.001, 0.005), 4),
                "co2_avoided_g": round(random.uniform(0.0005, 0.002), 4),
                "timestamp": time.time()
            }

        elif event_type == "node_status":
            yield {
                "type": "node_status",
                "node_id": random.choice(NODE_IDS),
                "status": random.choice(["active", "active", "active", "idle"]),
                "latency_ms": round(random.uniform(30, 55), 1),
                "timestamp": time.time()
            }

        elif event_type == "queue_update":
            yield {
                "type": "queue_update",
                "depth": random.randint(0, 5),
                "timestamp": time.time()
            }

        elif event_type == "stats_update":
            yield {
                "type": "stats_update",
                "total_tokens": random.randint(10000, 50000),
                "total_cost_saved_usd": round(random.uniform(5.0, 25.0), 2),
                "total_co2_avoided_g": round(random.uniform(2.0, 10.0), 4),
                "timestamp": time.time()
            }

"""
Inference client stub for MeshRun.

This is the only file that talks to the backend. All other modules import
from here. Every function returns hardcoded mock data until the coordinator
integration is complete.
"""


def get_routing_path(prompt: str) -> dict:
    """Return routing info for a given prompt."""
    # TODO: replace with real coordinator call
    return {
        "job_id": "job-8a3f2b",
        "route": ["node-a", "node-b", "node-c"],
        "node_addresses": {
            "node-a": "192.168.1.10:5001",
            "node-b": "192.168.1.11:5001",
            "node-c": "192.168.1.12:5001",
        },
        "estimated_wait": 0.4,
    }


def submit_inference_job(prompt: str, priority: str = "normal") -> dict:
    """Submit a synchronous inference job and return the result."""
    # TODO: replace with real coordinator call
    return {
        "job_id": "job-8a3f2b",
        "status": "completed",
        "output": "This is a mock inference result for the prompt: " + prompt[:40],
        "tokens_generated": 42,
        "total_latency": 1.24,
        "hop_latencies": {"node-a": 0.41, "node-b": 0.38, "node-c": 0.45},
        "cost_saved_usd": 0.0031,
        "co2_avoided_g": 0.0012,
    }


def submit_async_job(prompt: str, priority: str = "normal") -> dict:
    """Submit an async inference job and return queue info."""
    # TODO: replace with real coordinator call
    return {
        "job_id": "job-9c1d4e",
        "status": "queued",
        "queue_position": 2,
        "estimated_wait": 8.3,
    }


def get_job_result(job_id: str) -> dict:
    """Retrieve the result of a previously submitted async job."""
    # TODO: replace with real coordinator call
    return {
        "job_id": job_id,
        "status": "completed",
        "output": "Mock result for job " + job_id,
        "tokens_generated": 38,
        "total_latency": 2.1,
        "hop_latencies": {"node-a": 0.7, "node-b": 0.8, "node-c": 0.6},
        "cost_saved_usd": 0.0028,
        "co2_avoided_g": 0.0010,
    }


def register_node(node_id: str, layers: str, vram_gb: float) -> dict:
    """Register a machine as a worker node."""
    # TODO: replace with real coordinator call
    return {
        "success": True,
        "node_id": node_id,
        "layers_assigned": layers,
        "message": "Node registered successfully",
    }


def deregister_node(node_id: str) -> dict:
    """Gracefully deregister a worker node."""
    # TODO: replace with real coordinator call
    return {"success": True, "message": "Node gracefully deregistered"}


def get_network_status() -> dict:
    """Fetch full network status including nodes and queue."""
    # TODO: replace with real coordinator call
    return {
        "active_nodes": 4,
        "total_layers": 28,
        "covered_layers": 28,
        "model": "qwen2.5-3b (int8)",
        "queue_depth": 2,
        "nodes": [
            {"id": "node-a", "address": "192.168.1.10:5001", "layers": "0-6",   "status": "active",      "credits": 42.1, "latency_ms": 38},
            {"id": "node-b", "address": "192.168.1.11:5001", "layers": "7-13",  "status": "active",      "credits": 38.7, "latency_ms": 41},
            {"id": "node-c", "address": "192.168.1.12:5001", "layers": "14-20", "status": "idle",        "credits": 21.3, "latency_ms": 44},
            {"id": "node-d", "address": "192.168.1.13:5001", "layers": "21-27", "status": "unreachable", "credits": 9.8,  "latency_ms": None},
        ],
        "queue": [
            {"position": 1, "job_id": "job-8a3f", "preview": "Summarize this research paper...",       "priority_score": 94.2, "wait_time": 0.4},
            {"position": 2, "job_id": "job-2c1d", "preview": "Explain transformer architecture...",    "priority_score": 87.1, "wait_time": 1.2},
            {"position": 3, "job_id": "job-9e7b", "preview": "Write unit tests for this function...", "priority_score": 71.3, "wait_time": 2.8},
        ],
    }


def get_credits(node_id: str) -> dict:
    """Fetch credit balance and history for a node."""
    # TODO: replace with real coordinator call
    return {
        "node_id": node_id,
        "balance": 128.4,
        "compute_contributed_hours": 3.2,
        "priority_score": 94.2,
        "history": [
            {"time": "2m ago",  "event": "Forward pass (layers 0-6)", "credits": 2.4},
            {"time": "8m ago",  "event": "Forward pass (layers 0-6)", "credits": 2.1},
            {"time": "15m ago", "event": "Forward pass (layers 0-6)", "credits": 3.0},
            {"time": "1h ago",  "event": "Joined network",            "credits": 10.0},
        ],
    }


def detect_local_hardware() -> dict:
    """Detect local GPU and RAM for node registration."""
    # TODO: replace with real hardware detection using torch or psutil
    return {
        "vram_gb": 8.0,
        "ram_gb": 16.0,
        "gpu_name": "NVIDIA RTX 4060",
        "suggested_layers": "0-6",
        "can_run_model": True,
    }

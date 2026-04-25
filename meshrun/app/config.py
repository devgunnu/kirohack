"""Global configuration for the MeshRun app."""

import uuid

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """MeshRun application settings, loaded from environment or .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    coordinator_url: str = "http://localhost:8000"
    coordinator_ws: str = "ws://localhost:8000/ws"
    dashboard_port: int = 7654
    dashboard_host: str = "127.0.0.1"
    default_model: str = "qwen2.5-3b"
    node_id: str = str(uuid.uuid4())
    credits_alpha: float = 0.7
    credits_beta: float = 0.3


settings = Config()


def get_priority_score(compute_contributed: float, wait_time: float) -> float:
    """Compute priority score as a weighted sum of compute and wait time."""
    return settings.credits_alpha * compute_contributed + settings.credits_beta * wait_time

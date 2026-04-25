import toml
import os
from pathlib import Path

STATE_DIR = Path.home() / ".meshrun"
STATE_FILE = STATE_DIR / "config.toml"

DEFAULT_STATE = {
    "joined": False,
    "node_id": "",
    "model": "",
    "compute_allocation": 0,
    "compute_gb": 0.0,
    "layers_assigned": "",
    "credits_balance": 0.0,
    "earning_rate": 0.0,
    "worker_pid": 0,
}

def get_state() -> dict:
    if not STATE_FILE.exists():
        return DEFAULT_STATE.copy()
    with open(STATE_FILE, "r") as f:
        return toml.load(f)

def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        toml.dump(state, f)

def is_joined() -> bool:
    return get_state().get("joined", False)

def require_joined() -> None:
    """Call at the top of every command. Exits with onboarding prompt if not joined."""
    if not is_joined():
        from meshrun.app.display.spinners import console
        from rich.panel import Panel
        console.print()
        console.print(Panel(
            "[bold cyan]Welcome to MeshRun![/bold cyan]\n\n"
            "You haven't joined the mesh yet.\n"
            "Run [bold]meshrun join[/bold] to get started.\n\n"
            "[dim]MeshRun lets you run AI inference on compute\n"
            "you already own — no cloud costs, no per-token fees.[/dim]",
            title="[yellow]⚡ First Time Setup Required[/yellow]",
            border_style="yellow"
        ))
        console.print()
        raise SystemExit(0)

def clear_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()

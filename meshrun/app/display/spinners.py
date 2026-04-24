"""Reusable Rich spinner/status helpers for MeshRun CLI commands."""

from rich.console import Console
from rich.status import Status

console = Console()


def spinner_routing() -> Status:
    """Spinner shown while routing a request through the mesh."""
    return console.status("[cyan]Routing request through mesh...", spinner="dots")


def spinner_connecting() -> Status:
    """Spinner shown while connecting to the coordinator."""
    return console.status("[cyan]Connecting to coordinator...", spinner="dots")


def spinner_joining() -> Status:
    """Spinner shown while registering a node with the coordinator."""
    return console.status("[green]Registering node with coordinator...", spinner="dots")


def spinner_loading() -> Status:
    """Generic loading spinner."""
    return console.status("[cyan]Loading...", spinner="dots")


def print_success(msg: str) -> None:
    """Print a success message with a green checkmark."""
    console.print(f"[bold green]✓[/bold green] {msg}")


def print_error(msg: str) -> None:
    """Print an error message with a red cross."""
    console.print(f"[bold red]✗[/bold red] {msg}")


def print_info(msg: str) -> None:
    """Print an informational message with a dim arrow."""
    console.print(f"[dim]→[/dim] {msg}")

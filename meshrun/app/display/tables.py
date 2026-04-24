"""Reusable Rich Table components for MeshRun CLI commands."""

from rich.table import Table

from app.display.spinners import console


def show_nodes_table() -> None:
    """Render and print the MeshRun nodes table."""
    table = Table(title="MeshRun Nodes")

    table.add_column("Node ID", style="cyan")
    table.add_column("Address", style="dim")
    table.add_column("Layers", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Credits Earned", justify="right")
    table.add_column("Latency", justify="right")

    # TODO: replace with real data
    table.add_row("node-a", "192.168.1.10:5001", "0–6", "[green]Active[/green]", "42.1", "38ms")
    table.add_row("node-b", "192.168.1.11:5001", "7–13", "[green]Active[/green]", "38.7", "41ms")
    table.add_row("node-c", "192.168.1.12:5001", "14–20", "[yellow]Idle[/yellow]", "21.3", "44ms")
    table.add_row("node-d", "192.168.1.13:5001", "21–27", "[red]Unreachable[/red]", "9.8", "--")

    console.print(table)


def show_status_table() -> None:
    """Render and print the queue status table."""
    table = Table(title="Queue Status")

    table.add_column("Position", justify="center")
    table.add_column("Job ID", style="cyan")
    table.add_column("Prompt Preview", style="dim")
    table.add_column("Priority Score", justify="right")
    table.add_column("Wait Time", justify="right")

    # TODO: replace with real data
    table.add_row("1", "job-8a3f", "Summarize this research paper...", "94.2", "0.4s")
    table.add_row("2", "job-2c1d", "Explain transformer architecture...", "87.1", "1.2s")
    table.add_row("3", "job-9e7b", "Write unit tests for this function...", "71.3", "2.8s")

    console.print(table)


def show_credits_history() -> None:
    """Render and print the credit history table."""
    table = Table(title="Credit History")

    table.add_column("Time", style="dim")
    table.add_column("Event")
    table.add_column("Credits", justify="right", style="green")

    # TODO: replace with real data
    table.add_row("2m ago", "Forward pass (layers 0–6)", "+2.4")
    table.add_row("8m ago", "Forward pass (layers 0–6)", "+2.1")
    table.add_row("15m ago", "Forward pass (layers 0–6)", "+3.0")
    table.add_row("1h ago", "Joined network", "+10.0")

    console.print(table)

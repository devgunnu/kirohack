"""Reusable Rich Table components for MeshRun CLI commands."""

from typing import Any

from rich.table import Table

from meshrun.app.display.spinners import console


def show_nodes_table(nodes: list[dict[str, Any]] | None = None) -> None:
    """Render and print the MeshRun nodes table."""
    table = Table(title="MeshRun Nodes")

    table.add_column("Node ID", style="cyan")
    table.add_column("Address", style="dim")
    table.add_column("Layers", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Credits Earned", justify="right")
    table.add_column("Latency", justify="right")

    if nodes:
        for node in nodes:
            status = node.get("status", "unknown")
            status_display = {
                "active": "[green]Active[/green]",
                "idle": "[yellow]Idle[/yellow]",
                "unreachable": "[red]Unreachable[/red]",
                "unhealthy": "[red]Unhealthy[/red]",
                "unknown": "[dim]Unknown[/dim]",
            }.get(status, f"[dim]{status}[/dim]")

            credits = node.get("credits", 0.0)
            try:
                credits_str = f"{float(credits):.1f}"
            except (TypeError, ValueError):
                credits_str = str(credits)

            table.add_row(
                node.get("node_id", "N/A"),
                node.get("address", "N/A"),
                node.get("layers", "N/A"),
                status_display,
                credits_str,
                str(node.get("latency", "--")),
            )
    else:
        table.add_row("[dim]No nodes registered[/dim]", "", "", "", "", "")

    console.print(table)


def show_status_table(jobs: list[dict[str, Any]] | None = None) -> None:
    """Render and print the queue status table."""
    table = Table(title="Queue Status")

    table.add_column("Position", justify="center")
    table.add_column("Job ID", style="cyan")
    table.add_column("Prompt Preview", style="dim")
    table.add_column("Priority Score", justify="right")
    table.add_column("Wait Time", justify="right")

    if jobs:
        for i, job in enumerate(jobs, 1):
            preview = str(job.get("prompt", ""))[:40]
            try:
                priority = f"{float(job.get('priority', 0.0)):.1f}"
            except (TypeError, ValueError):
                priority = str(job.get("priority", "0.0"))
            table.add_row(
                str(job.get("position", i)),
                job.get("job_id", "N/A"),
                f"{preview}...",
                priority,
                str(job.get("wait_time", "0.0s")),
            )
    else:
        table.add_row("[dim]Queue empty[/dim]", "", "", "", "")

    console.print(table)


def show_credits_history(history: list[dict[str, Any]] | None = None) -> None:
    """Render and print the credit history table."""
    table = Table(title="Credit History")

    table.add_column("Time", style="dim")
    table.add_column("Event")
    table.add_column("Credits", justify="right", style="green")

    if history:
        for entry in history:
            credits = entry.get("credits", 0.0)
            try:
                credits_val = float(credits)
                credits_display = f"+{credits_val:.1f}" if credits_val >= 0 else f"{credits_val:.1f}"
            except (TypeError, ValueError):
                credits_display = str(credits)
            table.add_row(
                entry.get("time", "N/A"),
                entry.get("event", "N/A"),
                credits_display,
            )
    else:
        table.add_row("[dim]No credit history[/dim]", "", "")

    console.print(table)

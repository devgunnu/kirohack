"""Show network and queue status."""

import typer
from rich.panel import Panel

from meshrun.app.client.inference import get_network_status
from meshrun.app.daemon import LOG_FILE, read_pid
from meshrun.app.display.panels import show_network_summary
from meshrun.app.display.spinners import console, print_success, spinner_connecting
from meshrun.app.display.tables import show_status_table
from meshrun.app.state import get_state

app = typer.Typer(help="Show network and queue status.")


@app.callback(invoke_without_command=True)
def status():
    from meshrun.app.state import require_joined
    require_joined()

    state = get_state()
    pid = read_pid()
    if pid is not None:
        worker_line = f"[green]● running[/green] (PID [cyan]{pid}[/cyan])"
    else:
        worker_line = "[red]● not running[/red] — run [bold]meshrun join[/bold] to restart"

    console.print()
    console.print(Panel(
        f"[bold]Node ID:[/bold]     [cyan]{state.get('node_id', '?')}[/cyan]\n"
        f"[bold]Worker:[/bold]      {worker_line}\n"
        f"[bold]Model:[/bold]       [cyan]{state.get('model', '?')}[/cyan]\n"
        f"[bold]Compute:[/bold]     [cyan]{state.get('compute_allocation', 0)}% "
        f"({state.get('compute_gb', 0)} GB)[/cyan]\n"
        f"[bold]Logs:[/bold]        [dim]{LOG_FILE}[/dim]",
        title="[bold cyan]Your Node[/bold cyan]",
        border_style="cyan",
    ))

    with spinner_connecting():
        data = get_network_status()
    print_success("Connected to coordinator.")
    show_network_summary(
        active_nodes=data.get("active_nodes", 0),
        layers_covered=f"{data.get('covered_layers', 0)} / {data.get('total_layers', 0)}",
        model_loaded=data.get("model", "N/A"),
        queue_depth=f"{data.get('queue_depth', 0)} jobs",
    )
    show_status_table(data.get("queue") or None)

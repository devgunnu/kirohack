"""Reusable Rich Panel components for MeshRun CLI commands."""

from rich.panel import Panel

from meshrun.app.display.spinners import console


def show_submit_result(
    job_id: str,
    prompt_preview: str,
    tokens: int = 0,
    latency: str = "N/A",
    hops: list[str] | None = None,
    cost_saved: str = "N/A",
    co2_avoided: str = "N/A",
) -> None:
    """Display inference result panel after a successful submission."""
    preview = prompt_preview[:60]
    hops = hops or []

    body = (
        f"[bold]Job ID:[/bold]          {job_id}\n"
        f"[bold]Prompt:[/bold]          {preview}\n"
        f"[bold]Tokens:[/bold]          {tokens}\n"
        f"[bold]Total latency:[/bold]   {latency}\n"
        f"[bold]Hop latency:[/bold]     {', '.join(hops)}\n"
        f"[bold]Cost saved:[/bold]      {cost_saved}\n"
        f"[bold]CO2 avoided:[/bold]     {co2_avoided}"
    )

    console.print(Panel(body, title="[bold green]Inference Complete[/bold green]", border_style="green"))


def show_credits_panel(
    node_id: str,
    balance: float = 0.0,
    compute: str = "0.0 GPU-hours",
    priority: float = 0.0,
) -> None:
    """Display the credits summary panel for a node."""
    body = (
        f"[bold]Node ID:[/bold]              {node_id}\n"
        f"[bold]Credit balance:[/bold]        {balance}\n"
        f"[bold]Compute contributed:[/bold]   {compute}\n"
        f"[bold]Priority score:[/bold]        {priority} (0.7 × compute + 0.3 × wait_time)"
    )

    console.print(Panel(body, title="[bold yellow]Your Credits[/bold yellow]", border_style="yellow"))


def show_join_success(node_id: str, layers: str) -> None:
    """Display the node registration success panel."""
    body = (
        f"[bold]Node ID:[/bold]        {node_id}\n"
        f"[bold]Layers hosted:[/bold]   {layers}\n"
        f"[bold]Status:[/bold]          Active\n"
        f"[bold]Message:[/bold]         You are now earning credits for every forward pass."
    )

    console.print(Panel(body, title="[bold green]Node Registered[/bold green]", border_style="green"))


def show_network_summary(
    active_nodes: int = 0,
    layers_covered: str = "0 / 0",
    model_loaded: str = "N/A",
    queue_depth: str = "0 jobs",
) -> None:
    """Display the network summary panel."""
    body = (
        f"[bold]Active nodes:[/bold]         {active_nodes}\n"
        f"[bold]Total layers covered:[/bold] {layers_covered}\n"
        f"[bold]Model loaded:[/bold]         {model_loaded}\n"
        f"[bold]Queue depth:[/bold]          {queue_depth}"
    )

    console.print(Panel(body, title="[bold blue]Network Summary[/bold blue]", border_style="blue"))

"""Reusable Rich Panel components for MeshRun CLI commands."""

from rich.panel import Panel

from app.display.spinners import console


def show_submit_result(job_id: str, prompt_preview: str) -> None:
    """Display inference result panel after a successful submission."""
    preview = prompt_preview[:60]
    tokens = 42  # TODO: replace with real data
    latency = "1.24s"  # TODO: replace with real data
    hops = ["Node A: 0.41s", "Node B: 0.38s", "Node C: 0.45s"]  # TODO: replace with real data
    cost_saved = "$0.0031"  # TODO: replace with real data
    co2_avoided = "0.0012g"  # TODO: replace with real data

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


def show_credits_panel(node_id: str) -> None:
    """Display the credits summary panel for a node."""
    balance = 128.4  # TODO: replace with real data
    compute = "3.2 GPU-hours"  # TODO: replace with real data
    priority = 94.2  # TODO: replace with real data

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


def show_network_summary() -> None:
    """Display the network summary panel."""
    active_nodes = 4  # TODO: replace with real data
    layers_covered = "28 / 28"  # TODO: replace with real data
    model_loaded = "qwen2.5-3b (int8)"  # TODO: replace with real data
    queue_depth = "2 jobs"  # TODO: replace with real data

    body = (
        f"[bold]Active nodes:[/bold]         {active_nodes}\n"
        f"[bold]Total layers covered:[/bold] {layers_covered}\n"
        f"[bold]Model loaded:[/bold]         {model_loaded}\n"
        f"[bold]Queue depth:[/bold]          {queue_depth}"
    )

    console.print(Panel(body, title="[bold blue]Network Summary[/bold blue]", border_style="blue"))

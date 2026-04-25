"""Submit an inference job to the mesh."""

import time
from typing import Optional

import typer
from rich.live import Live
from rich.text import Text

from meshrun.app.client.inference import get_job_result, submit_async_job, submit_inference_job
from meshrun.app.display.panels import show_submit_result
from meshrun.app.display.spinners import console, print_error, print_info, print_success, spinner_routing

app = typer.Typer(help="Submit an inference job to the mesh.")


@app.callback(invoke_without_command=True)
def submit(
    prompt: str = typer.Argument(..., help="The prompt to run inference on."),
    model: str = typer.Option("qwen2.5-3b", "--model", "-m", help="Model to use for inference."),
    priority: str = typer.Option("normal", "--priority", "-p", help="Job priority: high, normal, or low."),
    async_mode: bool = typer.Option(False, "--async", "-a", help="Submit job async and return a job ID immediately."),
    job_id: Optional[str] = typer.Option(None, "--job-id", "-j", help="Retrieve result of a previously submitted async job."),
):
    from meshrun.app.state import require_joined
    require_joined()

    # Retrieve existing async job
    if job_id:
        with spinner_routing():
            result = get_job_result(job_id)
        if result["status"] == "completed":
            print_success(f"Job [cyan]{job_id}[/cyan] complete.")
            show_submit_result(result["job_id"], result.get("output", "")[:60])
        else:
            print_info(f"Job [cyan]{job_id}[/cyan] status: {result['status']}")
        return

    if not prompt:
        print_error("Please provide a prompt.")
        raise typer.Exit(1)

    # Async submission
    if async_mode:
        with spinner_routing():
            result = submit_async_job(prompt, priority)
        print_success(f"Job queued. ID: [bold cyan]{result['job_id']}[/bold cyan]")
        print_info(f"Queue position: {result['queue_position']} — estimated wait: {result['estimated_wait']}s")
        print_info(f"Retrieve result with: [bold]meshrun submit --job-id {result['job_id']}[/bold]")
        return

    # Synchronous submission
    with spinner_routing():
        result = submit_inference_job(prompt, priority)

    # ── Hop visualization ────────────────────────────────────────
    console.print()
    console.rule("[dim]Inference Pipeline[/dim]")
    console.print()

    hops = result["hop_latencies"]
    nodes = list(hops.keys())
    latencies = list(hops.values())

    # Animate hops one by one
    for i, (node, latency) in enumerate(zip(nodes, latencies)):
        console.print(f"  [dim]→[/dim] [cyan]{node}[/cyan] [dim](layers {['0-6','7-13','14-20'][i] if i < 3 else ''})[/dim]  [green]{int(latency*1000)}ms[/green]")
        time.sleep(latency * 0.8)

    console.print()
    console.rule("[dim]Output[/dim]")
    console.print()

    # Token streaming
    words = result["output"].split()
    displayed = Text()
    with Live(displayed, console=console, refresh_per_second=12) as live:
        for word in words:
            displayed.append(word + " ")
            live.update(displayed)
            time.sleep(0.05)

    console.print()
    console.rule("[dim]Summary[/dim]")
    console.print()
    print_success("Inference complete.")
    show_submit_result(result["job_id"], prompt[:60])

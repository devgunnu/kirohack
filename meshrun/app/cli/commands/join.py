"""Register this machine as a worker node."""

import uuid

import typer

from app.client.inference import detect_local_hardware, register_node
from app.display.panels import show_join_success
from app.display.spinners import console, print_error, print_info, print_success, spinner_joining

app = typer.Typer(help="Register this machine as a worker node.")


@app.callback(invoke_without_command=True)
def join():
    console.print()
    console.rule("[bold cyan]MeshRun Node Setup[/bold cyan]")
    console.print()

    # Detect hardware
    with spinner_joining():
        hardware = detect_local_hardware()

    print_info(f"Detected GPU: [bold]{hardware['gpu_name']}[/bold]")
    print_info(f"Available VRAM: [bold]{hardware['vram_gb']} GB[/bold]")
    print_info(f"Suggested layer range: [bold]{hardware['suggested_layers']}[/bold]")
    console.print()

    if not hardware["can_run_model"]:
        print_error("Insufficient hardware to run the model. Minimum 6GB VRAM required.")
        raise typer.Exit(1)

    # Confirm
    confirmed = typer.confirm("Register this machine as a worker node?")
    if not confirmed:
        print_info("Cancelled.")
        raise typer.Exit(0)

    node_id = f"node-{uuid.uuid4().hex[:6]}"

    with spinner_joining():
        result = register_node(node_id, hardware["suggested_layers"], hardware["vram_gb"])

    if result["success"]:
        print_success(f"Registered as [cyan]{node_id}[/cyan]")
        show_join_success(node_id, hardware["suggested_layers"])
    else:
        print_error("Registration failed. Check coordinator connection.")
        raise typer.Exit(1)

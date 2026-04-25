"""Start the MeshRun Coordinator server."""

import typer
from rich import print as rprint

app = typer.Typer(help="Start the MeshRun Coordinator server.")


@app.callback(invoke_without_command=True)
def coordinator(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address for gRPC server."),
    port: int = typer.Option(50051, "--port", "-p", help="Port for gRPC server."),
    model_id: str = typer.Option("llama-3b", "--model", "-m", help="Default model ID for this cluster."),
    total_layers: int = typer.Option(28, "--layers", "-l", help="Total transformer layers in the model."),
    dtype: str = typer.Option("int8", "--dtype", "-d", help="Weight data type: fp16 or int8."),
    model_url: str = typer.Option("", "--model-url", "-u", help="HTTP URL to the safetensors model file."),
):
    """Start the Coordinator gRPC server."""
    import signal
    import sys

    from meshrun.coordinator.server import CoordinatorServer

    rprint(f"[bold cyan]Starting MeshRun Coordinator[/bold cyan]")
    rprint(f"  gRPC: {host}:{port}")
    rprint(f"  Model: {model_id} ({total_layers} layers, {dtype})")
    rprint(f"  Model URL: {model_url or '(not specified)'}")

    server = CoordinatorServer(host=host, port=port)

    server.start()

    # Store model config for auto-assignment via the public setter
    # (set_model_config is being added on CoordinatorServer by another agent).
    server.set_model_config(
        model_id=model_id,
        total_layers=total_layers,
        dtype=dtype,
        model_url=model_url,
    )

    rprint("[green]Coordinator started. Press Ctrl+C to stop.[/green]")

    def handle_sigterm(signum, frame):
        rprint("\n[yellow]Shutting down...[/yellow]")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)

    # Block forever
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()

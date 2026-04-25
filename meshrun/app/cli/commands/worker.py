"""Start a MeshRun Worker Node."""

import typer
from rich import print as rprint

app = typer.Typer(help="Start a MeshRun Worker Node.")


@app.callback(invoke_without_command=True)
def worker(
    coordinator: str = typer.Option("localhost:50051", "--coordinator", "-c", help="Coordinator gRPC address (host:port)."),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address for TCP data plane."),
    data_port: int = typer.Option(9100, "--data-port", "-d", help="TCP port for data plane."),
    grpc_port: int = typer.Option(50052, "--grpc-port", "-g", help="TCP port for gRPC control plane callbacks."),
    advertise: str = typer.Option("", "--advertise", "-a", help="Advertised address (host:port) for other nodes. Auto-detected if not specified."),
    gpu_memory_limit_mb: int = typer.Option(4096, "--gpu-limit", help="Maximum GPU memory to use (MB)."),
    device: int = typer.Option(0, "--device", help="CUDA device index."),
):
    """Start a MeshRun Worker Node."""
    import signal
    import sys

    from meshrun.worker.node import NodeConfig, WorkerNode
    from meshrun.worker.assignment_server import WorkerAssignmentServer

    # Auto-detect advertise address if not specified
    if advertise:
        advertise_host, advertise_port_str = advertise.rsplit(":", 1)
        advertise_port = int(advertise_port_str)
    else:
        # Try to detect Tailscale IP
        advertise_host = _detect_tailscale_ip()
        if not advertise_host:
            # Fall back to hostname
            import socket
            advertise_host = socket.gethostbyname(socket.gethostname())
        advertise_port = data_port

    rprint(f"[bold cyan]Starting MeshRun Worker Node[/bold cyan]")
    rprint(f"  Coordinator: {coordinator}")
    rprint(f"  Data plane: {host}:{data_port}")
    rprint(f"  gRPC callbacks: {host}:{grpc_port}")
    rprint(f"  Advertised address: {advertise_host}:{advertise_port}")
    rprint(f"  GPU limit: {gpu_memory_limit_mb} MB, device {device}")

    config = NodeConfig(
        host=host,
        data_port=data_port,
        grpc_port=grpc_port,
        coordinator_address=coordinator,
        gpu_memory_limit_mb=gpu_memory_limit_mb,
        device_index=device,
    )

    node = WorkerNode(config=config)

    # Override the address property to return the advertised address
    node._advertise_host = advertise_host
    node._advertise_port = advertise_port

    # Monkey-patch the address property
    original_address = type(node).address.fget

    def patched_address(self):
        return f"{self._advertise_host}:{self._advertise_port}"

    type(node).address = property(patched_address)

    # Phase 1: Startup (detect GPU, create ResourceMonitor) — transitions to REGISTERING
    capacity = node.startup()
    rprint(f"[green]GPU detected: {capacity.gpu_memory_total_mb} MB total, {capacity.gpu_memory_free_mb} MB free[/green]")

    # Phase 2: Start the worker gRPC server BEFORE registering so the coordinator can
    # immediately push AcceptLayerAssignment to us after registration completes.
    assignment_server = WorkerAssignmentServer(
        worker_node=node,
        host=host,
        port=grpc_port,
    )
    assignment_server.start()
    rprint(f"[green]Worker gRPC server listening on {host}:{grpc_port}[/green]")

    # Phase 3: Register with Coordinator
    response = node.register_with_coordinator()
    if response.status.name != "OK":
        rprint(f"[red]Registration failed: {response.message}[/red]")
        assignment_server.stop()
        raise typer.Exit(1)
    rprint(f"[green]Registered with coordinator[/green]")

    # Phase 4: Wait for layer assignment (blocking)
    rprint("[yellow]Waiting for layer assignment from coordinator...[/yellow]")

    rprint("[green]Worker started. Press Ctrl+C to stop.[/green]")

    def handle_sigterm(signum, frame):
        rprint("\n[yellow]Shutting down worker...[/yellow]")
        try:
            assignment_server.stop()
        except Exception:
            pass
        try:
            node.stop_heartbeat()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)

    # Block forever (the serving loop runs in background threads)
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        try:
            assignment_server.stop()
        except Exception:
            pass


def _detect_tailscale_ip() -> str | None:
    """Try to detect the Tailscale IP address of this machine.

    Uses the `tailscale ip --4` command if available.
    Returns None if Tailscale is not installed or not connected.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["tailscale", "ip", "--4"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode == 0:
            ip = result.stdout.strip()
            if ip.startswith("100."):
                return ip
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None

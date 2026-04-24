"""Launch the MeshRun web dashboard."""

import threading
import time
import webbrowser

import typer

from app.config import settings
from app.display.spinners import console, print_info, print_success

app = typer.Typer(help="Launch the MeshRun web dashboard.")


@app.callback(invoke_without_command=True)
def dashboard():
    from app.dashboard.server import start

    url = f"http://{settings.dashboard_host}:{settings.dashboard_port}"
    print_info(f"Starting dashboard at [bold cyan]{url}[/bold cyan]")

    # Start server in background thread
    thread = threading.Thread(
        target=start,
        kwargs={"host": settings.dashboard_host, "port": settings.dashboard_port},
        daemon=True
    )
    thread.start()
    time.sleep(1.2)

    webbrowser.open(url)
    print_success("Dashboard open. Press [bold]Ctrl+C[/bold] to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print_info("Dashboard stopped.")

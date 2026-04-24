"""Show network and queue status."""

import typer

from app.client.inference import get_network_status
from app.display.panels import show_network_summary
from app.display.spinners import print_success, spinner_connecting
from app.display.tables import show_status_table

app = typer.Typer(help="Show network and queue status.")


@app.callback(invoke_without_command=True)
def status():
    with spinner_connecting():
        data = get_network_status()
    print_success("Connected to coordinator.")
    show_network_summary()
    show_status_table()

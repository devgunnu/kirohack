"""List all nodes in the mesh network."""

import typer

from app.client.inference import get_network_status
from app.display.spinners import print_success, spinner_connecting
from app.display.tables import show_nodes_table

app = typer.Typer(help="List all nodes in the mesh network.")


@app.callback(invoke_without_command=True)
def nodes():
    with spinner_connecting():
        get_network_status()
    print_success("Fetched node list.")
    show_nodes_table()

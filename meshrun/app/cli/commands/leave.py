"""Deregister this machine from the mesh network."""

import typer

from app.client.inference import deregister_node
from app.display.spinners import print_error, print_info, print_success, spinner_loading

app = typer.Typer(help="Deregister this machine from the mesh network.")


@app.callback(invoke_without_command=True)
def leave():
    confirmed = typer.confirm("Deregister this node from the mesh? In-flight requests will be drained first.")
    if not confirmed:
        print_info("Cancelled.")
        raise typer.Exit(0)

    with spinner_loading():
        result = deregister_node("local-node")

    if result["success"]:
        print_success("Node gracefully deregistered. You are no longer earning credits.")
    else:
        print_error("Failed to deregister. Check coordinator connection.")
        raise typer.Exit(1)

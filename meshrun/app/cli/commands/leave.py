"""Deregister this machine from the mesh network."""

import typer

from meshrun.app.client.inference import deregister_node
from meshrun.app.daemon import is_running, stop_worker
from meshrun.app.display.spinners import print_error, print_info, print_success, spinner_loading
from meshrun.app.state import clear_state, get_state

app = typer.Typer(help="Deregister this machine from the mesh network.")


@app.callback(invoke_without_command=True)
def leave():
    from meshrun.app.state import require_joined
    require_joined()

    confirmed = typer.confirm("Stop your worker and leave the mesh? In-flight requests will be drained first.")
    if not confirmed:
        print_info("Cancelled.")
        raise typer.Exit(0)

    state = get_state()
    node_id = state.get("node_id") or "local-node"

    with spinner_loading():
        stopped = stop_worker() if is_running() else False
        result = deregister_node(node_id)

    if stopped:
        print_success("Worker daemon stopped.")
    else:
        print_info("No local worker daemon was running.")

    if result["success"]:
        print_success("Node gracefully deregistered. You are no longer earning credits.")
    else:
        print_error(f"Deregister reported failure: {result.get('message', 'unknown error')}")

    clear_state()

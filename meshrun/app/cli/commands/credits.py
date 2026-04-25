"""Show your credit balance and history."""

import typer

from meshrun.app.client.inference import get_credits
from meshrun.app.config import settings
from meshrun.app.display.panels import show_credits_panel
from meshrun.app.display.spinners import print_success, spinner_loading
from meshrun.app.display.tables import show_credits_history

app = typer.Typer(help="Show your credit balance and history.")


@app.callback(invoke_without_command=True)
def credits():
    from meshrun.app.state import require_joined
    require_joined()

    with spinner_loading():
        data = get_credits("local-node")
    print_success("Credits loaded.")
    show_credits_panel("local-node")
    show_credits_history()

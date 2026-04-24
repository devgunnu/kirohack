"""Show your credit balance and history."""

import typer

from app.client.inference import get_credits
from app.config import settings
from app.display.panels import show_credits_panel
from app.display.spinners import print_success, spinner_loading
from app.display.tables import show_credits_history

app = typer.Typer(help="Show your credit balance and history.")


@app.callback(invoke_without_command=True)
def credits():
    with spinner_loading():
        data = get_credits("local-node")
    print_success("Credits loaded.")
    show_credits_panel("local-node")
    show_credits_history()

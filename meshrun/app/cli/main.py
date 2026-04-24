"""MeshRun CLI entry point."""

import typer
from rich import print as rprint

from app.cli.commands import submit, status, join, leave, nodes, credits, dashboard

app = typer.Typer(
    name="meshrun",
    help="[bold cyan]MeshRun[/bold cyan] — distributed AI inference on compute you already own.",
    add_completion=True,
    rich_markup_mode="rich",
)

app.add_typer(submit.app, name="submit")
app.add_typer(status.app, name="status")
app.add_typer(join.app, name="join")
app.add_typer(leave.app, name="leave")
app.add_typer(nodes.app, name="nodes")
app.add_typer(credits.app, name="credits")
app.add_typer(dashboard.app, name="dashboard")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        rprint("[bold cyan]MeshRun[/bold cyan] — run AI on your mesh, not the cloud.")
        rprint("Run [bold]meshrun --help[/bold] to see available commands.")


if __name__ == "__main__":
    app()

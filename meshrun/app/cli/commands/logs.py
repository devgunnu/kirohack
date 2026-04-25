"""Tail the worker daemon's log output."""

import time

import typer

from meshrun.app.daemon import LOG_FILE, is_running
from meshrun.app.display.spinners import console, print_error, print_info

app = typer.Typer(help="Tail the worker daemon's log output.")


@app.callback(invoke_without_command=True)
def logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow the log as new lines are appended."),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of trailing lines to show."),
):
    if not LOG_FILE.exists():
        print_error(f"No log file at {LOG_FILE}. Has the worker ever been started?")
        raise typer.Exit(1)

    with open(LOG_FILE, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        chunk = min(size, 64 * 1024)
        fh.seek(size - chunk)
        data = fh.read().decode("utf-8", errors="replace")
        tail = data.splitlines()[-lines:]
        for line in tail:
            console.print(line)

        if not follow:
            return

        if not is_running():
            print_info("Worker daemon is not running; nothing new to follow.")
            return

        console.print()
        print_info("Following log. Press Ctrl+C to stop.")
        try:
            while True:
                line = fh.readline()
                if not line:
                    time.sleep(0.3)
                    continue
                console.print(line.decode("utf-8", errors="replace").rstrip())
        except KeyboardInterrupt:
            pass

import typer
import uuid
import time
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.console import Console
from rich.columns import Columns
from rich.text import Text
from rich.rule import Rule
from app.display.spinners import console, print_success, print_error, print_info, spinner_joining
from app.client.inference import detect_local_hardware, register_node, get_earning_rate
from app.state import save_state, is_joined, get_state

app = typer.Typer(help="Register this machine as a worker node.")

MODELS = [
    {"name": "Qwen2.5-3B (int8)",  "size": "3.0 GB", "vram": 3.0, "speed": "★★★★★ Recommended", "key": "qwen2.5-3b"},
    {"name": "Llama 3.2-3B (int8)", "size": "3.2 GB", "vram": 3.2, "speed": "★★★★☆ Good performance", "key": "llama3.2-3b"},
    {"name": "Phi-3-mini (int8)",   "size": "2.4 GB", "vram": 2.4, "speed": "★★★☆☆ Fastest / low VRAM", "key": "phi-3-mini"},
]

COMPUTE_OPTIONS = [25, 50, 75, 100]

@app.callback(invoke_without_command=True)
def join():
    if is_joined():
        state = get_state()
        console.print()
        console.print(Panel(
            f"[bold green]Already joined![/bold green]\n\n"
            f"Node ID:   [cyan]{state['node_id']}[/cyan]\n"
            f"Model:     [cyan]{state['model']}[/cyan]\n"
            f"Layers:    [cyan]{state['layers_assigned']}[/cyan]\n"
            f"Compute:   [cyan]{state['compute_allocation']}% ({state['compute_gb']} GB)[/cyan]\n"
            f"Credits:   [cyan]{state['credits_balance']}[/cyan]\n"
            f"Earning:   [cyan]+{state['earning_rate']} credits / forward pass[/cyan]",
            title="[green]✓ MeshRun Active[/green]",
            border_style="green"
        ))
        console.print()
        return

    # ── Welcome banner ──────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Welcome to MeshRun[/bold cyan]")
    console.print()
    console.print("[dim]Run AI inference on compute you already own.[/dim]")
    console.print("[dim]Contribute your idle GPU, earn credits, get priority access.[/dim]")
    console.print()

    # ── STEP 1: Hardware detection ───────────────────────────────
    console.rule("[bold]Step 1 of 4 — Hardware Detection[/bold]")
    console.print()

    with spinner_joining():
        hardware = detect_local_hardware()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", width=24)
    table.add_column(style="cyan bold")
    table.add_row("GPU",              hardware["gpu_name"])
    table.add_row("VRAM Available",   f"{hardware['vram_gb']} GB")
    table.add_row("RAM",              f"{hardware['ram_gb']} GB")
    table.add_row("Suggested layers", hardware["suggested_layers"])
    table.add_row("Can run model",    "[green]✓ Yes[/green]" if hardware["can_run_model"] else "[red]✗ No[/red]")
    console.print(table)
    console.print()

    if not hardware["can_run_model"]:
        print_error("Insufficient hardware. Minimum 6GB VRAM required.")
        raise typer.Exit(1)

    input("\nPress Enter to continue...")

    # ── STEP 2: Compute allocation ───────────────────────────────
    console.print()
    console.rule("[bold]Step 2 of 4 — Compute Allocation[/bold]")
    console.print()
    console.print("How much VRAM do you want to contribute to the mesh?\n")

    compute_table = Table(show_header=True, header_style="bold cyan")
    compute_table.add_column("Option", justify="center", width=8)
    compute_table.add_column("Allocation", justify="center", width=12)
    compute_table.add_column("VRAM", justify="center", width=10)
    compute_table.add_column("Credits / Pass", justify="center", width=16)

    for i, pct in enumerate(COMPUTE_OPTIONS):
        gb = round(hardware["vram_gb"] * pct / 100, 1)
        rate = get_earning_rate(pct, gb)
        recommended = " [dim](recommended)[/dim]" if pct == 50 else ""
        compute_table.add_row(
            str(i + 1),
            f"{pct}%{recommended}",
            f"{gb} GB",
            f"[green]+{rate}[/green]"
        )

    console.print(compute_table)
    console.print()

    choice = Prompt.ask(
        "Select option",
        choices=["1", "2", "3", "4"],
        default="2"
    )
    compute_pct = COMPUTE_OPTIONS[int(choice) - 1]
    compute_gb = round(hardware["vram_gb"] * compute_pct / 100, 1)
    earning_rate = get_earning_rate(compute_pct, compute_gb)

    print_success(f"Compute allocated: [cyan]{compute_pct}%[/cyan] ({compute_gb} GB) → [green]+{earning_rate} credits / pass[/green]")
    input("\nPress Enter to continue...")

    # ── STEP 3: Model selection ──────────────────────────────────
    console.print()
    console.rule("[bold]Step 3 of 4 — Model Selection[/bold]")
    console.print()
    console.print("Choose which model your node will host layers for:\n")

    model_table = Table(show_header=True, header_style="bold cyan")
    model_table.add_column("Option", justify="center", width=8)
    model_table.add_column("Model", width=22)
    model_table.add_column("Size", justify="center", width=10)
    model_table.add_column("Rating", width=28)

    for i, m in enumerate(MODELS):
        model_table.add_row(
            str(i + 1),
            f"[cyan]{m['name']}[/cyan]" if i == 0 else m["name"],
            m["size"],
            m["speed"]
        )

    console.print(model_table)
    console.print()

    model_choice = Prompt.ask(
        "Select model",
        choices=["1", "2", "3"],
        default="1"
    )
    selected_model = MODELS[int(model_choice) - 1]
    print_success(f"Model selected: [cyan]{selected_model['name']}[/cyan]")
    input("\nPress Enter to continue...")

    # ── STEP 4: Confirm ──────────────────────────────────────────
    console.print()
    console.rule("[bold]Step 4 of 4 — Confirm & Register[/bold]")
    console.print()

    node_id = f"node-{uuid.uuid4().hex[:6]}"
    starting_credits = 10.0
    priority_start = round(0.7 * earning_rate + 0.3 * 0, 1)

    confirm_table = Table(show_header=False, box=None, padding=(0, 2))
    confirm_table.add_column(style="dim", width=24)
    confirm_table.add_column(style="cyan bold")
    confirm_table.add_row("Node ID",           node_id)
    confirm_table.add_row("Model",             selected_model["name"])
    confirm_table.add_row("Compute allocated", f"{compute_pct}% ({compute_gb} GB VRAM)")
    confirm_table.add_row("Layers hosting",    hardware["suggested_layers"])
    confirm_table.add_row("Starting credits",  f"{starting_credits}")
    confirm_table.add_row("Earning rate",      f"+{earning_rate} credits / forward pass")
    confirm_table.add_row("Priority score",    f"{priority_start} / 100")

    console.print(confirm_table)
    console.print()

    confirmed = typer.confirm("Register this node and join the mesh?")
    if not confirmed:
        print_info("Cancelled.")
        raise typer.Exit(0)

    # Register
    with spinner_joining():
        result = register_node(node_id, hardware["suggested_layers"], compute_gb)
        time.sleep(0.8)

    if result["success"]:
        # Save state
        save_state({
            "joined": True,
            "node_id": node_id,
            "model": selected_model["key"],
            "compute_allocation": compute_pct,
            "compute_gb": compute_gb,
            "layers_assigned": hardware["suggested_layers"],
            "credits_balance": starting_credits,
            "earning_rate": earning_rate,
        })

        console.print()
        console.print(Panel(
            f"[bold green]You're live on the mesh![/bold green]\n\n"
            f"  Node ID:  [cyan]{node_id}[/cyan]\n"
            f"  Model:    [cyan]{selected_model['name']}[/cyan]\n"
            f"  Layers:   [cyan]{hardware['suggested_layers']}[/cyan]\n"
            f"  Earning:  [green]+{earning_rate} credits per forward pass[/green]\n\n"
            f"[dim]Run [bold]meshrun status[/bold] to see the network.\n"
            f"Run [bold]meshrun submit \"your prompt\"[/bold] to run inference.[/dim]",
            title="[bold green]✓ Joined MeshRun[/bold green]",
            border_style="green"
        ))
        console.print()
    else:
        print_error("Registration failed. Check coordinator connection.")
        raise typer.Exit(1)

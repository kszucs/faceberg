"""Command-line interface for Faceberg."""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from faceberg.config import CatalogConfig

console = Console()





@click.group()
@click.version_option(version="0.1.0", prog_name="faceberg")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="faceberg.yml",
    help="Path to faceberg.yml config file (only needed for sync)",
)
@click.option(
    "--token",
    "-t",
    help="HuggingFace API token (can also set HF_TOKEN env variable)",
    type=str,
    envvar="HF_TOKEN",
    default=None,
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx, config, verbose, token):
    """Faceberg - Bridge HuggingFace datasets with Apache Iceberg.

    A command-line tool to expose HuggingFace datasets as Iceberg tables,
    enabling powerful analytics and time-travel capabilities.
    """
    config = CatalogConfig.from_yaml(config)

    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["catalog"] = config.to_catalog(token)
    ctx.obj["token"] = token
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("table_name", required=False)
@click.pass_context
def sync(ctx, table_name):
    """Sync Iceberg tables with HuggingFace datasets from config.

    Discovers datasets and creates/updates Iceberg tables. For new tables,
    creates initial metadata and namespaces on-demand. For existing tables,
    checks if dataset revision has changed and skips if already up-to-date.

    Example:
        faceberg --config=faceberg.yml sync
        faceberg --config=faceberg.yml sync namespace1.table1
    """
    catalog = ctx.obj["catalog"]

    console.print(f"[bold blue]Catalog:[/bold blue] {catalog.uri}")

    # Sync tables with progress
    if table_name:
        console.print(f"\n[bold blue]Syncing table:[/bold blue] {table_name}")
    else:
        console.print("\n[bold blue]Syncing tables...[/bold blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Discovering and syncing...", total=None)

        try:
            tables = catalog.sync(table_name=table_name)
            if tables:
                progress.update(task, description=f"✓ Synced {len(tables)} table(s)")
            else:
                progress.update(task, description="✓ All tables up-to-date")
            console.print("\n[bold green]Done![/bold green]")
        except Exception as e:
            progress.update(task, description=f"✗ Failed: {e}")
            raise


@main.command()
@click.pass_context
def init(ctx):
    """Initialize an empty catalog.

    For local catalogs, creates the directory and catalog.json file.
    For remote catalogs (HuggingFace), creates a new dataset repository.

    Use this command before syncing datasets to explicitly create the catalog.
    For RemoteCatalog, this is required to create the HF repository before
    you can sync datasets to it.

    Example:
        # Initialize local catalog
        faceberg --config=faceberg.yml init

        # Initialize remote catalog (creates HF dataset repo)
        export HF_TOKEN=your_token
        faceberg --config=faceberg.yml init
    """
    catalog = ctx.obj["catalog"]
    is_remote = catalog.uri.startswith("hf://")

    if is_remote:
        console.print(f"[bold blue]Initializing remote catalog:[/bold blue] {catalog.uri}")
    else:
        console.print(f"[bold blue]Initializing local catalog:[/bold blue] {catalog.uri}")

    try:
        catalog.init()
        console.print("[bold green]✓ Catalog initialized successfully![/bold green]")

        if is_remote:
            console.print("\n[dim]Next steps:[/dim]")
            console.print("  1. Ensure faceberg.yml config file is configured")
            console.print("  2. Run: faceberg --config=faceberg.yml sync")
        else:
            console.print("\n[dim]Next steps:[/dim]")
            console.print("  1. Ensure faceberg.yml config file is configured")
            console.print("  2. Run: faceberg --config=faceberg.yml sync")
    except ValueError as e:
        # Handle "repository already exists" error
        if "already exists" in str(e):
            console.print(f"[bold yellow]Warning:[/bold yellow] {e}")
            console.print("[dim]Catalog already initialized. Use 'sync' to add tables.[/dim]")
        else:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise click.Abort()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if ctx.obj.get("verbose"):
            import traceback

            console.print("\n[bold red]Traceback:[/bold red]")
            console.print(traceback.format_exc())
        raise click.Abort()


@main.command("list")
@click.pass_context
def list_tables(ctx):
    """List all tables in catalog.

    Example:
        faceberg --config=faceberg.yml list
    """
    catalog = ctx.obj["catalog"]

    console.print(f"[bold blue]Catalog:[/bold blue] {catalog.uri}\n")

    # List all namespaces and tables
    namespaces = catalog.list_namespaces()
    if not namespaces:
        console.print("[yellow]No tables found[/yellow]")
        return

    for ns in namespaces:
        ns_str = ".".join(ns) if isinstance(ns, tuple) else ns
        console.print(f"[bold cyan]{ns_str}[/bold cyan]")
        tables = catalog.list_tables(ns)
        for table in tables:
            table_str = ".".join(table) if isinstance(table, tuple) else table
            console.print(f"  • {table_str}")


@main.command()
@click.argument("table_name")
@click.pass_context
def info(ctx, table_name):
    """Show information about a table.

    Displays schema, partitioning, current snapshot, and data location.

    Example:
        faceberg --config=faceberg.yml info default.dataset1
    """
    catalog = ctx.obj["catalog"]

    console.print(f"[bold blue]Table:[/bold blue] {table_name}")

    try:
        table = catalog.load_table(table_name)
        console.print(f"[bold blue]Location:[/bold blue] {table.location()}")
        console.print("[bold blue]Schema:[/bold blue]")
        for field in table.schema().fields:
            console.print(f"  • {field.name}: {field.field_type}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


@main.command()
@click.argument("table_name")
@click.option("--limit", "-n", type=int, default=5, help="Number of rows to display")
@click.pass_context
def scan(ctx, table_name, limit):
    """Scan and display sample data from a table.

    Performs a simple scan operation to test table querying and displays
    the first few rows as a quick verification that the table is readable.

    Example:
        faceberg --config=faceberg.yml scan default.imdb
        faceberg --config=faceberg.yml scan default.imdb --limit=10
    """
    catalog = ctx.obj["catalog"]

    console.print(f"[bold blue]Scanning table:[/bold blue] {table_name}\n")

    try:
        # Load table
        table = catalog.load_table(table_name)

        # Perform scan and convert to pandas
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Reading data...", total=None)
            df = table.scan().to_pandas()
            progress.update(task, description=f"✓ Read {len(df)} rows")

        # Display basic info
        console.print(f"\n[bold green]Total rows:[/bold green] {len(df)}")
        console.print(f"[bold green]Columns:[/bold green] {list(df.columns)}\n")

        # Display sample rows
        if len(df) > 0:
            console.print(f"[bold blue]First {min(limit, len(df))} rows:[/bold blue]")
            console.print(df.head(limit).to_string(index=False))
        else:
            console.print("[yellow]Table is empty[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if ctx.obj.get("verbose"):
            import traceback

            console.print("\n[bold red]Traceback:[/bold red]")
            console.print(traceback.format_exc())


if __name__ == "__main__":
    main()

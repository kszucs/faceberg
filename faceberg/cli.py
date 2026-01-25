"""Command-line interface for Faceberg."""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from faceberg.catalog import LocalCatalog, RemoteCatalog

console = Console()


@click.group()
@click.argument("uri", type=str)
@click.version_option(version="0.1.0", prog_name="faceberg")
@click.option(
    "--token",
    "-t",
    help="HuggingFace API token (can also set HF_TOKEN env variable)",
    type=str,
    envvar="HF_TOKEN",
    default=None,
)
@click.pass_context
def main(ctx, uri, token):
    """Faceberg - Bridge HuggingFace datasets with Apache Iceberg.

    A command-line tool to expose HuggingFace datasets as Iceberg tables,
    enabling powerful analytics and time-travel capabilities.
    """
    # Create catalog instance based on URI
    if uri.startswith("hf://"):
        # Explicit HuggingFace protocol: hf://datasets/org/repo
        hf_repo = uri.replace("hf://datasets/", "")
        catalog = RemoteCatalog(hf_repo, hf_token=token)
    elif uri.startswith("file://"):
        # Explicit protocol or existing local path
        catalog = LocalCatalog(uri, hf_token=token)
    elif Path(uri).is_dir():
        catalog = LocalCatalog(uri, hf_token=token)
    else:
        catalog = RemoteCatalog(uri, hf_token=token)

    ctx.ensure_object(dict)
    ctx.obj["catalog"] = catalog
    ctx.obj["token"] = token


@main.command()
@click.argument("dataset")
@click.option("--table", "-t", help="Explicit table identifier (namespace.table)")
@click.option("--config", "-c", default="default", help="Dataset config name")
@click.pass_context
def add(ctx, dataset, table, config):
    """Add a table to the catalog.

    DATASET: HuggingFace dataset in format 'org/repo'

    By default, the table identifier is inferred from the dataset:
    org/repo -> namespace 'org', table 'repo' (identifier: org.repo)

    Examples:
        # Add with inferred identifier (deepmind.code_contests)
        faceberg add deepmind/code_contests

        # Add with explicit identifier
        faceberg add deepmind/code_contests --table myns.mytable

        # Add with non-default config
        faceberg add squad --config plain_text --table default.squad
    """
    catalog = ctx.obj["catalog"]

    # Determine table identifier
    if table:
        # Explicit identifier provided
        table_identifier = table
    else:
        # Infer from dataset: org/repo -> org.repo
        try:
            namespace, table_name = dataset.split("/", 1)
            table_identifier = f"{namespace}.{table_name}"
        except ValueError:
            console.print("[red]Error: dataset must be in format 'org/repo'[/red]")
            raise click.Abort()

    # Add dataset to catalog and create Iceberg table
    try:
        table = catalog.add_dataset(
            identifier=table_identifier, dataset=dataset, config=config
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
    except Exception as e:
        if "already exists" in str(e).lower():
            console.print(f"[yellow]Table {table_identifier} already exists[/yellow]")
        else:
            console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()

    console.print(f"[green]✓ Added {table_identifier} to catalog[/green]")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Config: {config}")
    console.print(f"  Location: {table.metadata_location}")


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
            tables = catalog.sync_datasets(table_name=table_name)
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
            df = table.scan(limit=limit).to_pandas()
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


if __name__ == "__main__":
    main()

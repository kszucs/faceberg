"""Command-line interface for Faceberg."""

import os
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from faceberg.catalog import LocalCatalog, RemoteCatalog
from faceberg.config import CatalogConfig

console = Console()


def _is_hf_repo_id(path: str) -> bool:
    """Check if path looks like a HuggingFace repo ID (org/repo format)."""
    return "/" in path and not path.startswith(("./", "../", "/"))


def _get_catalog(catalog_path: str, config: CatalogConfig = None):
    """Create catalog instance based on path (local or remote)."""
    if _is_hf_repo_id(catalog_path):
        # Remote catalog
        token = os.getenv("HF_TOKEN")
        return RemoteCatalog(hf_repo_id=catalog_path, hf_token=token, config=config)
    else:
        # Local catalog
        return LocalCatalog(location=catalog_path, config=config)


@click.group()
@click.version_option(version="0.1.0", prog_name="faceberg")
@click.option(
    "--catalog",
    type=str,
    default=".",
    help="Catalog location: local path or HuggingFace repo ID (org/repo)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to faceberg.yml config file (only needed for sync)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx, catalog, config, verbose):
    """Faceberg - Bridge HuggingFace datasets with Apache Iceberg.

    A command-line tool to expose HuggingFace datasets as Iceberg tables,
    enabling powerful analytics and time-travel capabilities.
    """
    ctx.ensure_object(dict)
    ctx.obj["catalog_path"] = catalog
    ctx.obj["config_path"] = config
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
        faceberg --catalog=. --config=faceberg.yml sync
        faceberg --catalog=org/catalog-repo sync namespace1.table1
    """
    catalog_path = ctx.obj["catalog_path"]
    config_path = ctx.obj["config_path"]

    # Config is required for sync
    if not config_path:
        console.print("[bold red]Error:[/bold red] --config is required for sync command")
        console.print("Usage: faceberg --catalog=. --config=faceberg.yml sync")
        raise click.Abort()

    # Load config and create catalog
    config = CatalogConfig.from_yaml(config_path)
    catalog = _get_catalog(catalog_path, config=config)

    console.print(f"[bold blue]Catalog:[/bold blue] {catalog.name}")
    console.print(f"[bold blue]Location:[/bold blue] {catalog.catalog_dir}")

    # Get HF token
    token = os.getenv("HF_TOKEN")

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
            tables = catalog.sync(
                config=config,
                token=token,
                table_name=table_name,
            )
            if tables:
                progress.update(task, description=f"✓ Synced {len(tables)} table(s)")
            else:
                progress.update(task, description="✓ All tables up-to-date")
            console.print("\n[bold green]Done![/bold green]")
        except Exception as e:
            progress.update(task, description=f"✗ Failed: {e}")
            raise


@main.command()
@click.argument("table_name", required=False)
@click.pass_context
def init(ctx, table_name):
    """Initialize catalog and create Iceberg tables from config.

    This is an alias for the 'sync' command. It discovers configured datasets
    and creates corresponding Iceberg metadata locally.

    Example:
        faceberg init
        faceberg init --config=my-config.yml
    """
    # Just invoke sync
    ctx.invoke(sync, table_name=table_name)


@main.command("list")
@click.pass_context
def list_tables(ctx):
    """List all tables in catalog.

    Example:
        faceberg --catalog=. list
        faceberg --catalog=org/catalog-repo list
    """
    catalog_path = ctx.obj["catalog_path"]
    catalog = _get_catalog(catalog_path)

    console.print(f"[bold blue]Catalog:[/bold blue] {catalog.name}")
    console.print(f"[bold blue]Location:[/bold blue] {catalog.catalog_dir}\n")

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
        faceberg --catalog=. info default.dataset1
    """
    catalog_path = ctx.obj["catalog_path"]
    catalog = _get_catalog(catalog_path)

    console.print(f"[bold blue]Table:[/bold blue] {table_name}")

    try:
        table = catalog.load_table(table_name)
        console.print(f"[bold blue]Location:[/bold blue] {table.location()}")
        console.print(f"[bold blue]Schema:[/bold blue]")
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
        faceberg --catalog=. scan default.imdb
        faceberg --catalog=. scan default.imdb --limit=10
        faceberg --catalog=org/catalog-repo scan default.dataset1
    """
    catalog_path = ctx.obj["catalog_path"]
    catalog = _get_catalog(catalog_path)

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


@main.command()
@click.pass_context
def push(ctx):
    """Push local catalog to HuggingFace.

    Note: With RemoteCatalog, changes are automatically pushed during sync.
    Use RemoteCatalog in your code for automatic syncing to HuggingFace.

    Example:
        faceberg --catalog=org/catalog-repo --config=faceberg.yml sync
    """
    console.print("[bold blue]Pushing catalog to HuggingFace...[/bold blue]")
    console.print("[yellow]⚠️  Use RemoteCatalog for automatic HF syncing[/yellow]")


@main.command()
@click.pass_context
def pull(ctx):
    """Pull catalog from HuggingFace to local.

    Note: With RemoteCatalog, the catalog is automatically cached locally.
    Simply use a HuggingFace repo ID as the catalog path.

    Example:
        faceberg --catalog=org/catalog-repo list
    """
    console.print("[bold blue]Pulling catalog from HuggingFace...[/bold blue]")
    console.print("[yellow]⚠️  Use RemoteCatalog to read from HF automatically[/yellow]")


if __name__ == "__main__":
    main()

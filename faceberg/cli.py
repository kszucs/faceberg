"""Command-line interface for Faceberg."""

import os
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from faceberg.catalog import LocalCatalog
from faceberg.config import CatalogConfig

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="faceberg")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="faceberg.yml",
    help="Path to faceberg.yml config file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx, config, verbose):
    """Faceberg - Bridge HuggingFace datasets with Apache Iceberg.

    A command-line tool to expose HuggingFace datasets as Iceberg tables,
    enabling powerful analytics and time-travel capabilities.
    """
    ctx.ensure_object(dict)
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
        faceberg sync                      # Sync all tables
        faceberg sync namespace1.table1    # Sync specific table
    """
    config_path = ctx.obj["config_path"]

    # Load config and create catalog
    config = CatalogConfig.from_yaml(config_path)
    catalog = LocalCatalog(
        name=config.name,
        location=config.location,
        config=config,
    )

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
        faceberg list
    """
    console.print("[bold blue]Listing tables...[/bold blue]")
    console.print("[yellow]⚠️  Not implemented yet[/yellow]")


@main.command()
@click.argument("table_name")
@click.pass_context
def info(ctx, table_name):
    """Show information about a table.

    Displays schema, partitioning, current snapshot, and data location.

    Example:
        faceberg info default.dataset1
    """
    console.print(f"[bold blue]Table info:[/bold blue] {table_name}")
    console.print("[yellow]⚠️  Not implemented yet[/yellow]")


@main.command()
@click.pass_context
def push(ctx):
    """Push local catalog to HuggingFace.

    Uploads .faceberg/ contents to HF catalog repository.

    Example:
        faceberg push
    """
    console.print("[bold blue]Pushing catalog to HuggingFace...[/bold blue]")
    console.print("[yellow]⚠️  Not implemented yet[/yellow]")


@main.command()
@click.pass_context
def pull(ctx):
    """Pull catalog from HuggingFace to local.

    Downloads catalog from HF repository to local .faceberg/ directory.

    Example:
        faceberg pull
    """
    console.print("[bold blue]Pulling catalog from HuggingFace...[/bold blue]")
    console.print("[yellow]⚠️  Not implemented yet[/yellow]")


if __name__ == "__main__":
    main()

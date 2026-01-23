"""Command-line interface for Faceberg."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from faceberg.config import FacebergConfig

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
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def init(ctx, config_file):
    """Initialize a new Faceberg catalog from config file.

    Creates the catalog directory structure and initializes catalog.json.

    Example:
        faceberg init faceberg.yml
    """
    console.print("[bold blue]Initializing Faceberg catalog...[/bold blue]")

    try:
        config = FacebergConfig.from_yaml(config_file)

        console.print(f"Catalog name: {config.catalog.name}")
        console.print(f"Catalog location: {config.catalog.location}")
        console.print(f"Datasets: {len(config.datasets)}")

        console.print("[yellow]⚠️  Not implemented yet[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@main.command()
@click.argument("table_name", required=False)
@click.pass_context
def create(ctx, table_name):
    """Create Iceberg tables for datasets in config.

    If TABLE_NAME is provided, creates only that table.
    Otherwise, creates all tables defined in config.

    Example:
        faceberg create                    # Create all tables
        faceberg create default.dataset1   # Create specific table
    """
    console.print("[bold blue]Creating Iceberg tables...[/bold blue]")
    if table_name:
        console.print(f"Table: {table_name}")
    else:
        console.print("Creating all tables from config")

    console.print("[yellow]⚠️  Not implemented yet[/yellow]")


@main.command()
@click.argument("table_name", required=False)
@click.pass_context
def sync(ctx, table_name):
    """Sync Iceberg metadata with current HF dataset state.

    Discovers new Parquet files and updates table metadata.

    Example:
        faceberg sync                      # Sync all tables
        faceberg sync default.dataset1     # Sync specific table
    """
    console.print("[bold blue]Syncing Iceberg tables...[/bold blue]")
    if table_name:
        console.print(f"Table: {table_name}")
    else:
        console.print("Syncing all tables")

    console.print("[yellow]⚠️  Not implemented yet[/yellow]")


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

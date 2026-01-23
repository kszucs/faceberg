"""Init command implementation."""

from pathlib import Path

from rich.console import Console

from faceberg.catalog import JsonCatalog
from faceberg.config import FacebergConfig

console = Console()


def init_catalog(config: FacebergConfig) -> JsonCatalog:
    """Initialize Faceberg catalog from config.

    Args:
        config: Faceberg configuration

    Returns:
        Initialized catalog
    """
    catalog_location = Path(config.catalog.location)

    console.print(f"[bold blue]Initializing catalog:[/bold blue] {config.catalog.name}")
    console.print(f"[bold blue]Location:[/bold blue] {catalog_location}")

    # Create catalog
    catalog = JsonCatalog(
        name=config.catalog.name,
        warehouse=str(catalog_location),
    )

    console.print(f"[green]✓[/green] Catalog initialized")

    # Create default namespace
    try:
        catalog.create_namespace("default")
        console.print(f"[green]✓[/green] Created namespace: default")
    except Exception:
        console.print(f"[yellow]![/yellow] Namespace 'default' already exists")

    console.print(f"\n[bold green]Catalog ready![/bold green]")
    console.print(f"Run [bold]faceberg create[/bold] to create tables for datasets")

    return catalog

"""Create command implementation."""

import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from faceberg.catalog import JsonCatalog
from faceberg.config import FacebergConfig
from faceberg.discovery import DatasetDiscovery
from faceberg.schema import infer_schema_from_parquet

console = Console()


def create_tables(
    config: FacebergConfig,
    table_name: Optional[str] = None,
) -> None:
    """Create Iceberg tables for datasets in config.

    Args:
        config: Faceberg configuration
        table_name: Specific table to create (None for all)
    """
    # Load catalog
    catalog = JsonCatalog(
        name=config.catalog.name,
        warehouse=str(config.catalog.location),
    )

    # Initialize discovery
    token = os.getenv("HF_TOKEN")
    discovery = DatasetDiscovery(token=token)

    # Determine which datasets to process
    if table_name:
        # Parse table name (format: namespace.dataset_config)
        parts = table_name.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid table name: {table_name}. Expected format: namespace.table_name")

        namespace, table = parts
        # Extract dataset name and config from table name
        # Assuming format: dataset_config or dataset_default
        if "_" in table:
            dataset_name, config_name = table.rsplit("_", 1)
        else:
            dataset_name = table
            config_name = "default"

        # Find matching dataset
        dataset_configs = [
            ds for ds in config.datasets
            if ds.name == dataset_name
        ]
        if not dataset_configs:
            raise ValueError(f"Dataset {dataset_name} not found in config")

        datasets_to_process = [(dataset_configs[0], [config_name])]
    else:
        # Process all datasets
        datasets_to_process = [
            (ds, ds.configs) for ds in config.datasets
        ]

    # Create tables
    for dataset_config, configs_filter in datasets_to_process:
        console.print(f"\n[bold blue]Processing dataset:[/bold blue] {dataset_config.repo}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Discovering dataset structure...", total=None)

            # Discover dataset
            try:
                dataset_info = discovery.discover_dataset(
                    repo_id=dataset_config.repo,
                    configs=configs_filter,
                )
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to discover dataset: {e}")
                continue

            progress.update(task, description=f"Found {len(dataset_info.configs)} config(s)")

            # Create table for each config
            for config_name in dataset_info.configs:
                progress.update(task, description=f"Creating table for config: {config_name}")

                try:
                    _create_table_for_config(
                        catalog=catalog,
                        dataset_config=dataset_config,
                        dataset_info=dataset_info,
                        config_name=config_name,
                        discovery=discovery,
                    )
                    console.print(f"[green]✓[/green] Created table: default.{dataset_config.name}_{config_name}")
                except Exception as e:
                    console.print(f"[red]✗[/red] Failed to create table for {config_name}: {e}")

    console.print(f"\n[bold green]Done![/bold green] Run [bold]faceberg list[/bold] to see tables")


def _create_table_for_config(
    catalog: JsonCatalog,
    dataset_config,
    dataset_info,
    config_name: str,
    discovery: DatasetDiscovery,
) -> None:
    """Create Iceberg table for a specific dataset config.

    Args:
        catalog: JsonCatalog instance
        dataset_config: Dataset configuration from faceberg.yml
        dataset_info: Discovered dataset information
        config_name: Configuration name
        discovery: DatasetDiscovery instance
    """
    # Generate table identifier
    table_id = f"default.{dataset_config.name}_{config_name}"

    # Check if table already exists
    if catalog.table_exists(table_id):
        console.print(f"[yellow]![/yellow] Table {table_id} already exists, skipping")
        return

    # Get sample Parquet file for schema inference
    sample_file = discovery.get_sample_parquet_file(dataset_info, config_name)

    # Infer schema
    schema = infer_schema_from_parquet(sample_file)

    # Get all Parquet files
    parquet_files = discovery.get_parquet_files_for_table(dataset_info, config_name)

    # Determine metadata and data locations
    metadata_location = f"{catalog.warehouse}/metadata/default_{dataset_config.name}_{config_name}"
    data_location = f"hf://datasets/{dataset_info.repo_id}"

    # Create table
    table = catalog.create_table(
        identifier=table_id,
        schema=schema,
        location=metadata_location,
        properties={
            "write.data.path": data_location,
            "faceberg.source.repo": dataset_info.repo_id,
            "faceberg.source.config": config_name,
        },
    )

    # Register existing Parquet files
    if parquet_files:
        table.add_files(file_paths=parquet_files)

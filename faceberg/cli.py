"""Command-line interface for Faceberg."""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .catalog import RemoteCatalog, catalog
from .config import Config
from .pretty import progress_bars, progress_tree, tree

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
    # Create catalog instance using factory function
    cat = catalog(uri, hf_token=token)
    ctx.ensure_object(dict)
    ctx.obj["catalog"] = cat
    ctx.obj["token"] = token

    console.print(f"[bold blue]🤗🧊 Catalog:[/bold blue] {cat.uri}\n")


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
    if table is None:
        identifier = tuple(dataset.split("/"))
    else:
        identifier = table
    name = ".".join(identifier)

    console.print(f"[bold blue]Adding dataset:[/bold blue] {dataset}")
    console.print(f"[bold blue]Table identifier:[/bold blue] {name}\n")

    # Add with live progress display
    with progress_bars(config, console, [identifier]) as progress_callback:
        table = catalog.add_dataset(
            identifier=identifier,
            repo=dataset,
            config=config,
            progress_callback=progress_callback,
        )

    # Display summary
    console.print(f"\n[green]✓ Added {name} to catalog[/green]")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Config: {config}")
    console.print(f"  Location: {table.metadata_location}")

    # Display table schema
    console.print("\n[bold blue]Table schema:[/bold blue]")
    console.print(table)


@main.command()
@click.argument("table_name", required=False)
@click.option("--tree-view", "-t", is_flag=True, help="Show the catalog as a tree")
@click.pass_context
def sync(ctx, table_name, tree_view):
    """Sync Iceberg tables with HuggingFace datasets from config.

    Discovers datasets and creates/updates Iceberg tables. For new tables,
    creates initial metadata and namespaces on-demand. For existing tables,
    checks if dataset revision has changed and skips if already up-to-date.

    Example:
        faceberg --config=faceberg.yml sync
        faceberg --config=faceberg.yml sync namespace1.table1
    """
    catalog = ctx.obj["catalog"]

    # Build tree view with tracking
    config = catalog.config()

    if tree_view:
        with progress_tree(config, console) as progress_callback:
            catalog.sync_datasets(progress_callback=progress_callback)
    else:
        datasets = config.datasets()
        with progress_bars(config, console, list(datasets)) as progress_callback:
            catalog.sync_datasets(progress_callback=progress_callback)

    console.print("\n[bold green]✓ Sync complete![/bold green]")


@main.command()
@click.argument("config_path", required=False)
@click.option("--sync", "-s", is_flag=True, help="Sync after initialization")
@click.pass_context
def init(ctx, config_path, sync):
    """Initialize a catalog with optional initial configuration.

    For local catalogs, creates the directory and faceberg.yml file.
    For remote catalogs (HuggingFace), creates a new dataset repository.

    If a config file is provided via --config/-c, the catalog will be populated
    with the tables defined in that file. If no config is specified, looks for
    faceberg.yml in the current directory. If neither is found, creates an
    empty catalog.

    Example:
        # Initialize with explicit config file
        faceberg catalog.db init tables.yml

        # Initialize with auto-discovered config (looks for ./faceberg.yml)
        faceberg catalog.db init

        # Initialize remote catalog (creates HF dataset repo)
        export HF_TOKEN=your_token
        faceberg hf://datasets/user/catalog init tables.yml
    """
    catalog = ctx.obj["catalog"]
    is_remote = isinstance(catalog, RemoteCatalog)

    if is_remote:
        console.print(f"[bold blue]Initializing remote catalog:[/bold blue] {catalog.uri}")
    else:
        console.print(f"[bold blue]Initializing local catalog:[/bold blue] {catalog.uri}")

    # Load config if provided or auto-discover
    config_path = Path(config_path or "faceberg.yml")
    if config_path.exists():
        try:
            config = Config.from_yaml(config_path)
            console.print(f"[dim]Loading config from: {config_path}[/dim]")
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            raise click.Abort()
    else:
        config = Config()

    # Initialize catalog with optional config
    catalog.init(config)
    console.print("[bold green]✓ Catalog initialized successfully![/bold green]")

    # Sync datasets if requested
    if sync:
        console.print("\n[bold blue]Syncing datasets...[/bold blue]")
        with progress_bars(config, console, list(config.datasets())) as progress_callback:
            catalog.sync_datasets(progress_callback=progress_callback)
        console.print("\n[bold green]✓ Sync complete![/bold green]")

    # Display additional info for remote catalogs
    if is_remote:
        if catalog.hf_repo_type == "space":
            space_url = catalog.hf_repo.replace("/", "-")
            console.print(f"\n[cyan]Space URL:[/cyan] https://{space_url}.hf.space")
        console.print(
            "[cyan]Repository:[/cyan] https://huggingface.co/"
            f"{catalog.hf_repo_type}s/{catalog.hf_repo}"
        )

    # Recommend next steps
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  • Run [cyan]faceberg add <dataset>[/cyan] to add tables")
    console.print("  • Run [cyan]faceberg sync[/cyan] to sync tables from datasets")
    console.print("  • Use [cyan]faceberg scan <table>[/cyan] to view sample data")
    console.print("  • Run [cyan]faceberg serve[/cyan] to start the REST catalog server")
    console.print("  • Run [cyan]faceberg quack[/cyan] to open DuckDB with the catalog")


@main.command("list")
@click.pass_context
def list_tables(ctx):
    """List all tables in catalog.

    Example:
        faceberg --config=faceberg.yml list
    """
    catalog = ctx.obj["catalog"]
    config = catalog.config()
    rendered = tree(config)

    console.print(f"[bold blue]Catalog:[/bold blue] {catalog.uri}\n")
    console.print(rendered)


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


@main.command("remove")
@click.argument("identifier")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def remove(ctx, identifier, yes):
    """Remove a table or namespace from the catalog.

    Automatically detects whether the identifier is a table (namespace.table)
    or a namespace. For namespaces, they must be empty before removal.

    Examples:
        # Remove a table
        faceberg catalog.db remove default.dataset1

        # Remove a namespace (must be empty)
        faceberg catalog.db remove myns

        # Skip confirmation prompt
        faceberg catalog.db remove default.dataset1 --yes
    """
    catalog = ctx.obj["catalog"]

    # Determine if identifier is a table or namespace
    if "." in identifier:
        # Likely a table identifier (namespace.table)
        if catalog.table_exists(identifier):
            # It's a table
            if not yes:
                if not click.confirm(f"Are you sure you want to remove table '{identifier}'?"):
                    console.print("[yellow]Aborted[/yellow]")
                    return

            try:
                catalog.drop_table(identifier)
                console.print(f"[green]✓ Removed table {identifier}[/green]")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                raise click.Abort()
        else:
            console.print(f"[bold red]Error:[/bold red] Table {identifier} does not exist")
            raise click.Abort()
    else:
        # It's a namespace identifier
        namespaces = catalog.list_namespaces()
        namespace_tuple = tuple([identifier])

        if namespace_tuple not in namespaces:
            console.print(f"[bold red]Error:[/bold red] Namespace {identifier} does not exist")
            raise click.Abort()

        # Check if namespace has tables
        tables = catalog.list_tables(identifier)
        if tables:
            console.print(f"[bold red]Error:[/bold red] Namespace {identifier} is not empty")
            console.print(f"  Contains {len(tables)} table(s):")
            for table in tables:
                table_str = ".".join(table) if isinstance(table, tuple) else table
                console.print(f"    • {table_str}")
            console.print(
                "\n[yellow]Remove all tables first before removing the namespace[/yellow]"
            )
            raise click.Abort()

        # Prompt for confirmation
        if not yes:
            if not click.confirm(f"Are you sure you want to remove namespace '{identifier}'?"):
                console.print("[yellow]Aborted[/yellow]")
                return

        try:
            catalog.drop_namespace(identifier)
            console.print(f"[green]✓ Removed namespace {identifier}[/green]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise click.Abort()


@main.command()
@click.option("--endpoint", default=None, help="REST catalog endpoint URL")
@click.pass_context
def quack(ctx, endpoint):
    """Open DuckDB shell with REST catalog attached.

    Starts an interactive DuckDB session with the Iceberg REST catalog
    pre-configured. For remote catalogs on HuggingFace Spaces, automatically
    connects to the Space's REST endpoint. For local catalogs, connects to
    localhost:8181 by default.

    Examples:
        # Connect to remote catalog (auto-detects Space URL)
        faceberg user/catalog quack

        # Connect to local REST server (uses localhost:8181 by default)
        faceberg /tmp/catalog quack

        # Connect to custom endpoint
        faceberg /tmp/catalog quack --endpoint http://localhost:9000
    """
    try:
        from faceberg.interactive import quack as quack_fn

        # Auto-detect endpoint if not specified
        if endpoint is None:
            catalog = ctx.obj["catalog"]
            if isinstance(catalog, RemoteCatalog) and catalog._hf_repo_type == "space":
                # Construct Space URL from repo ID (e.g., "user/catalog" -> "user-catalog.hf.space")
                space_url = catalog._hf_repo.replace("/", "-")
                endpoint = f"https://{space_url}.hf.space"
                console.print(f"[dim]Connecting to Space: {endpoint}[/dim]")
            else:
                # Local catalog - use localhost
                endpoint = "http://localhost:8181"
                console.print(f"[dim]Connecting to local server: {endpoint}[/dim]")

        quack_fn(endpoint=endpoint, catalog_name="faceberg")
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] DuckDB not installed. Install with: pip install duckdb"
        )
        raise click.Abort()
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8181, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--prefix", default="", help="URL prefix for REST API")
@click.pass_context
def serve(ctx, host, port, reload, prefix):
    """Start REST catalog server.

    Exposes the catalog via HTTP endpoints following the Apache Iceberg
    REST catalog specification. Supports both LocalCatalog and RemoteCatalog.

    The server provides read-only operations:
    - List and load namespaces
    - List and load tables
    - Check existence of namespaces and tables

    Examples:
        # Serve local catalog
        faceberg /path/to/catalog serve --port 8181

        # Serve remote catalog on HuggingFace Hub
        faceberg hf://datasets/org/repo serve --token $HF_TOKEN

        # Enable auto-reload for development
        faceberg /tmp/catalog serve --reload

        # Use custom URL prefix
        faceberg /tmp/catalog serve --prefix my-catalog
    """
    try:
        import uvicorn

        from faceberg.server import create_app
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] Server dependencies not installed. "
            "Install with: pip install 'faceberg[server]' or pip install litestar uvicorn"
        )
        raise click.Abort()

    catalog = ctx.obj["catalog"]
    token = ctx.obj.get("token")

    console.print("[bold green]Starting REST catalog server...[/bold green]")
    console.print(f"  Catalog: {catalog.uri}")
    console.print(f"  Listening on: http://{host}:{port}")
    console.print(f"  API docs: http://{host}:{port}/schema")
    if prefix:
        console.print(f"  URL prefix: /{prefix}")

    app = create_app(catalog.uri, hf_token=token, prefix=prefix)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

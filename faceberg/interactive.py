"""Interactive querying utilities for Faceberg catalogs.

This module provides supplemental utilities for hands-on experimentation
with Iceberg catalogs using DuckDB.
"""

import os
import shutil
import subprocess
import tempfile

try:
    import readline  # noqa: F401 - enables arrow keys and history in input()
except ImportError:
    pass  # readline not available on this platform

from rich.console import Console

console = Console()


# TODO(kszucs): rename it to shell.py, maybe other shells could be supported like clickhouse

def quack(endpoint, catalog_name="iceberg_catalog"):
    """Open interactive DuckDB shell with REST catalog attached.

    Args:
        endpoint: REST catalog endpoint URL (e.g., http://localhost:8181)
        catalog_name: Name for the attached catalog in DuckDB

    Raises:
        ImportError: If duckdb is not installed
        Exception: If connection or attachment fails
    """

    console.print("[bold green]Starting DuckDB shell with Iceberg REST catalog...[/bold green]")
    console.print(f"  Endpoint: {endpoint}")
    console.print(f"  Catalog name: {catalog_name}")

    # Try to find the native duckdb CLI
    duckdb_cli = shutil.which("duckdb")

    # Launch native DuckDB CLI with catalog pre-attached
    # Create temporary initialization script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
        init_script = f.name
        f.write(f"""-- Faceberg initialization script
INSTALL httpfs;
LOAD httpfs;
INSTALL iceberg;
LOAD iceberg;

-- Attach REST catalog
ATTACH 'warehouse' AS {catalog_name} (
    TYPE ICEBERG,
    ENDPOINT '{endpoint}',
    AUTHORIZATION_TYPE 'none'
);

-- Welcome message
.print
.print ✓ Connected to Iceberg REST catalog
.print   Catalog name: {catalog_name}
.print   Endpoint: {endpoint}
.print
.print Quick tips:
.print   • List catalogs: SELECT * FROM duckdb_databases();
.print   • Show all namespaces: SHOW ALL TABLES;
.print   • List tables: SHOW TABLES FROM {catalog_name};
.print   • Exit: .quit or Ctrl+D
.print
""")

    try:
        console.print("\n[dim]Loading extensions and attaching catalog...[/dim]")

        # Launch DuckDB CLI with initialization script
        subprocess.run([duckdb_cli, "-init", init_script], check=False)

    finally:
        # Clean up temporary file
        try:
            os.unlink(init_script)
        except Exception:
            pass

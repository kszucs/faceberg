# Faceberg

Bridge HuggingFace datasets with Apache Iceberg.

## Overview

Faceberg is a command-line tool that exposes HuggingFace datasets as Apache Iceberg tables, enabling:

- **Time travel**: Query historical versions of your data
- **Schema evolution**: Add/rename/drop columns without breaking existing queries
- **Partitioning**: Efficient query pruning on large datasets
- **ACID guarantees**: Reliable reads and writes
- **Metadata separation**: Iceberg metadata in one repo, data files in original dataset repos

## Installation

```bash
pip install faceberg
```

Or for development:

```bash
git clone https://github.com/kszucs/faceberg
cd faceberg
pip install -e .
```

## Quick Start

### Option 1: Read from Remote Catalog (HuggingFace Hub)

Connect to an existing catalog hosted on HuggingFace:

```python
from faceberg.catalog import RemoteCatalog

# Connect to remote catalog (no config needed for reading)
catalog = RemoteCatalog(
    hf_repo="your-org/your-catalog-dataset",
    hf_token=None,  # Or your HF token for private repos
)

# Load and query a table
table = catalog.load_table("default.my_table")
df = table.scan().to_pandas()
print(df.head())
```

### Option 2: Create Remote Catalog (HuggingFace Hub)

Initialize a new catalog on HuggingFace and sync datasets:

```python
from faceberg.catalog import RemoteCatalog
import os

# Create faceberg.yml config file first:
# uri: hf://datasets/your-org/your-catalog-dataset
#
# default:
#   imdb:
#     dataset: stanfordnlp/imdb
#     config: plain_text

# Create remote catalog (initializes HF dataset repository)
catalog = RemoteCatalog(
    hf_repo="your-org/your-catalog-dataset",
    hf_token=os.getenv("HF_TOKEN"),  # Required for creating repos
)

# Initialize the catalog (creates empty HF repo)
catalog.init()

# Sync datasets to create Iceberg tables
tables = catalog.sync_datasets()
print(f"Synced {len(tables)} tables")

# Query the table
table = catalog.load_table("default.imdb")
df = table.scan().to_pandas()
print(df.head())
```

### Option 3: Create Local Catalog from HuggingFace Datasets

Sync HuggingFace datasets to a local Iceberg catalog:

```python
from faceberg.catalog import LocalCatalog
from faceberg.database import Catalog, Namespace, Table

# Create faceberg.yml config file first:
# uri: mycatalog
#
# default:
#   imdb:
#     dataset: stanfordnlp/imdb
#     config: plain_text

# Or create programmatically and save
catalog_dir = "mycatalog"
config = Catalog(
    uri=f"file:///{catalog_dir}",
    namespaces={
        "default": Namespace(
            tables={
                "imdb": Table(
                    dataset="stanfordnlp/imdb",
                    uri="",  # Empty until synced
                    config="plain_text",
                ),
            }
        )
    },
)
config.to_yaml(f"{catalog_dir}/faceberg.yml")

# Create local catalog
catalog = LocalCatalog(path=catalog_dir)

# Sync datasets to create Iceberg tables
tables = catalog.sync_datasets()
print(f"Synced {len(tables)} tables")

# Query the table
table = catalog.load_table("default.imdb")
df = table.scan().to_pandas()
print(df.head())
```

### Option 4: Read Existing Local Catalog

Read from a local catalog without syncing:

```python
from faceberg.catalog import LocalCatalog

# Open existing local catalog (no config needed)
catalog = LocalCatalog(path="mycatalog/")

# List available tables
for ns in catalog.list_namespaces():
    print(f"Namespace: {ns}")
    for table_id in catalog.list_tables(ns):
        print(f"  - {table_id}")

# Load and query a table
table = catalog.load_table("default.imdb")
df = table.scan().to_pandas()
```

### Option 5: Query with DuckDB

DuckDB can read Iceberg tables created by Faceberg:

```python
import duckdb

# Create DuckDB connection and load extensions
conn = duckdb.connect()
conn.execute("INSTALL httpfs")
conn.execute("LOAD httpfs")
conn.execute("INSTALL iceberg")
conn.execute("LOAD iceberg")

# Query the Iceberg table
metadata_path = "mycatalog/default/imdb/metadata/v1.metadata.json"
result = conn.execute(f"""
    SELECT split, COUNT(*) as count
    FROM iceberg_scan('{metadata_path}')
    GROUP BY split
    ORDER BY split
""").fetchall()

print(result)
```

## CLI Usage

The CLI operates on catalog URIs. Use local paths for local catalogs or `hf://` URIs for remote catalogs:

```bash
# Initialize a new local catalog
faceberg mycatalog init

# Initialize a new remote catalog (creates HF dataset repository)
export HF_TOKEN=your_token
faceberg hf://datasets/org/catalog-repo init

# Sync datasets to local catalog (reads faceberg.yml from catalog directory)
faceberg mycatalog sync

# Sync specific table
faceberg mycatalog sync default.my_table

# Sync to remote catalog (automatically pushes to HuggingFace)
faceberg hf://datasets/org/catalog-repo sync

# List tables in local catalog
faceberg mycatalog list

# List tables in remote catalog
faceberg hf://datasets/org/catalog-repo list

# Show table info
faceberg mycatalog info default.my_table

# Scan and display sample data from a table
faceberg mycatalog scan default.my_table
faceberg mycatalog scan default.my_table --limit=10
```

### Config File Format

Create a `faceberg.yml` file in your catalog directory (only needed for syncing):

```yaml
# Absolute Catalog URI (required)
uri: file://path/to/mycatalog  # For local catalogs
# uri: hf://datasets/org/repo  # For remote catalogs

# Namespaces (dict of namespace_name -> tables)
default:
  imdb:
    dataset: stanfordnlp/imdb
    config: plain_text
  glue:
    dataset: glue
    config: mrpc

analytics:
  sales:
    dataset: your-org/sales-data
    config: default
```

After syncing, the `uri` field in each table will be automatically populated with the metadata location:

```yaml
uri: mycatalog

default:
  imdb:
    dataset: stanfordnlp/imdb
    config: plain_text
    uri: file:///mycatalog/default/imdb/metadata/v1.metadata.json
```

## Features

- **Flexible catalogs**: Local directories or HuggingFace repositories
- **Config-driven syncing**: Define datasets to sync in YAML
- **Read without config**: Open existing catalogs without configuration
- **Remote-first**: Catalogs on HuggingFace can be read by anyone
- **Dataset-aware**: Handles multiple configs and splits
- **hf:// protocol**: Leverages HF dataset storage efficiently

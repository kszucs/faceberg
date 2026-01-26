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

Choose your path:

- **[Local Catalog Quickstart](QUICKSTART_LOCAL.md)** - Create a local Iceberg catalog using the CLI
- **[Remote Catalog Quickstart](QUICKSTART_REMOTE.md)** - Create a remote catalog on HuggingFace Hub using the CLI

## How It Works

Faceberg bridges HuggingFace datasets with Apache Iceberg by creating lightweight metadata tables that reference the original dataset files. Here's what happens under the hood:

### The Sync Process

1. **Fetch Dataset Info**: Faceberg downloads the dataset metadata from HuggingFace (schema, splits, file locations)
2. **Create Iceberg Metadata**: Generates Iceberg table metadata (schema, manifest lists, manifest files) that point to the original Parquet files
3. **Store Metadata**: Saves the Iceberg metadata locally or on HuggingFace Hub
4. **No Data Duplication**: The actual dataset files remain on HuggingFace; only lightweight metadata is stored in your catalog

### Metadata Separation

```
HuggingFace Dataset Repo              Your Catalog (Local or HF)
┌───────────────────────┐            ┌──────────────────────────┐
│ org/dataset-name/     │            │ mycatalog/               │
│ ├── data/             │            │ ├── faceberg.yml         │
│ │   ├── train-*.pq ◄──┼────────────┼─│ └── default/           │
│ │   └── test-*.pq  ◄──┼────────────┼─│     └── my_table/      │
│ └── ...               │            │ │         └── metadata/  │
└───────────────────────┘            │ │             ├── v1.metadata.json
                                     │ │             ├── snap-*.avro  ◄─┐
                                     │ │             └── *.avro  ◄──────┤
                                     └─┴───────────────────────────────┘│
                                                                         │
                         Avro manifest files contain hf:// URIs ────────┘
                         pointing to original Parquet files
```

The Iceberg metadata structure:
- **v1.metadata.json**: Table schema, current snapshot ID, and table properties
- **snap-\*.avro** (manifest lists): Track which manifest files belong to each snapshot
- **\*.avro** (manifest files): Contain `hf://` URIs pointing to original Parquet files, plus partition info and file statistics
- **version-hint.text**: Quick reference to the latest metadata version

This design enables:
- **Zero data duplication**: Original files stay where they are
- **Efficient queries**: Iceberg's metadata allows predicate pushdown and partition pruning
- **Standard tooling**: Query with DuckDB, PyIceberg, Spark, or any Iceberg-compatible engine
- **Versioning**: Track dataset revisions via Iceberg snapshots

## Usage Patterns

Faceberg supports both CLI and programmatic workflows:

### CLI Workflow

Best for quick exploration and automation:

```bash
# Initialize catalog
faceberg mycatalog init

# Add datasets
faceberg mycatalog add org/dataset-name --config default

# Sync to create Iceberg metadata
faceberg mycatalog sync

# Query
faceberg mycatalog scan default.dataset_name
```

### Programmatic Workflow (Local Catalog)

Best for notebooks and custom pipelines:

```python
from faceberg.catalog import LocalCatalog
from faceberg.database import Catalog, Namespace, Table

# Define catalog configuration
config = Catalog(
    uri="file:///mycatalog",
    namespaces={
        "default": Namespace(
            tables={
                "imdb": Table(
                    dataset="stanfordnlp/imdb",
                    config="plain_text",
                ),
            }
        )
    },
)

# Save and sync
config.to_yaml("mycatalog/faceberg.yml")
catalog = LocalCatalog(path="mycatalog")
catalog.sync_datasets()

# Query with PyIceberg
table = catalog.load_table("default.imdb")
df = table.scan().to_pandas()
print(df.head())
```

### Programmatic Workflow (Remote Catalog)

Share catalogs on HuggingFace:

```python
from faceberg.catalog import RemoteCatalog
import os

# Create and sync remote catalog
catalog = RemoteCatalog(
    hf_repo="your-org/your-catalog",
    hf_token=os.getenv("HF_TOKEN"),
)
catalog.init()

# Add tables via config or programmatically
# (config editing same as local, just with hf:// URIs)

catalog.sync_datasets()

# Anyone can now read your catalog
public_catalog = RemoteCatalog(hf_repo="your-org/your-catalog")
table = public_catalog.load_table("default.my_table")
```

### Query with DuckDB

Use standard Iceberg tooling:

```python
import duckdb

conn = duckdb.connect()
conn.execute("INSTALL httpfs; LOAD httpfs")
conn.execute("INSTALL iceberg; LOAD iceberg")

# Query local catalog
result = conn.execute("""
    SELECT * FROM iceberg_scan('mycatalog/default/my_table/metadata/v1.metadata.json')
    LIMIT 10
""").fetchall()

# Query remote catalog (works with hf:// URIs)
result = conn.execute("""
    SELECT * FROM iceberg_scan('hf://datasets/org/catalog/default/my_table/metadata/v1.metadata.json')
    LIMIT 10
""").fetchall()
```

## CLI Reference

The CLI operates on catalog URIs. Use local paths for local catalogs or `org/repo` format for remote catalogs on HuggingFace:

```bash
# Initialize a new local catalog
faceberg mycatalog init

# Initialize a new remote catalog (creates HF dataset repository)
export HF_TOKEN=your_token
faceberg org/catalog-repo init

# Add datasets to catalog
faceberg mycatalog add openai/gsm8k --config main
faceberg org/catalog-repo add deepmind/code_contests --config default

# Sync datasets to local catalog (reads faceberg.yml from catalog directory)
faceberg mycatalog sync

# Sync specific table
faceberg mycatalog sync default.my_table

# Sync to remote catalog (automatically pushes to HuggingFace)
faceberg org/catalog-repo sync

# List tables in local catalog
faceberg mycatalog list

# List tables in remote catalog
faceberg org/catalog-repo list

# Show table info
faceberg mycatalog info default.my_table

# Scan and display sample data from a table
faceberg mycatalog scan default.my_table
faceberg mycatalog scan default.my_table --limit=10
```

## Features

- **Flexible catalogs**: Local directories or HuggingFace repositories
- **Config-driven syncing**: Define datasets to sync in YAML
- **Read without config**: Open existing catalogs without configuration
- **Remote-first**: Catalogs on HuggingFace can be read by anyone
- **Dataset-aware**: Handles multiple configs and splits
- **hf:// protocol**: Leverages HF dataset storage efficiently

# Quickstart: Remote Catalog

This guide shows how to create a remote Iceberg catalog on HuggingFace Hub using the Faceberg CLI.

## Prerequisites

1. Install Faceberg:
```bash
pip install faceberg
```

2. Set up your HuggingFace token (required for creating/updating remote catalogs):
```bash
export HF_TOKEN=your_huggingface_token
```

Get your token from [HuggingFace Settings](https://huggingface.co/settings/tokens).

## Step 1: Initialize Remote Catalog

Create a new remote catalog on HuggingFace Hub:

```bash
faceberg your-username/testcatalog init
```

This creates a new dataset repository on HuggingFace that will store your Iceberg catalog metadata.

## Step 2: Add HuggingFace Datasets

Add three tables to your remote catalog using the CLI. The `add` command automatically fetches dataset information and generates the metadata URIs:

```bash
# Add DeepMind Code Contests
faceberg your-username/testcatalog add deepmind/code_contests --config default

# Add OpenAI HumanEval
faceberg your-username/testcatalog add openai/openai_humaneval --config openai_humaneval

# Add GSM8K
faceberg your-username/testcatalog add openai/gsm8k --config main
```

## Step 3: Sync Datasets

Sync the datasets to create Iceberg table metadata and push to HuggingFace:

```bash
faceberg your-username/testcatalog sync
```

This downloads dataset metadata, creates Iceberg table files, and automatically pushes them to your HuggingFace repository.

## Step 4: Query with Faceberg

List available tables:

```bash
faceberg your-username/testcatalog list
```

Scan tables to see sample data:

```bash
# View first 10 rows of Code Contests
faceberg your-username/testcatalog scan default.code_contests --limit 10

# View first 10 rows of HumanEval
faceberg your-username/testcatalog scan default.openai_humaneval --limit 10

# View first 10 rows of GSM8K
faceberg your-username/testcatalog scan default.gsm8k --limit 10
```

## Step 5: Query with DuckDB

You can query the remote Iceberg tables directly with DuckDB using the explicit `hf://` metadata.json paths:

```python
import duckdb

# Create DuckDB connection and load extensions
conn = duckdb.connect()
conn.execute("INSTALL httpfs")
conn.execute("LOAD httpfs")
conn.execute("INSTALL iceberg")
conn.execute("LOAD iceberg")

# Query Code Contests table from HuggingFace
result = conn.execute("""
    SELECT name, description, difficulty
    FROM iceberg_scan('hf://datasets/your-username/testcatalog/default/code_contests/metadata/v1.metadata.json')
    LIMIT 5
""").fetchall()

print("Code Contests Sample:")
for row in result:
    print(row)

# Query OpenAI HumanEval table from HuggingFace
result = conn.execute("""
    SELECT task_id, prompt
    FROM iceberg_scan('hf://datasets/your-username/testcatalog/default/openai_humaneval/metadata/v1.metadata.json')
    LIMIT 5
""").fetchall()

print("\nHumanEval Sample:")
for row in result:
    print(row)

# Query GSM8K table from HuggingFace
result = conn.execute("""
    SELECT question, answer
    FROM iceberg_scan('hf://datasets/your-username/testcatalog/default/gsm8k/metadata/v1.metadata.json')
    LIMIT 5
""").fetchall()

print("\nGSM8K Sample:")
for row in result:
    print(row)
```

## What's on HuggingFace?

After syncing, your HuggingFace dataset repository contains:

```
your-username/testcatalog/
├── faceberg.yml           # Catalog configuration
└── default/               # Namespace
    ├── code_contests/     # Table directory
    │   └── metadata/
    │       ├── v1.metadata.json           # Main Iceberg metadata
    │       ├── snap-*.avro                # Manifest list (snapshot)
    │       ├── *.avro                     # Manifest files (data file references)
    │       └── version-hint.text          # Version tracking
    ├── openai_humaneval/
    │   └── metadata/
    │       ├── v1.metadata.json
    │       ├── snap-*.avro
    │       ├── *.avro
    │       └── version-hint.text
    └── gsm8k/
        └── metadata/
            ├── v1.metadata.json
            ├── snap-*.avro
            ├── *.avro
            └── version-hint.text
```

Each table's metadata directory contains:
- **v1.metadata.json**: The main Iceberg table metadata (schema, partitioning, snapshots)
- **snap-*.avro**: Manifest list files that track which manifest files belong to each snapshot
- **\*.avro**: Manifest files containing references to the actual data files (Parquet files on HuggingFace)
- **version-hint.text**: Tracks the current metadata version for quick lookups

The `faceberg.yml` file in your HuggingFace repository contains:

```yaml
uri: hf://datasets/your-username/testcatalog

default:
  code_contests:
    dataset: deepmind/code_contests
    revision: <commit-sha>
    config: default
    uri: hf://datasets/your-username/testcatalog/default/code_contests/metadata/v1.metadata.json
  openai_humaneval:
    dataset: openai/openai_humaneval
    revision: <commit-sha>
    config: openai_humaneval
    uri: hf://datasets/your-username/testcatalog/default/openai_humaneval/metadata/v1.metadata.json
  gsm8k:
    dataset: openai/gsm8k
    revision: <commit-sha>
    config: main
    uri: hf://datasets/your-username/testcatalog/default/gsm8k/metadata/v1.metadata.json
```

## Sharing Your Catalog

Since your catalog is hosted on HuggingFace, anyone can read it without authentication:

```python
from faceberg.catalog import RemoteCatalog

# Anyone can read your public catalog
catalog = RemoteCatalog(hf_repo="your-username/testcatalog")
table = catalog.load_table("default.code_contests")
df = table.scan().to_pandas()
```

To make it private, set your HuggingFace repository to private.

## Next Steps

- Explore [QUICKSTART_LOCAL.md](QUICKSTART_LOCAL.md) to learn about local catalogs
- Read the main [README.md](README.md) for programmatic usage and advanced features

# Quickstart: Local Catalog

This guide shows how to create a local Iceberg catalog from HuggingFace datasets using the Faceberg CLI.

## Prerequisites

Install Faceberg:
```bash
pip install faceberg
```

## Step 1: Initialize Local Catalog

Create a new local catalog directory:

```bash
faceberg mycatalog init
```

This creates a `mycatalog/` directory with an empty catalog structure.

## Step 2: Add HuggingFace Datasets

Add three tables to your catalog using the CLI. The `add` command automatically fetches dataset information and generates the metadata URIs:

```bash
# Add DeepMind Code Contests
faceberg mycatalog add deepmind/code_contests --config default

# Add OpenAI HumanEval
faceberg mycatalog add openai/openai_humaneval --config openai_humaneval

# Add GSM8K
faceberg mycatalog add openai/gsm8k --config main
```

## Step 3: Sync Datasets

Sync the datasets to create Iceberg table metadata:

```bash
faceberg mycatalog sync
```

This downloads dataset metadata and creates Iceberg table files locally.

## Step 4: Query with Faceberg

List available tables:

```bash
faceberg mycatalog list
```

Scan tables to see sample data:

```bash
# View first 10 rows of Code Contests
faceberg mycatalog scan default.code_contests --limit 10

# View first 10 rows of HumanEval
faceberg mycatalog scan default.openai_humaneval --limit 10

# View first 10 rows of GSM8K
faceberg mycatalog scan default.gsm8k --limit 10
```

## Step 5: Query with DuckDB

You can query the Iceberg tables directly with DuckDB using the explicit metadata.json paths:

```python
import duckdb

# Create DuckDB connection and load extensions
conn = duckdb.connect()
conn.execute("INSTALL httpfs")
conn.execute("LOAD httpfs")
conn.execute("INSTALL iceberg")
conn.execute("LOAD iceberg")

# Query Code Contests table
result = conn.execute("""
    SELECT name, description, difficulty
    FROM iceberg_scan('mycatalog/default/code_contests/metadata/v1.metadata.json')
    LIMIT 5
""").fetchall()

print("Code Contests Sample:")
for row in result:
    print(row)

# Query OpenAI HumanEval table
result = conn.execute("""
    SELECT task_id, prompt
    FROM iceberg_scan('mycatalog/default/openai_humaneval/metadata/v1.metadata.json')
    LIMIT 5
""").fetchall()

print("\nHumanEval Sample:")
for row in result:
    print(row)

# Query GSM8K table
result = conn.execute("""
    SELECT question, answer
    FROM iceberg_scan('mycatalog/default/gsm8k/metadata/v1.metadata.json')
    LIMIT 5
""").fetchall()

print("\nGSM8K Sample:")
for row in result:
    print(row)
```

## What's Inside Your Catalog?

After syncing, your local catalog structure looks like this:

```
mycatalog/
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

The `faceberg.yml` file contains the catalog configuration:

```yaml
uri: file:///path/to/mycatalog

default:
  code_contests:
    dataset: deepmind/code_contests
    revision: <commit-sha>
    config: default
    uri: file:///path/to/mycatalog/default/code_contests/metadata/v1.metadata.json
  openai_humaneval:
    dataset: openai/openai_humaneval
    revision: <commit-sha>
    config: openai_humaneval
    uri: file:///path/to/mycatalog/default/openai_humaneval/metadata/v1.metadata.json
  gsm8k:
    dataset: openai/gsm8k
    revision: <commit-sha>
    config: main
    uri: file:///path/to/mycatalog/default/gsm8k/metadata/v1.metadata.json
```

## Next Steps

- Explore [QUICKSTART_REMOTE.md](QUICKSTART_REMOTE.md) to learn about remote catalogs on HuggingFace
- Read the main [README.md](README.md) for programmatic usage and advanced features

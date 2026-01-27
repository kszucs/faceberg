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

Create a new remote catalog on HuggingFace Hub (you can use any repository name):

```bash
faceberg user/catalog init
```

This creates a HuggingFace Space at:
```
https://user-catalog.hf.space
```

The Space automatically deploys a REST server that exposes your catalog following the Apache Iceberg REST specification. The API is accessible at endpoints like:
- `https://user-catalog.hf.space/v1/config` - Catalog configuration
- `https://user-catalog.hf.space/v1/namespaces` - List namespaces
- `https://user-catalog.hf.space/v1/namespaces/{namespace}/tables` - List tables

The Space also stores all your Iceberg catalog metadata files.

## Step 2: Add HuggingFace Datasets

Add three tables to your remote catalog using the CLI. The `add` command automatically fetches dataset information and generates the metadata URIs:

```bash
# Add DeepMind Code Contests
faceberg user/catalog add deepmind/code_contests --config default

# Add OpenAI HumanEval
faceberg user/catalog add openai/openai_humaneval --config openai_humaneval

# Add GSM8K
faceberg user/catalog add openai/gsm8k --config main
```

## Step 3: Sync Datasets

Sync the datasets to create Iceberg table metadata and push to HuggingFace:

```bash
faceberg user/catalog sync
```

This downloads dataset metadata, creates Iceberg table files, and automatically pushes them to your HuggingFace repository.

## Step 4: Query with Faceberg

List available tables:

```bash
faceberg user/catalog list
```

Scan tables to see sample data:

```bash
# View first 10 rows of Code Contests
faceberg user/catalog scan default.code_contests --limit 10

# View first 10 rows of HumanEval
faceberg user/catalog scan default.openai_humaneval --limit 10

# View first 10 rows of GSM8K
faceberg user/catalog scan default.gsm8k --limit 10
```

## Step 5: Query with DuckDB

You can query the remote Iceberg tables directly with DuckDB using the REST API deployed to your Space:

```python
import duckdb

# Create DuckDB connection and load Iceberg extension
conn = duckdb.connect()
conn.execute("INSTALL iceberg")
conn.execute("LOAD iceberg")

# Attach the REST catalog from your Space
conn.execute("""
    ATTACH 'https://user-catalog.hf.space' AS faceberg (TYPE ICEBERG)
""")

# Query Code Contests table
result = conn.execute("""
    SELECT name, description, difficulty
    FROM faceberg.default.code_contests
    LIMIT 5
""").fetchall()

print("Code Contests Sample:")
for row in result:
    print(row)

# Query OpenAI HumanEval table
result = conn.execute("""
    SELECT task_id, prompt
    FROM faceberg.default.openai_humaneval
    LIMIT 5
""").fetchall()

print("\nHumanEval Sample:")
for row in result:
    print(row)

# Query GSM8K table
result = conn.execute("""
    SELECT question, answer
    FROM faceberg.default.gsm8k
    LIMIT 5
""").fetchall()

print("\nGSM8K Sample:")
for row in result:
    print(row)
```

Alternatively, you can use the Faceberg CLI's interactive DuckDB shell:

```bash
faceberg user/catalog quack
```

This automatically connects to your Space's REST endpoint and opens an interactive session.

## What's on HuggingFace?

After syncing, your HuggingFace Space contains:

```
user/catalog/
├── Dockerfile             # Space container configuration
├── README.md              # Space documentation with API endpoints
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

The `faceberg.yml` file in your Space contains:

```yaml
uri: hf://spaces/user/catalog

default:
  code_contests:
    dataset: deepmind/code_contests
    revision: <commit-sha>
    config: default
    uri: hf://spaces/user/catalog/default/code_contests/metadata/v1.metadata.json
  openai_humaneval:
    dataset: openai/openai_humaneval
    revision: <commit-sha>
    config: openai_humaneval
    uri: hf://spaces/user/catalog/default/openai_humaneval/metadata/v1.metadata.json
  gsm8k:
    dataset: openai/gsm8k
    revision: <commit-sha>
    config: main
    uri: hf://spaces/user/catalog/default/gsm8k/metadata/v1.metadata.json
```

## Sharing Your Catalog

Since your catalog is hosted on HuggingFace Spaces, anyone can access it through the REST API without authentication:

```python
from pyiceberg.catalog.rest import RestCatalog

# Anyone can connect to your public catalog via REST
catalog = RestCatalog(
    name="faceberg",
    uri="https://user-catalog.hf.space",
)

# Load and query tables
table = catalog.load_table("default.code_contests")
df = table.scan().to_pandas()
```

Alternatively, use Faceberg's RemoteCatalog to work with the Space directly:

```python
from faceberg.catalog import RemoteCatalog

# Load catalog from Space
catalog = RemoteCatalog("user/catalog")
table = catalog.load_table("default.code_contests")
df = table.scan().to_pandas()
```

To make your catalog private, set your HuggingFace Space to private.

## Next Steps

- Explore [QUICKSTART_LOCAL.md](QUICKSTART_LOCAL.md) to learn about local catalogs
- Read the main [README.md](README.md) for programmatic usage and advanced features

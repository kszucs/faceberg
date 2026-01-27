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
faceberg mycatalog scan deepmind.code_contests --limit 10

# View first 10 rows of HumanEval
faceberg mycatalog scan openai.openai_humaneval --limit 10

# View first 10 rows of GSM8K
faceberg mycatalog scan openai.gsm8k --limit 10
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
    FROM iceberg_scan('mycatalog/deepmind/code_contests/metadata/v1.metadata.json')
    LIMIT 5
""").fetchall()

print("Code Contests Sample:")
for row in result:
    print(row)

# Query OpenAI HumanEval table
result = conn.execute("""
    SELECT task_id, prompt
    FROM iceberg_scan('mycatalog/openai/openai_humaneval/metadata/v1.metadata.json')
    LIMIT 5
""").fetchall()

print("\nHumanEval Sample:")
for row in result:
    print(row)

# Query GSM8K table
result = conn.execute("""
    SELECT question, answer
    FROM iceberg_scan('mycatalog/openai/gsm8k/metadata/v1.metadata.json')
    LIMIT 5
""").fetchall()

print("\nGSM8K Sample:")
for row in result:
    print(row)
```

## Step 6: Serve via REST API

You can expose your local catalog via a REST API server that follows the Apache Iceberg REST catalog specification. This allows other Iceberg readers/writers (like Spark, Trino, or PyIceberg REST clients) to connect to your catalog remotely.

### Start the REST Server

```bash
# Start server on default port (8181)
faceberg mycatalog serve

# Custom host and port
faceberg mycatalog serve --host 0.0.0.0 --port 8181

# Development mode with auto-reload
faceberg mycatalog serve --reload

# With URL prefix
faceberg mycatalog serve --prefix my-catalog
```

The server will start and display:
```
Starting REST catalog server...
  Catalog: file:///path/to/mycatalog
  Listening on: http://0.0.0.0:8181
  API docs: http://0.0.0.0:8181/schema
```

### Connect from PyIceberg REST Client

```python
from pyiceberg.catalog.rest import RestCatalog

# Connect to REST catalog
# IMPORTANT: Configure HfFileIO to handle hf:// URIs in data files
catalog = RestCatalog(
    name="faceberg",
    uri="http://localhost:8181",
    **{
        "py-io-impl": "faceberg.catalog.HfFileIO",
    }
)

# List namespaces
namespaces = catalog.list_namespaces()
print(f"Namespaces: {namespaces}")

# List tables
tables = catalog.list_tables("deepmind")
print(f"Tables: {tables}")

# Load a table
table = catalog.load_table("deepmind.code_contests")
print(f"Schema: {table.schema()}")

# Query data (uses HfFileIO to read hf:// URIs)
df = table.scan().to_pandas()
print(df.head())
```

### Connect from DuckDB via REST

> **Note:** DuckDB REST catalog support is still evolving. As of DuckDB 1.4.3, REST catalog configuration may not be fully supported. Check the [DuckDB documentation](https://duckdb.org/docs/extensions/iceberg) for the latest status.

```python
import duckdb

# Connect to REST catalog via DuckDB
# Note: This requires DuckDB >= 1.5.0 with full REST catalog support
con = duckdb.connect()
con.execute("""
    INSTALL iceberg;
    LOAD iceberg;

    CREATE OR REPLACE SECRET iceberg_rest (
        TYPE ICEBERG,
        CATALOG_TYPE REST,
        CATALOG_URL 'http://localhost:8181'
    );
""")

# Query tables via REST
result = con.execute("""
    SELECT * FROM default.code_contests LIMIT 10
""").fetchdf()

print(result)
```

### Connect from Spark

```scala
// Configure Spark to use REST catalog
spark.conf.set("spark.sql.catalog.faceberg", "org.apache.iceberg.spark.SparkCatalog")
spark.conf.set("spark.sql.catalog.faceberg.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
spark.conf.set("spark.sql.catalog.faceberg.uri", "http://localhost:8181")

// Query tables
spark.sql("SELECT * FROM faceberg.default.code_contests LIMIT 10").show()
```

### REST API Endpoints

The server exposes these endpoints following the Iceberg REST spec:

**Catalog Configuration:**
```bash
# Get catalog configuration
curl http://localhost:8181/v1/config
```

**Namespace Operations:**
```bash
# List all namespaces
curl http://localhost:8181/v1/namespaces

# Get namespace properties
curl http://localhost:8181/v1/namespaces/default

# Check if namespace exists (returns 204 or 404)
curl -I http://localhost:8181/v1/namespaces/default
```

**Table Operations:**
```bash
# List tables in namespace
curl http://localhost:8181/v1/namespaces/default/tables

# Load table metadata
curl http://localhost:8181/v1/namespaces/default/tables/code_contests

# Check if table exists (returns 204 or 404)
curl -I http://localhost:8181/v1/namespaces/default/tables/code_contests
```

### API Documentation

The server provides automatic OpenAPI documentation:

- **Swagger UI:** http://localhost:8181/schema/swagger
- **ReDoc:** http://localhost:8181/schema/redoc
- **OpenAPI JSON:** http://localhost:8181/schema/openapi.json

### Serve Remote Catalogs

You can also serve remote catalogs from HuggingFace Hub:

```bash
export HF_TOKEN=your_token_here
faceberg hf://datasets/org/repo serve --port 8181
```

This allows you to expose HuggingFace-hosted catalogs via REST API for team collaboration.

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

- **Remote Catalogs:** Explore [QUICKSTART_REMOTE.md](QUICKSTART_REMOTE.md) to learn about hosting catalogs on HuggingFace Hub
- **CLI Reference:** See [README.md](README.md#cli-reference) for full command documentation
- **Programmatic Usage:** Check the [examples](examples/) directory for Python API examples
- **REST Integration:** Use the REST server to integrate with Spark, Trino, or other Iceberg engines

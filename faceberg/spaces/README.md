---
title: Faceberg REST Catalog
emoji: 🗃️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Faceberg REST Catalog

An Apache Iceberg REST catalog with interactive web interface for browsing and querying Hugging Face datasets using [Faceberg](https://github.com/kszucs/faceberg).

## Features

- **📚 Interactive Browser**: Explore namespaces, tables, and schemas through an intuitive web interface
- **🔍 SQL Query Interface**: Run queries directly in your browser using DuckDB-WASM with full Iceberg support
- **🌐 REST API**: Full Iceberg REST catalog specification at `/v1/*` endpoints
- **🚀 Zero Setup**: No installation required - just visit the Space URL

## Usage

### Web Interface

Visit the Space URL (e.g., `https://your-username-your-space.hf.space`) to:

1. **Browse Catalog**: View all namespaces and tables with detailed metadata
   - Expand namespaces to see tables
   - View table schemas with column names, types, and constraints
   - See row counts, file counts, and HuggingFace dataset links

2. **Query with DuckDB**: Run interactive SQL queries in your browser
   - Click "Query with DuckDB" tab
   - Initialize DuckDB-WASM (loads ~10MB with Iceberg extension)
   - Write SQL queries using `iceberg_scan('metadata_location')`
   - View results in a formatted table

**Example Queries:**
```sql
-- Scan full table (limited)
SELECT * FROM iceberg_scan('metadata_location') LIMIT 100;

-- Filter by partition
SELECT * FROM iceberg_scan('metadata_location')
WHERE split = 'train' LIMIT 10;

-- Aggregate statistics
SELECT split, COUNT(*) as count
FROM iceberg_scan('metadata_location')
GROUP BY split;
```

### REST API

Connect with any Iceberg client:

```python
from pyiceberg.catalog import load_catalog

catalog = load_catalog(
    "rest",
    uri="https://your-username-your-space.hf.space",
)

# List namespaces
namespaces = catalog.list_namespaces()

# Load table
table = catalog.load_table("namespace.table_name")

# Query with DuckDB
import duckdb
duckdb.sql("SELECT * FROM iceberg_scan('table') LIMIT 10").show()
```

## About

Faceberg enables storing Apache Iceberg table metadata directly on Hugging Face Hub as datasets, making your data lake tables easily shareable and version-controlled.

**DuckDB-WASM Integration**: Powered by DuckDB-WASM with native Iceberg and httpfs extensions, enabling full metadata-aware querying directly in your browser without server load.

Learn more:
- [Faceberg on GitHub](https://github.com/kszucs/faceberg)
- [Apache Iceberg](https://iceberg.apache.org/)
- [DuckDB](https://duckdb.org/)

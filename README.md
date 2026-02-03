![Faceberg](https://github.com/kszucs/faceberg/blob/main/faceberg.png?raw=true)

# Faceberg

**Bridge HuggingFace datasets with Apache Iceberg tables — no data copying, just metadata.**

Faceberg maps HuggingFace datasets to Apache Iceberg tables. Your catalog metadata lives on HuggingFace Spaces with an auto-deployed REST API, and any Iceberg-compatible query engine can access the data.

## Installation

```bash
pip install faceberg
```

## Quick Start

```bash
export HF_TOKEN=your_huggingface_token

# Create a catalog on HuggingFace Hub
faceberg user/mycatalog init

# Add datasets
faceberg user/mycatalog add stanfordnlp/imdb --config plain_text
faceberg user/mycatalog add openai/gsm8k --config main

# Query with interactive DuckDB shell
faceberg user/mycatalog quack
```

```sql
SELECT label, substr(text, 1, 100) as preview
FROM iceberg_catalog.stanfordnlp.imdb
LIMIT 10;
```

## How It Works

```
HuggingFace Hub
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌─────────────────────┐    ┌─────────────────────────┐ │
│  │  HF Datasets        │    │  HF Spaces (Catalog)    │ │
│  │  (Original Parquet) │◄───│  • Iceberg metadata     │ │
│  │                     │    │  • REST API endpoint    │ │
│  │  stanfordnlp/imdb/  │    │  • faceberg.yml         │ │
│  │   └── *.parquet     │    │                         │ │
│  └─────────────────────┘    └───────────┬─────────────┘ │
│                                         │               │
└─────────────────────────────────────────┼───────────────┘
                                          │ Iceberg REST API
                                          ▼
                              ┌─────────────────────────┐
                              │     Query Engines       │
                              │  DuckDB, Pandas, Spark  │
                              └─────────────────────────┘
```

**No data is copied** — only metadata is created. Query with DuckDB, PyIceberg, Spark, or any Iceberg-compatible tool.

## Python API

```python
import os
from faceberg import catalog

cat = catalog("user/mycatalog", hf_token=os.environ.get("HF_TOKEN"))
table = cat.load_table("stanfordnlp.imdb")
df = table.scan(limit=100).to_pandas()
```

## Share Your Catalog

Your catalog is accessible to anyone via the REST API:

```python
import duckdb

conn = duckdb.connect()
conn.execute("INSTALL iceberg; LOAD iceberg")
conn.execute("ATTACH 'https://user-mycatalog.hf.space' AS cat (TYPE ICEBERG)")

result = conn.execute("SELECT * FROM cat.stanfordnlp.imdb LIMIT 5").fetchdf()
```

## Documentation

**[Read the docs →](https://faceberg.kszucs.dev/)**

- [Getting Started](https://faceberg.kszucs.dev/) — Full quickstart guide
- [Local Catalogs](https://faceberg.kszucs.dev/local.html) — Use local catalogs for development
- [DuckDB Integration](https://faceberg.kszucs.dev/integrations/duckdb.html) — Advanced SQL queries
- [Pandas Integration](https://faceberg.kszucs.dev/integrations/pandas.html) — Load into DataFrames

## Development

```bash
git clone https://github.com/kszucs/faceberg
cd faceberg
pip install -e .
```

## License

Apache 2.0

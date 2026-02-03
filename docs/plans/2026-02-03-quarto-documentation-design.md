# Quarto Documentation Design

## Overview

Replace existing markdown documentation with a Quarto-based documentation site featuring testable code examples.

## Goals

1. Create user-friendly documentation with clear navigation
2. Make all Python code examples testable via `quarto render`
3. Structure docs to easily add new query engine integrations
4. Explain the core value proposition: mapping HF datasets to Iceberg tables

## Documentation Structure

```
docs/
├── _quarto.yml           # Quarto configuration
├── _variables.yml        # Shared variables (dataset names)
├── index.qmd             # Landing page with overview
├── quickstart.qmd        # Remote catalog workflow (primary)
├── local.qmd             # Local catalogs for testing/development
├── design.qmd            # Architecture + user-focused diagram
└── integrations/
    ├── _metadata.yml     # Shared metadata for section
    ├── duckdb.qmd        # DuckDB integration guide
    └── pandas.qmd        # Pandas integration guide
```

## Navigation

- **Home** - Overview, installation, quick example
- **Quickstart** - Remote catalog workflow (init → add → query)
- **Local Catalogs** - Testing focus, catalog structure
- **Architecture** - How Faceberg works
- **Integrations**
  - DuckDB
  - Pandas
  - *(future: Spark, Polars, etc.)*

## Architecture Diagram

The design page will feature a user-focused diagram showing how everything stays on HuggingFace:

```
┌─────────────────────────────────────────────────────────┐
│                    HuggingFace Hub                       │
│                                                          │
│  ┌─────────────────────┐    ┌─────────────────────────┐ │
│  │  HF Datasets        │    │  HF Spaces (Catalog)    │ │
│  │  (Original Parquet) │◄───│  • Iceberg metadata     │ │
│  │                     │    │  • REST API endpoint    │ │
│  │  org/dataset/       │    │  • faceberg.yml config  │ │
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

Key points:
- Zero infrastructure: Everything hosted on HuggingFace
- No data copying: Metadata references original dataset files via `hf://` URIs
- REST catalog: Space auto-deploys Iceberg REST server
- Standard protocol: Any Iceberg-compatible tool can connect

## Datasets for Examples

Use these well-known datasets consistently across all pages:

1. **stanfordnlp/imdb** - Movie reviews, intuitive for newcomers
2. **Salesforce/wikitext** - Classic NLP benchmark
3. **openai/gsm8k** - Grade school math, highly recognized

## API Usage Pattern

```python
from faceberg import catalog

# Remote catalog
cat = catalog("user/mycatalog", hf_token=os.environ.get("HF_TOKEN"))
cat.init()
cat.add_dataset("default.imdb", "stanfordnlp/imdb", config="plain_text")
table = cat.load_table("default.imdb")
df = table.scan(limit=5).to_pandas()
```

## Testable Code Strategy

Quarto configuration for executable Python:

```yaml
execute:
  enabled: true
  cache: true
  freeze: auto
```

Each page will use setup blocks to create isolated test environments:

```python
#| label: setup
#| include: false
import tempfile
import os
os.chdir(tempfile.mkdtemp())
```

## Migration Plan

### Remove
- `QUICKSTART_LOCAL.md`
- `QUICKSTART_REMOTE.md`
- `docs/*.md` (except `docs/plans/`)

### Keep
- `README.md` (update links to new docs)
- `ARCHITECTURE.md` (detailed technical reference for contributors)
- `docs/plans/*.md` (internal design documents)

### Content to Preserve
- CLI command sequences (init, add, sync, list, scan)
- REST server setup (`serve` command)
- `quack` command usage
- Catalog directory structure explanation
- `faceberg.yml` configuration examples

## Page Content Details

### index.qmd
- Hero: "Bridge HuggingFace datasets with Apache Iceberg"
- Installation: `pip install faceberg`
- 10-line quickstart example
- Links to detailed guides

### quickstart.qmd
- Prerequisites (HF token)
- Step-by-step: init → add → sync → query
- Using `quack` for interactive queries
- Link to deployed HF Space

### local.qmd
- When to use local catalogs
- Same workflow with local paths
- Directory structure explanation
- REST server for local development

### design.qmd
- Simplified architecture diagram (Mermaid)
- Core concepts explained
- Link to ARCHITECTURE.md for details

### integrations/duckdb.qmd
- Direct iceberg_scan() queries
- REST catalog attachment
- Using `quack` helper command

### integrations/pandas.qmd
- Loading tables to DataFrames
- Working with table scans
- Memory considerations

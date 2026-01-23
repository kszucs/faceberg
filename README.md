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

1. Create a `faceberg.yml` config file:

```yaml
catalog:
  name: my_catalog
  location: .faceberg/

datasets:
  - name: dataset1
    repo: kszucs/dataset1
  - name: dataset2
    repo: kszucs/dataset2
```

2. Initialize the catalog:

```bash
faceberg init faceberg.yml
```

3. Create Iceberg tables:

```bash
faceberg create
```

4. Query your data:

```python
from faceberg.catalog import JsonCatalog

catalog = JsonCatalog("my_catalog", ".faceberg/")
table = catalog.load_table("default.dataset1_default")
df = table.scan().to_pandas()
```

## Commands

- `faceberg init` - Initialize catalog from config
- `faceberg create` - Create Iceberg tables for datasets
- `faceberg sync` - Sync metadata with dataset updates
- `faceberg list` - List all tables
- `faceberg info <table>` - Show table information
- `faceberg push` - Push catalog to HuggingFace
- `faceberg pull` - Pull catalog from HuggingFace

## Features

- **Config-driven**: Define your datasets in `faceberg.yml`
- **Local-first**: Test everything locally before pushing to HF
- **Dataset-aware**: Handles multiple configs and splits
- **hf:// protocol**: Leverages HF XET storage for efficiency

## License

MIT

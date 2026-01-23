# Faceberg Architecture

This document describes the architecture of Faceberg, focusing on how it bridges HuggingFace datasets to Apache Iceberg tables.

## Overview

Faceberg provides a clean abstraction layer that converts HuggingFace dataset metadata into Iceberg table metadata. The architecture is designed around three main phases:

1. **Discovery**: Find and understand HuggingFace dataset structure
2. **Bridging**: Transform dataset metadata into Iceberg-ready format (TableInfo)
3. **Conversion**: Create Iceberg metadata files that reference the original data

## Module Structure

```
faceberg/
├── bridge.py      # Discovery + schema conversion → TableInfo
├── convert.py     # TableInfo → Iceberg metadata files
├── catalog.py     # Iceberg catalog implementation
├── config.py      # Configuration management
└── cli.py         # Command-line interface
```

## Core Abstractions

### 1. DatasetInfo (Discovery Layer)

**Location**: `faceberg/bridge.py`

**Purpose**: Represents the structure of a HuggingFace dataset after discovery.

**Key attributes**:
- `repo_id`: HuggingFace repository ID
- `configs`: List of available configurations
- `splits`: Dict mapping config → list of splits
- `parquet_files`: Nested dict: config → split → list of file paths

**Key methods**:
- `discover(repo_id, configs, token)`: Discovers dataset structure from HuggingFace Hub
- `to_table_infos(namespace, table_name_prefix, token)`: **Bridge method** that converts to TableInfo objects

**Example**:
```python
# Discover a dataset
dataset_info = DatasetInfo.discover("stanfordnlp/imdb")

# Convert to TableInfo objects (one per config)
table_infos = dataset_info.to_table_infos(
    namespace="default",
    table_name_prefix="imdb",
)
```

### 2. TableInfo (Bridge Output)

**Location**: `faceberg/bridge.py`

**Purpose**: Bridge between HuggingFace datasets and Apache Iceberg. Contains all metadata needed to create an Iceberg table.

**Key attributes**:
- `namespace`: Iceberg namespace (e.g., "default")
- `table_name`: Table name (e.g., "imdb_plain_text")
- `schema`: Iceberg Schema with field IDs
- `partition_spec`: PartitionSpec (partitioned by split)
- `files`: List of FileInfo objects with file metadata
- `source_repo`: HuggingFace repo ID (for traceability)
- `source_config`: Dataset configuration name

**Key properties**:
- `identifier`: Returns "namespace.table_name"
- `total_rows`: Sum of rows across all files
- `total_size`: Sum of bytes across all files
- `get_table_properties()`: Returns Iceberg table properties including source metadata

**Example**:
```python
table_info = table_infos[0]
print(f"Table: {table_info.identifier}")
print(f"Schema: {len(table_info.schema.fields)} fields")
print(f"Files: {len(table_info.files)} files")
print(f"Total rows: {table_info.total_rows:,}")
```

### 3. FileInfo (Data File Abstraction)

**Location**: `faceberg/bridge.py`

**Purpose**: Represents a single data file in an Iceberg table.

**Key attributes**:
- `path`: Full hf:// URI to the file
- `size_bytes`: File size in bytes
- `row_count`: Number of rows in the file
- `split`: Split name (train, test, validation, etc.)

**Example**:
```python
file_info = FileInfo(
    path="hf://datasets/stanfordnlp/imdb/plain_text/train-00000.parquet",
    size_bytes=0,  # Enriched during metadata creation
    row_count=0,   # Enriched during metadata creation
    split="train",
)
```

### 4. IcebergMetadataWriter (Conversion Layer)

**Location**: `faceberg/convert.py`

**Purpose**: Writes Iceberg metadata files in metadata-only mode. Creates metadata that references existing HuggingFace files without copying or modifying them.

**Key methods**:
- `create_metadata_from_files(file_infos, table_uuid, properties)`: Creates all Iceberg metadata files

**What it creates**:
1. Manifest file (`.avro`) - Contains list of data files with statistics
2. Manifest list (`.avro`) - Points to manifest files
3. Table metadata (`v1.metadata.json`) - Complete table metadata
4. Version hint (`version-hint.text`) - Current metadata version

**Example**:
```python
writer = IcebergMetadataWriter(
    table_path=Path("/path/to/table"),
    schema=table_info.schema,
    partition_spec=table_info.partition_spec,
)

metadata_location = writer.create_metadata_from_files(
    file_infos=table_info.files,
    table_uuid=str(uuid.uuid4()),
    properties=table_info.get_table_properties(),
)
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     HuggingFace Dataset                         │
│              (stanfordnlp/imdb on HF Hub)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ DatasetInfo.discover()
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                        DatasetInfo                              │
│  • repo_id: "stanfordnlp/imdb"                                 │
│  • configs: ["plain_text"]                                     │
│  • splits: {"plain_text": ["train", "test", "unsupervised"]}  │
│  • parquet_files: {"plain_text": {"train": [...], ...}}       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ .to_table_infos()
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                        TableInfo                                │
│  • identifier: "default.imdb_plain_text"                       │
│  • schema: Schema (with field IDs)                             │
│  • partition_spec: PartitionSpec (by split)                    │
│  • files: [FileInfo, FileInfo, ...]                            │
│  • source_repo: "stanfordnlp/imdb"                             │
│  • source_config: "plain_text"                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ IcebergMetadataWriter
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Iceberg Metadata Files                       │
│  • v1.metadata.json - Table metadata                           │
│  • manifest.avro - List of data files                          │
│  • manifest-list.avro - Points to manifests                    │
│  • version-hint.text - Current version                         │
└─────────────────────────────────────────────────────────────────┘
```

## Schema Conversion Pipeline

The schema conversion happens in `faceberg/bridge.py`:

```
HuggingFace Features
        ↓
PyArrow Schema (.arrow_schema)
        ↓
PyIceberg Schema without IDs (_pyarrow_to_schema_without_ids)
        ↓
PyIceberg Schema with unique IDs (assign_fresh_schema_ids)
        ↓
Add 'split' field if requested
        ↓
Final Iceberg Schema
```

This ensures:
- Globally unique field IDs across nested structures
- Type compatibility between HF datasets and Iceberg
- Proper handling of complex types (nested structures, lists, etc.)

## Catalog Integration

The `FacebergCatalog` class integrates everything:

```python
class FacebergCatalog(JsonCatalog):
    def create_tables(self, token, table_name):
        # 1. Discover datasets
        dataset_info = DatasetInfo.discover(repo_id, configs, token)

        # 2. Convert to TableInfo objects
        table_infos = dataset_info.to_table_infos(...)

        # 3. Create Iceberg metadata for each table
        for table_info in table_infos:
            self._create_table_from_table_info(table_info)
```

## Metadata-Only Mode

Faceberg operates in **metadata-only mode** by default:

- **No data copying**: Original HuggingFace Parquet files are referenced via `hf://` URIs
- **Metadata creation**: Only Iceberg metadata files are written locally
- **Remote data access**: Iceberg readers can access data directly from HuggingFace Hub
- **Fast setup**: No need to download or transform large datasets

### Benefits

1. **Storage efficiency**: No data duplication
2. **Fast table creation**: Only metadata is written
3. **HuggingFace integration**: Seamless access to HF datasets
4. **Iceberg compatibility**: Works with any Iceberg reader (DuckDB, Spark, etc.)

### File References

Iceberg metadata references files using `hf://` URIs:
```
hf://datasets/stanfordnlp/imdb/plain_text/train-00000.parquet
```

These URIs are understood by HuggingFace's filesystem implementation, allowing direct data access.

## Partitioning

Tables are partitioned by split (train/test/validation):

- **Partition field**: `split` (first column in schema)
- **Transform**: Identity (split value stored in metadata)
- **Pruning**: Iceberg readers can filter by split efficiently

Example partition spec:
```python
PartitionSpec(
    PartitionField(
        source_id=1,  # split field ID
        field_id=1000,
        transform=IdentityTransform(),
        name="split",
    )
)
```

## Testing

The architecture is fully tested:

- **Unit tests**: Each abstraction tested independently
- **Integration tests**: End-to-end table creation with real datasets
- **Demo script**: `demo_table_info.py` shows the complete flow

Run tests:
```bash
pytest faceberg/tests/
```

Run demo:
```bash
python demo_table_info.py
```

## Future Enhancements

The architecture supports future conversion modes:

1. **COPY_DATA mode**: Copy Parquet files locally without modification
2. **REWRITE_DATA mode**: Rewrite with content-defined chunking and page indexes

These modes can be added by implementing new writer classes that extend the base `DataWriter` pattern.

## Key Design Principles

1. **Separation of concerns**: Discovery, conversion, and manifestation are independent
2. **Immutable data**: Original HuggingFace files are never modified
3. **Type safety**: Strong typing with dataclasses and PyIceberg types
4. **Testability**: Each component can be tested independently
5. **Extensibility**: Easy to add new conversion modes or catalog backends

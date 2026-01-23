# Moving FileInfo and TableInfo to bridge.py

## Summary

Moved `FileInfo` and `TableInfo` classes from `convert.py` to `bridge.py` to better reflect their role as **outputs of the bridge layer**.

## Rationale

The bridge layer's job is to:
1. **Discover** HuggingFace datasets (DatasetInfo)
2. **Convert** schemas (build_iceberg_schema_from_features)
3. **Create** TableInfo objects (output)

`TableInfo` and `FileInfo` are the **final outputs** of the bridge layer - they represent a HuggingFace dataset that has been fully transformed into Iceberg-ready metadata. They logically belong in `bridge.py`.

The `convert.py` module only needs to **consume** these outputs (specifically `FileInfo`) to write Iceberg metadata files. It doesn't create or modify them.

## Changes Made

### 1. Moved Classes to bridge.py

**Moved from `convert.py` to `bridge.py`:**
- `FileInfo` - Data file metadata
- `TableInfo` - Complete table metadata (bridge output)
- `build_split_partition_spec()` - Helper function used during TableInfo creation

### 2. Updated Imports

**bridge.py:**
```python
# Now defines FileInfo, TableInfo, and build_split_partition_spec
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.transforms import IdentityTransform
```

**convert.py:**
```python
# Now imports FileInfo from bridge (TableInfo not needed here)
from faceberg.bridge import FileInfo
```

**catalog.py:**
```python
# Updated to import from bridge
from faceberg.bridge import TableInfo
```

**__init__.py:**
```python
# Updated exports
from faceberg.bridge import (
    DatasetInfo,
    FileInfo,
    TableInfo,
    build_iceberg_schema_from_features,
    build_split_partition_spec,
    infer_schema_from_dataset,
)
from faceberg.convert import IcebergMetadataWriter
```

### 3. Updated Documentation

- **ARCHITECTURE.md**: Updated locations for FileInfo and TableInfo
- **All docstrings**: Maintained accurate descriptions

## Module Responsibilities (After Change)

### bridge.py
**Purpose**: Bridge between HuggingFace and Iceberg formats

**Inputs**: HuggingFace dataset (repo_id, configs, splits)

**Outputs**:
- `DatasetInfo` - Discovery results
- `TableInfo` - Complete Iceberg table metadata
- `FileInfo` - Individual file metadata

**Functions**:
- Discovery: `DatasetInfo.discover()`, `DatasetInfo.to_table_infos()`
- Schema conversion: `build_iceberg_schema_from_features()`, `infer_schema_from_dataset()`
- Partitioning: `build_split_partition_spec()`

### convert.py
**Purpose**: Convert TableInfo to physical Iceberg metadata files

**Inputs**:
- `List[FileInfo]` - File metadata from TableInfo

**Outputs**:
- Iceberg metadata files (manifest, manifest list, table metadata, version hint)

**Classes**:
- `IcebergMetadataWriter` - Writes metadata in metadata-only mode

## Data Flow

```
HuggingFace Dataset
        ↓
    bridge.py
        ↓
    DatasetInfo (discovery)
        ↓
    TableInfo (bridge output)
        ├── schema: Schema
        ├── partition_spec: PartitionSpec
        ├── files: List[FileInfo]
        └── source metadata
        ↓
    convert.py
        ↓
    IcebergMetadataWriter
        ↓
Iceberg Metadata Files
```

## Benefits

1. **Clearer ownership**: TableInfo is clearly the bridge's output
2. **Better modularity**: convert.py is simpler (only needs FileInfo)
3. **Logical grouping**: All bridge outputs in one place
4. **Reduced coupling**: convert.py doesn't depend on TableInfo type

## Test Results

All tests pass after the change:
```
31 passed, 5 skipped in 15.12s
```

Package-level imports still work:
```python
from faceberg import FileInfo, TableInfo, IcebergMetadataWriter
```

## Files Modified

- [faceberg/bridge.py](faceberg/bridge.py) - Added FileInfo, TableInfo, build_split_partition_spec
- [faceberg/convert.py](faceberg/convert.py) - Removed classes, imports FileInfo from bridge
- [faceberg/catalog.py](faceberg/catalog.py) - Updated TableInfo import
- [faceberg/__init__.py](faceberg/__init__.py) - Updated exports
- [ARCHITECTURE.md](ARCHITECTURE.md) - Updated documentation

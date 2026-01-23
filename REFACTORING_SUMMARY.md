# Refactoring Summary: Bridge and Convert Modules

## Overview

The codebase has been refactored to better reflect the two-phase architecture:

1. **Bridge Phase** (`bridge.py`): Discovers HuggingFace datasets and creates TableInfo objects
2. **Convert Phase** (`convert.py`): Converts TableInfo objects into Iceberg metadata files

## Module Changes

### Before
```
faceberg/
├── discovery.py    # Dataset discovery
├── schema.py       # Schema conversion
├── table.py        # TableInfo, FileInfo, IcebergMetadataWriter
├── catalog.py
├── config.py
└── cli.py
```

### After
```
faceberg/
├── bridge.py       # Discovery + schema → TableInfo (merged discovery.py + schema.py)
├── convert.py      # TableInfo → Iceberg metadata (renamed from table.py)
├── catalog.py      # (imports updated)
├── config.py       # (unchanged)
└── cli.py          # (unchanged)
```

## Rationale

### Why `bridge.py`?

The name "bridge" emphasizes that this module serves as the **bridge between two formats**:
- **Input**: HuggingFace dataset structure (configs, splits, Parquet files)
- **Output**: TableInfo objects with Iceberg metadata

By merging `discovery.py` and `schema.py`, we consolidate all HuggingFace-to-Iceberg conversion logic:
- Dataset discovery (configs, splits, files)
- Schema conversion (HF Features → Iceberg Schema)
- TableInfo creation (the bridge output)

### Why `convert.py`?

The name "convert" emphasizes that this module **converts TableInfo to actual files**:
- **Input**: TableInfo objects (from bridge.py)
- **Output**: Iceberg metadata files on disk

This module is responsible for the **metadata-only conversion mode**, creating:
- Manifest files (`.avro`)
- Manifest lists (`.avro`)
- Table metadata (`v1.metadata.json`)
- Version hints (`version-hint.text`)

## What Changed

### 1. File Renames
- `discovery.py` → `bridge.py` (merged with schema.py)
- `schema.py` → merged into `bridge.py`
- `table.py` → `convert.py`

### 2. Test Renames
- `test_discovery.py` → `test_bridge.py`
- `test_schema.py` → (unchanged, but imports from bridge)

### 3. Import Updates

**Before:**
```python
from faceberg.discovery import DatasetInfo
from faceberg.schema import build_iceberg_schema_from_features
from faceberg.table import TableInfo, IcebergMetadataWriter
```

**After:**
```python
from faceberg.bridge import DatasetInfo, build_iceberg_schema_from_features
from faceberg.convert import TableInfo, IcebergMetadataWriter
```

### 4. Updated Files
- `faceberg/__init__.py` - Updated exports
- `faceberg/catalog.py` - Updated imports
- `faceberg/tests/test_bridge.py` - Renamed and updated
- `faceberg/tests/test_schema.py` - Updated imports
- `faceberg/tests/test_faceberg_catalog.py` - Updated imports
- `demo_table_info.py` - Updated imports
- `ARCHITECTURE.md` - Updated documentation

## Benefits

1. **Clearer architecture**: Two distinct phases with clear responsibilities
2. **Better names**: "bridge" and "convert" are more descriptive than "discovery/schema/table"
3. **Consolidated logic**: All HF-to-Iceberg conversion logic in one place (bridge.py)
4. **Easier to understand**: The flow is more obvious: bridge → convert
5. **Future-proof**: Easy to add more conversion modes (COPY_DATA, REWRITE_DATA)

## Data Flow

```
┌─────────────────────────────────────────┐
│      HuggingFace Dataset (HF Hub)       │
└──────────────────┬──────────────────────┘
                   │
                   │ bridge.py
                   │ ├─ DatasetInfo.discover()
                   │ ├─ build_iceberg_schema_from_features()
                   │ └─ DatasetInfo.to_table_infos()
                   ↓
┌─────────────────────────────────────────┐
│         TableInfo (Bridge Output)       │
│  • Iceberg schema with field IDs       │
│  • Partition spec (by split)           │
│  • List of FileInfo objects             │
│  • Source metadata (repo, config)       │
└──────────────────┬──────────────────────┘
                   │
                   │ convert.py
                   │ └─ IcebergMetadataWriter
                   ↓
┌─────────────────────────────────────────┐
│     Iceberg Metadata Files (Disk)      │
│  • v1.metadata.json                     │
│  • manifest.avro                        │
│  • manifest-list.avro                   │
│  • version-hint.text                    │
└─────────────────────────────────────────┘
```

## Testing

All tests pass after refactoring:
```bash
$ pytest faceberg/tests/ -v
31 passed, 5 skipped in 14.71s
```

Demo script works correctly:
```bash
$ python demo_table_info.py
✓ Discovers dataset from HF Hub
✓ Converts to TableInfo objects
✓ Creates Iceberg metadata files
```

## API Compatibility

The public API remains the same - only module names changed:

```python
# Still works (imports from correct modules automatically)
from faceberg import (
    DatasetInfo,
    TableInfo,
    IcebergMetadataWriter,
    build_iceberg_schema_from_features,
)
```

Users importing from the package level (`from faceberg import ...`) see no changes.
Only direct module imports need updating (`from faceberg.discovery import ...` → `from faceberg.bridge import ...`).

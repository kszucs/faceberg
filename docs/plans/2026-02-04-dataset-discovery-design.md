# Dataset Discovery Module Design

**Date:** 2026-02-04
**Status:** Approved

## Overview

Create a new `discover.py` module focused solely on discovering HuggingFace datasets. This module will extract the discovery functionality from `bridge.py` without any Iceberg-specific logic. It will eventually replace parts of `bridge.py` and `convert.py`.

## Goals

- Pure discovery: gather dataset information without Iceberg conversions
- Simple, focused API with a single function entry point
- Self-contained ParquetFile representation with all necessary metadata
- Comprehensive test coverage

## Public API

### ParquetFile

```python
@dataclass
class ParquetFile:
    uri: str              # Full hf:// URI with revision
    size: int             # File size in bytes
    blob_id: str          # Git blob ID (oid) from HuggingFace
    split: Optional[str]  # Split name (train, test, validation, etc.)
```

Moved from `iceberg.py` and enhanced with:
- `split` field for dataset split information
- Renamed `hash` to `blob_id` for clarity

### DatasetInfo

```python
@dataclass
class DatasetInfo:
    repo_id: str          # e.g., "squad"
    config: str           # Configuration name
    revision: str         # Git SHA
    features: Features    # HuggingFace Features object
    splits: List[str]     # List of split names
    data_dir: str         # Common data directory path
    files: List[ParquetFile]  # All discovered files (flat list)
```

Simplified from `bridge.py` version:
- Flat list of files instead of dict mapping splits to files
- Each file contains its own split information
- No Iceberg-specific fields

### discover_dataset()

```python
def discover_dataset(
    repo_id: str,
    config: str,
    token: Optional[str] = None,
) -> DatasetInfo:
    """Discover structure and files in a HuggingFace dataset.

    Args:
        repo_id: HuggingFace dataset repository ID (e.g., "squad")
        config: Configuration name to discover
        token: HuggingFace API token (uses HF_TOKEN env var if not provided)

    Returns:
        DatasetInfo with all files for the latest revision

    Raises:
        ValueError: If dataset not found, config doesn't exist, or metadata inconsistent
    """
```

Key simplifications from `bridge.py`:
- No `since_revision` parameter (no incremental discovery)
- No `to_table_info()` method (no Iceberg conversion)
- Single responsibility: discover and return information

## Discovery Process

### Step 1: Load Dataset Builder

```python
builder = dataset_builder_safe(repo_id, config=config, token=token)
revision = builder.hash
features = builder.info.features
```

Use the temporary directory workaround to avoid picking up local files.

### Step 2: Fetch File Metadata from HuggingFace Hub

```python
api = HfApi(token=token)
dataset_info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
file_metadata = {s.rfilename: s for s in dataset_info.siblings}
```

This gives us access to file sizes and blob IDs.

### Step 3: Process Data Files

```python
files = []
for split, file_uris in builder.config.data_files.items():
    for uri in file_uris:
        # Extract path from URI
        path = extract_path_from_uri(uri)

        # Get metadata (strict - fail if not found)
        if path not in file_metadata:
            raise ValueError(f"File {uri} not found in Hub API response")

        metadata = file_metadata[path]

        # Create ParquetFile
        files.append(ParquetFile(
            uri=uri,
            size=metadata.size,
            blob_id=metadata.oid,
            split=split,
        ))
```

### Step 4: Extract Common Data Directory

```python
try:
    file_paths = [extract_path_from_uri(f.uri) for f in files]
    data_dir = os.path.commonpath(file_paths) if file_paths else ""
except ValueError as e:
    raise ValueError(f"Unable to determine common data directory: {e}") from e
```

### Step 5: Return DatasetInfo

```python
return DatasetInfo(
    repo_id=repo_id,
    config=config,
    revision=revision,
    features=features,
    splits=list(builder.config.data_files.keys()),
    data_dir=data_dir,
    files=files,
)
```

## Error Handling

### Invalid Dataset/Config (ValueError)

```python
try:
    builder = dataset_builder_safe(repo_id, config=config, token=token)
except Exception as e:
    raise ValueError(
        f"Dataset {repo_id} config {config} not found or not accessible: {e}"
    ) from e
```

### Missing File Metadata (ValueError - Strict)

If a file from the builder isn't in the Hub API response, fail immediately:

```python
if path not in file_metadata:
    raise ValueError(
        f"File {uri} from dataset builder not found in Hub API response. "
        f"This may indicate an inconsistent dataset state."
    )
```

### Empty Data Directory (ValueError)

```python
try:
    data_dir = os.path.commonpath(file_paths)
except ValueError as e:
    raise ValueError(
        f"Unable to determine common data directory from files: {file_paths}"
    ) from e
```

### No Files Found (Valid)

Empty dataset is a valid state - return `DatasetInfo` with:
- `files = []`
- `data_dir = ""`
- `splits = []`

## Testing Strategy

### Test File: test_discover.py

**Unit Tests with Mocked APIs:**

1. **test_discover_dataset_success** - Multiple splits, multiple files per split
2. **test_discover_dataset_single_split** - Single split dataset
3. **test_discover_dataset_empty** - No files (valid state)
4. **test_discover_dataset_missing_file_metadata** - Should raise ValueError
5. **test_discover_dataset_invalid_dataset** - Should raise ValueError
6. **test_discover_dataset_invalid_config** - Should raise ValueError
7. **test_common_path_extraction** - Various path structures
8. **test_blob_id_population** - Verify blob_id correctly extracted from oid

**Integration Test (Optional):**

9. **test_discover_real_dataset** - Use small stable dataset (skip if no HF_TOKEN)

**Testing Practices:**
- Use pytest fixtures (`tmp_path`) for file operations
- Simple assertions (`assert x == y`)
- Test public API only (`discover_dataset()`)
- Single test file unless it grows large
- Realistic mock data matching HuggingFace API structure

## Module Structure

```
discover.py (~150-200 lines)
├── Imports
├── ParquetFile (dataclass)
├── DatasetInfo (dataclass)
├── dataset_builder_safe() (helper - copied from bridge.py)
└── discover_dataset() (public API)
```

## Dependencies

- `datasets` - load_dataset_builder, Features
- `huggingface_hub` - HfApi
- `os`, `tempfile` - path handling
- `dataclasses`, `typing` - standard library

## Implementation Notes

1. Copy `dataset_builder_safe()` from `bridge.py` to keep module self-contained
2. Extract path from URI helper may be needed (parse hf:// URIs)
3. Keep implementation simple - no optimizations until needed
4. Module will eventually replace discovery parts of `bridge.py` and `convert.py`

## Future Work (Out of Scope)

- Incremental discovery (comparing revisions) - belongs at higher level
- Iceberg schema conversion - stays in separate module
- File content validation - not needed for discovery
- Caching - add if performance becomes an issue

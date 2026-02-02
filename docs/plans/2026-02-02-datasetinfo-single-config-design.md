# DatasetInfo Single Config Simplification

**Date:** 2026-02-02
**Status:** Approved

## Overview

Simplify `DatasetInfo` to track a single config instead of multiple configs by requiring a `config: str` argument at construction time. This eliminates unnecessary dictionary-based tracking and simplifies the API.

## Current State

`DatasetInfo` currently:
- Discovers multiple configs at once (stores them in `configs: List[str]`)
- Tracks splits, parquet files, and data directories per config using dictionaries
- Requires passing config to methods like `to_table_info()` even though the config is already known

## Proposed Changes

### 1. API Changes

**Constructor:**
```python
# Before
DatasetInfo.discover(repo_id="stanfordnlp/imdb", configs=["plain_text"], token=token)

# After
DatasetInfo.discover(repo_id="stanfordnlp/imdb", config="plain_text", token=token)
```

**Method calls (config parameter removed):**
```python
# Before
table_info = dataset_info.to_table_info(namespace, table_name, config="plain_text", token=token)

# After
table_info = dataset_info.to_table_info(namespace, table_name, token=token)
```

### 2. Simplified Data Model

```python
@dataclass
class DatasetInfo:
    repo_id: str
    config: str                              # NEW: single config
    splits: List[str]                        # CHANGED: was Dict[str, List[str]]
    parquet_files: Dict[str, List[str]]     # CHANGED: was Dict[str, Dict[str, List[str]]]
    data_dir: str                            # CHANGED: was Dict[str, str]
    revision: Optional[str] = None
```

### 3. Simplified Methods

The following methods no longer need a `config` parameter:
- `to_table_info(namespace, table_name, token)`
- `to_table_info_incremental(namespace, table_name, old_revision, token)`
- `get_parquet_files_for_table()`
- `get_sample_parquet_file()`
- `discover_file_pattern()`

## Implementation Approach

### Core Changes

1. **`discover()` method:**
   - Change signature: `discover(cls, repo_id: str, config: str, token: Optional[str] = None)`
   - Remove the loop over multiple configs
   - Call `_discover_config()` once for the single config
   - Return simplified DatasetInfo directly

2. **`_discover_config()` method:**
   - No changes needed (already handles single config)
   - Returns `(splits, files, revision, data_dir)` which maps directly to new attributes

3. **Method body simplifications:**
   - Remove config parameter from all method signatures
   - Remove `if config not in self.configs` validation logic
   - Access `self.splits`, `self.parquet_files[split]`, `self.data_dir` directly

### Caller Updates

Two call sites in `catalog.py`:

```python
# Lines 987 and 1143: Change from
dataset_info = DatasetInfo.discover(repo_id=repo, configs=[config], token=token)

# To
dataset_info = DatasetInfo.discover(repo_id=repo, config=config, token=token)

# Lines 999 and 1160: Change from
table_info = dataset_info.to_table_info(..., config=table_entry.config, ...)

# To
table_info = dataset_info.to_table_info(..., ...)
```

## Testing Strategy

### Test Updates

All tests in `faceberg/tests/test_bridge.py` need updates:

1. **Update `discover()` calls:**
   ```python
   # Before
   dataset_info = DatasetInfo.discover("stanfordnlp/imdb", configs=["plain_text"])

   # After
   dataset_info = DatasetInfo.discover("stanfordnlp/imdb", config="plain_text")
   ```

2. **Update attribute assertions:**
   ```python
   # Before
   assert "plain_text" in dataset_info.configs
   assert "train" in dataset_info.splits["plain_text"]

   # After
   assert dataset_info.config == "plain_text"
   assert "train" in dataset_info.splits
   ```

3. **Update method calls:**
   ```python
   # Before
   files = dataset_info.get_parquet_files_for_table("plain_text")

   # After
   files = dataset_info.get_parquet_files_for_table()
   ```

### Coverage Goals

- Single config discovery works correctly
- Invalid config raises appropriate error
- All simplified methods work without config parameter
- Integration with catalog.py works end-to-end

## Migration Notes

This is a breaking API change. No backward compatibility will be maintained.

External code using `DatasetInfo.discover()` must update to:
- Pass `config` instead of `configs`
- Remove config parameters from method calls

Based on code analysis, usage is primarily internal (catalog.py) and tests.

## Benefits

- **Simpler data model:** No nested dictionaries for multi-config tracking
- **Clearer API:** Config is specified once at construction time
- **Less redundancy:** No need to pass config to every method
- **Easier to understand:** One DatasetInfo = one config, clear mental model

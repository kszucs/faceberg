# Catalog Refactoring Design

**Date:** 2026-01-28
**Status:** Approved

## Overview

Refactor the catalog implementation to:
1. Remove `_config` instance variable - `_load_config()` returns Config directly
2. Use `Config` and `Entry` instead of `cfg.Config` and `cfg.Entry`
3. Keep identifiers as `Identifier` tuples throughout, only unpack for file operations
4. Add `_staging_add()` and `_staging_delete()` helper methods
5. Simplify code with strict error handling, no backward compatibility

## Design Details

### 1. Config Management Pattern

**Current:** `_config` stored as instance variable, loaded once and reused.

**New:** `_load_config()` returns Config without storing it.

```python
def _load_config(self) -> Config:
    """Load and return config without storing it."""
    return Config.from_yaml(catalog_file)
```

**Usage pattern:**
```python
# OLD:
self._load_config()  # Stores in self._config
with self._staging_changes():
    if self._has_table(namespace, table_name):
        ...
    self._save_config()

# NEW:
with self._staging_changes():
    config = self._load_config()
    if identifier in config:
        ...
    self._save_config(config)
```

**Benefits:**
- No stale state - always fresh config
- Explicit data flow - config passed where needed
- Simpler reasoning - no hidden mutations

### 2. Identifier Handling

**Current:** Immediately unpack identifiers to `(namespace, table_name)` strings.

**New:** Keep as `Identifier` tuples throughout, only unpack for file paths.

```python
def _normalize_identifier(self, identifier: Union[str, Identifier]) -> Identifier:
    """Convert string or tuple to validated Identifier tuple."""
    if isinstance(identifier, str):
        parts = identifier.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid identifier: {identifier}. Expected 'namespace.table'")
        return tuple(parts)
    return tuple(identifier)

def _identifier_to_path(self, identifier: Identifier) -> Path:
    """Convert identifier to relative path for file operations."""
    return Path(*identifier)  # namespace/table_name
```

**Usage:**
```python
# Keep as Identifier throughout:
identifier = self._normalize_identifier(identifier)
config = self._load_config()

if identifier in config:
    entry = config[identifier]

# Only unpack for file paths:
table_path = self._staging_dir / self._identifier_to_path(identifier)
```

**Benefits:**
- Type-safe with Identifier throughout
- Unpacking isolated to path operations
- Aligned with PyIceberg conventions

### 3. Staging Operations

**Current:** Direct manipulation of `_staged_changes` list.

**New:** Dedicated helper methods with validation.

```python
def _staging_add(self, path_in_repo: Union[str, Path], path_or_fileobj: Union[str, Path]) -> None:
    """Record a file addition in staged changes."""
    if self._staged_changes is None:
        raise RuntimeError("_staging_add() must be called within _staging_changes() context")

    self._staged_changes.append(
        CommitOperationAdd(
            path_in_repo=str(path_in_repo),
            path_or_fileobj=str(path_or_fileobj)
        )
    )

def _staging_delete(self, path_in_repo: Union[str, Path]) -> None:
    """Record a file/directory deletion in staged changes."""
    if self._staged_changes is None:
        raise RuntimeError("_staging_delete() must be called within _staging_changes() context")

    self._staged_changes.append(
        CommitOperationDelete(path_in_repo=str(path_in_repo))
    )
```

**Usage:**
```python
# OLD:
self._staged_changes.append(
    CommitOperationAdd(path_in_repo=str(rel_path), path_or_fileobj=str(file_path))
)

# NEW:
self._staging_add(rel_path, file_path)
self._staging_delete(table_dir)
```

**Benefits:**
- Single point of validation
- Consistent string conversion
- Easier to extend later
- Cleaner call sites

### 4. Import and Helper Method Changes

**Imports:**
```python
# OLD:
from faceberg import config as cfg

# NEW:
from faceberg.config import Config, Entry
```

**Remove helper methods:**

Remove `_has_table`, `_get_table`, `_set_table`, `_delete_table`, `_has_namespace`, `_create_namespace`, `_get_namespace_tables`, `_delete_namespace`.

Use Config's Mapping interface directly:

```python
# Direct usage:
identifier = self._normalize_identifier(identifier)
config = self._load_config()

if identifier in config:  # Direct __contains__
    entry = config[identifier]  # Direct __getitem__

config[identifier] = Entry(...)  # Direct __setitem__
del config[identifier]  # Direct __delitem__
```

**Benefits:**
- Fewer indirection layers
- Config already supports Identifier keys
- Less code to maintain
- More Pythonic

### 5. Strict Error Handling & Simplifications

**Error handling philosophy:**
```python
# OLD - Soft error handling:
try:
    from_table = self._get_table(from_namespace, from_table_name)
except KeyError:
    raise NoSuchTableError(...)

# NEW - Strict, direct:
if identifier not in config:
    raise NoSuchTableError(f"Table {identifier} not found")
```

**Simplifications:**
1. Remove namespace helper methods - use `config._data` directly
2. Remove defensive checks - fail fast with clear errors
3. No backward compatibility code
4. Simplify `_save_config` signature: `def _save_config(self, config: Config)`
5. Remove `self._config` from `__init__`

**Principles:**
- Explicit over implicit
- Fail fast with clear errors
- Minimal abstraction layers
- Direct data structure access where appropriate

## Implementation Notes

1. Update all methods that currently use `_parse_identifier()` to use `_normalize_identifier()`
2. Update all methods that call `_load_config()` to receive and use the returned config
3. Replace all `_staged_changes.append()` with `_staging_add()` or `_staging_delete()`
4. Remove all helper methods listed above
5. Update `_save_config()` to take `config` parameter
6. Update all config access to use direct Mapping operations with Identifier keys
7. For namespace operations, access `config._data` directly

## Testing Considerations

- Existing tests should pass with minimal changes
- Verify identifier handling works correctly throughout
- Ensure staging operations maintain atomicity
- Test error cases to verify strict error handling

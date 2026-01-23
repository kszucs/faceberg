# Test File Consolidation

## Summary

The test suite has been consolidated from 4 test files to 2 test files, grouping related functionality together.

## Changes

### Before
```
faceberg/tests/
├── test_discovery.py          # DatasetInfo discovery tests
├── test_schema.py              # Schema conversion tests
├── test_catalog.py             # JsonCatalog tests
└── test_faceberg_catalog.py    # FacebergCatalog tests
```

### After
```
faceberg/tests/
├── test_bridge.py              # All bridge layer tests (discovery + schema)
└── test_catalog.py             # All catalog tests (JsonCatalog + FacebergCatalog)
```

## Rationale

### 1. test_bridge.py (merged test_discovery.py + test_schema.py)

**Why merge these?**
- Both test files relate to the **bridge layer** functionality
- `bridge.py` contains both discovery and schema conversion code
- Natural grouping: discovery → schema conversion → TableInfo creation

**What it contains:**
- Dataset discovery tests (9 tests)
  - `test_discover_public_dataset()`
  - `test_discover_with_specific_config()`
  - `test_discover_nonexistent_dataset()`
  - `test_discover_nonexistent_config()`
  - `test_get_parquet_files_for_table()`
  - `test_get_sample_parquet_file()`
  - `test_get_parquet_files_nonexistent_config()`
  - `test_extract_relative_path()`
  - `test_to_table_infos()`

- Schema conversion tests (7 tests)
  - `test_build_schema_from_simple_features()`
  - `test_build_schema_without_split_column()`
  - `test_build_schema_with_nested_features()`
  - `test_build_schema_with_class_label()`
  - `test_unique_field_ids()`
  - `test_infer_schema_from_dataset()`
  - `test_features_dict_to_features_object()`

**Total: 16 tests**

### 2. test_catalog.py (merged test_catalog.py + test_faceberg_catalog.py)

**Why merge these?**
- Both test files relate to **catalog functionality**
- `FacebergCatalog` extends `JsonCatalog`
- Natural grouping: base catalog → extended catalog

**What it contains:**
- JsonCatalog tests (11 tests)
  - `test_create_catalog()`
  - `test_create_namespace()`
  - `test_list_namespaces_empty()`
  - `test_create_table()`
  - `test_load_table()`
  - `test_list_tables()`
  - `test_table_exists()`
  - `test_drop_table()`
  - `test_rename_table()`
  - `test_catalog_persistence()`
  - `test_catalog_json_format()`

- FacebergCatalog tests (9 tests)
  - `test_faceberg_from_config()`
  - `test_faceberg_initialize()`
  - `test_faceberg_initialize_idempotent()`
  - `test_faceberg_create_tables_from_datasets()`
  - `test_faceberg_create_specific_table()`
  - `test_faceberg_create_table_already_exists()`
  - `test_faceberg_create_table_for_config()`
  - `test_faceberg_invalid_table_name_format()`
  - `test_faceberg_dataset_not_found_in_config()`

**Total: 20 tests**

## Benefits

1. **Clearer organization**: Tests grouped by layer (bridge vs catalog)
2. **Easier navigation**: Related tests in same file
3. **Reduced overhead**: Fewer test files to manage
4. **Consistent structure**: Matches module structure (bridge.py, catalog.py)
5. **Better discoverability**: Natural place to find tests for each layer

## Test Results

All tests pass after consolidation:
```bash
$ pytest faceberg/tests/ -v
31 passed, 5 skipped in 15.64s
```

## File Structure

```
test_bridge.py
├── Dataset Discovery Tests (lines 1-141)
│   ├── Public dataset discovery
│   ├── Config filtering
│   ├── Error handling
│   ├── File path extraction
│   └── TableInfo conversion
└── Schema Conversion Tests (lines 143-325)
    ├── Simple features
    ├── Nested structures
    ├── ClassLabel handling
    ├── Field ID uniqueness
    └── Dataset schema inference

test_catalog.py
├── JsonCatalog Tests (lines 1-171)
│   ├── Catalog creation
│   ├── Namespace management
│   ├── Table operations
│   └── Persistence
└── FacebergCatalog Tests (lines 173-362)
    ├── Config-based creation
    ├── Initialization
    ├── Dataset table creation
    └── Error handling
```

## Naming Convention

Test names follow the pattern:
- `test_*` for JsonCatalog tests
- `test_faceberg_*` for FacebergCatalog tests

This makes it easy to distinguish between the two catalog types within the same file.

## Future Considerations

If the test files grow significantly, we could consider:
1. Splitting by test type (unit vs integration)
2. Creating subdirectories (tests/bridge/, tests/catalog/)
3. Using pytest markers to group related tests

For now, two well-organized test files provide the best balance of simplicity and clarity.

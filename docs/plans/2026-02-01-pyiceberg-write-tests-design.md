# PyIceberg Write Tests Design

**Date:** 2026-02-01
**Status:** Approved

## Overview

Add comprehensive tests for write operations (specifically append) to verify that both local catalog and REST catalog support data modification correctly through PyIceberg's Table API.

## Motivation

Currently, the test suite in `test_catalog_pyiceberg.py` only covers read operations (scanning, metadata reading, partition filtering). We need to verify that:
1. The catalog supports write operations correctly
2. Both local and REST catalog behave identically for writes
3. Write operations maintain data integrity (counts, snapshots, partitions)

## Test Structure

### Section F: Write Operations (Local Catalog)

Five tests using the `catalog` fixture:

1. **`test_append_data_basic`**
   - Basic append operation with small PyArrow table
   - Verifies no exceptions are raised
   - Confirms operation completes successfully

2. **`test_append_data_verify_count`**
   - Records row count before append
   - Appends known number of rows
   - Verifies total count = original + appended

3. **`test_append_data_verify_scan`**
   - Appends data with unique text values
   - Scans table and filters for the new data
   - Verifies appended rows are present and readable

4. **`test_append_data_snapshot_history`**
   - Records snapshot count before append
   - Appends data and commits
   - Verifies new snapshot created
   - Confirms latest snapshot has correct operation type

5. **`test_append_data_partition_integrity`**
   - Appends data to specific partition (split='test')
   - Verifies partition spec unchanged
   - Confirms partition filtering still works
   - Validates partition metadata is correct

### Section G: Write Operations (REST Catalog)

Duplicate the five tests from Section F, but using `rest_catalog` fixture instead of `catalog`. Test names will be prefixed with `test_rest_` to match existing REST test naming convention.

This ensures complete parity between local and REST catalog write behavior.

## Test Data Strategy

Each test will create a small PyArrow table matching the `imdb_plain_text` schema:

```python
import pyarrow as pa

test_data = pa.Table.from_pydict({
    "split": ["test", "test"],  # Use existing partition value
    "text": ["Test review 1", "Test review 2"],  # Unique text for verification
    "label": [1, 0]  # Valid label values
})
```

**Rationale:**
- Small data (2 rows) keeps tests fast
- Matches existing schema exactly (split, text, label)
- Uses existing partition value to avoid creating new partitions
- Unique text values allow filtering to verify scan works

## Verification Strategy

### 1. Row Count Verification
```python
before_count = table.scan().to_arrow().num_rows
table.append(test_data)
after_count = table.scan().to_arrow().num_rows
assert after_count == before_count + len(test_data)
```

### 2. Scan Readability Verification
```python
table.append(test_data)
scan = table.scan().filter("text = 'Test review 1'")
result = scan.to_arrow()
assert result.num_rows > 0  # Appended data is readable
```

### 3. Snapshot History Verification
```python
snapshots_before = list(table.snapshots())
table.append(test_data)
snapshots_after = list(table.snapshots())
assert len(snapshots_after) == len(snapshots_before) + 1
assert snapshots_after[-1].summary.get('operation') == 'append'
```

### 4. Partition Integrity Verification
```python
spec_before = table.spec()
table.append(test_data)  # Data with split='test'
spec_after = table.spec()
assert spec_before == spec_after  # Partition spec unchanged

# Verify filtering still works
scan = table.scan().filter("split = 'test'")
result = scan.to_arrow()
assert all(result["split"].to_pylist() == ["test"])
```

## Implementation Notes

- Tests will be added to existing `test_catalog_pyiceberg.py`
- Place Section F after existing Section E (REST Catalog Tests)
- Place Section G at the end of the file
- Maintain existing docstring style (one-line summary)
- Use consistent assertion patterns with existing tests

## Success Criteria

- [ ] All 10 new tests pass
- [ ] Tests verify append operations work correctly
- [ ] Local and REST catalogs show identical behavior
- [ ] Data integrity verified across all dimensions
- [ ] No regressions in existing tests

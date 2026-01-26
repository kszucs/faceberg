"""Tests for reading catalogs using PyIceberg.

These tests verify that PyIceberg can properly read and scan tables
created by the Faceberg catalog from HuggingFace datasets.

PyIceberg's FsspecFileIO provides full support for the hf:// protocol,
enabling both metadata reading and data scanning from HuggingFace datasets.
"""

import pandas as pd
import pyarrow as pa
from pandas.api.types import is_string_dtype
from pyiceberg.transforms import IdentityTransform

# =============================================================================
# A. Table Scanning Tests
# =============================================================================


def test_scan_basic(catalog):
    """Test creating a basic scan object."""
    table = catalog.load_table("default.imdb_plain_text")
    scan = table.scan()

    # Verify scan object is created
    assert scan is not None

    # Verify scan has expected methods
    assert hasattr(scan, "to_arrow")
    assert hasattr(scan, "to_pandas")
    assert hasattr(scan, "to_arrow_batch_reader")


def test_scan_to_arrow(catalog):
    """Test scanning table to Arrow table."""
    table = catalog.load_table("default.imdb_plain_text")
    scan = table.scan()

    # Convert to Arrow table
    arrow_table = scan.to_arrow()

    # Verify it's an Arrow table
    assert isinstance(arrow_table, pa.Table)

    # Verify we have rows
    assert arrow_table.num_rows > 0

    # Verify expected columns are present
    column_names = arrow_table.schema.names
    assert "split" in column_names
    assert "text" in column_names
    assert "label" in column_names

    # Verify split column contains expected values
    split_values = arrow_table["split"].unique().to_pylist()
    assert any(split in split_values for split in ["train", "test", "unsupervised"])


def test_scan_to_pandas(catalog):
    """Test scanning table to Pandas DataFrame."""
    table = catalog.load_table("default.imdb_plain_text")
    scan = table.scan()

    # Convert to Pandas DataFrame
    df = scan.to_pandas()

    # Verify DataFrame shape
    assert len(df) > 0
    assert len(df.columns) > 0

    # Verify split column exists
    assert "split" in df.columns

    # Verify data types are reasonable (accepts both object and StringDtype)
    assert is_string_dtype(df["text"].dtype)


def test_scan_with_selected_fields(catalog):
    """Test scanning with column projection."""
    table = catalog.load_table("default.imdb_plain_text")

    # Scan with only specific columns selected
    scan = table.scan().select("text", "label")
    arrow_table = scan.to_arrow()

    # Verify only selected columns are present
    column_names = arrow_table.schema.names
    assert "text" in column_names
    assert "label" in column_names
    assert "split" not in column_names


def test_scan_limit(catalog):
    """Test scanning with row limit."""
    table = catalog.load_table("default.imdb_plain_text")

    # PyIceberg doesn't support limit() directly on scan, need to materialize first
    scan = table.scan()
    arrow_table = scan.to_arrow()

    # Take first 10 rows
    limited_table = arrow_table.slice(0, 10)

    # Verify exactly 10 rows
    assert limited_table.num_rows == 10


# =============================================================================
# B. Metadata Reading Tests
# =============================================================================


def test_read_schema(catalog):
    """Test reading table schema."""
    table = catalog.load_table("default.imdb_plain_text")
    schema = table.schema()

    # Verify schema has expected fields
    field_names = [field.name for field in schema.fields]
    assert "split" in field_names
    assert "text" in field_names
    assert "label" in field_names

    # Verify field IDs are assigned (all > 0)
    for field in schema.fields:
        assert field.field_id > 0

    # Verify split column is first field
    assert schema.fields[0].name == "split"


def test_read_partition_spec(catalog):
    """Test reading partition specification."""
    table = catalog.load_table("default.imdb_plain_text")
    spec = table.spec()

    # Verify partition spec has at least one field
    assert len(spec.fields) >= 1

    # Find the split partition field
    split_partition = None
    for field in spec.fields:
        if field.name == "split":
            split_partition = field
            break

    # Verify split partition exists with identity transform
    assert split_partition is not None
    assert isinstance(split_partition.transform, IdentityTransform)


def test_read_properties(catalog):
    """Test reading table properties."""
    table = catalog.load_table("default.imdb_plain_text")
    properties = table.properties

    # Verify HuggingFace properties exist
    assert "huggingface.dataset.repo" in properties
    assert properties["huggingface.dataset.repo"] == "stanfordnlp/imdb"

    assert "huggingface.dataset.config" in properties
    assert properties["huggingface.dataset.config"] == "plain_text"

    assert "huggingface.dataset.revision" in properties

    # Verify schema name mapping is present
    assert "schema.name-mapping.default" in properties


def test_read_snapshots(catalog):
    """Test reading table snapshots."""
    table = catalog.load_table("default.imdb_plain_text")
    snapshots = list(table.snapshots())

    # Verify at least one snapshot exists
    assert len(snapshots) > 0

    # Verify snapshot has expected attributes
    snapshot = snapshots[0]
    assert hasattr(snapshot, "snapshot_id")
    assert hasattr(snapshot, "manifest_list")
    assert snapshot.snapshot_id > 0


def test_current_snapshot(catalog):
    """Test reading current snapshot."""
    table = catalog.load_table("default.imdb_plain_text")
    snapshot = table.current_snapshot()

    # Verify current snapshot exists
    assert snapshot is not None

    # Verify snapshot has summary
    assert snapshot.summary is not None

    # Verify snapshot ID exists
    assert snapshot.snapshot_id > 0


# =============================================================================
# C. Partition Pruning Tests
# =============================================================================


def test_partition_filter_single_split(catalog):
    """Test partition pruning with single split filter."""
    table = catalog.load_table("default.imdb_plain_text")

    # Scan with split filter
    scan = table.scan().filter("split = 'train'")
    arrow_table = scan.to_arrow()

    # Verify all rows have split == "train"
    split_values = arrow_table["split"].unique().to_pylist()
    assert split_values == ["train"]

    # Verify we got some rows (not empty result)
    assert arrow_table.num_rows > 0


def test_partition_filter_multiple_splits(catalog):
    """Test partition pruning with multiple split filter."""
    table = catalog.load_table("default.imdb_plain_text")

    # Scan with IN filter for multiple splits
    scan = table.scan().filter("split IN ('train', 'test')")
    df = scan.to_pandas()

    # Verify only train and test splits are present
    unique_splits = df["split"].unique()
    assert set(unique_splits).issubset({"train", "test"})

    # Verify unsupervised split is excluded (if it exists in the dataset)
    assert "unsupervised" not in unique_splits

    # Verify we got some rows
    assert len(df) > 0


def test_scan_all_partitions(catalog):
    """Test scanning all partitions without filter."""
    table = catalog.load_table("default.imdb_plain_text")

    # Scan without filter
    scan = table.scan()
    arrow_table = scan.to_arrow()

    # Group by split to get all partitions
    split_values = set(arrow_table["split"].to_pylist())

    # Verify we have multiple splits
    assert len(split_values) > 1

    # Verify expected splits are present (IMDB has train/test/unsupervised)
    assert "train" in split_values or "test" in split_values


# =============================================================================
# D. Edge Case Tests
# =============================================================================


def test_scan_empty_result(catalog):
    """Test scanning with filter that returns no rows."""
    table = catalog.load_table("default.imdb_plain_text")

    # Scan with impossible filter
    scan = table.scan().filter("split = 'nonexistent_split'")
    arrow_table = scan.to_arrow()

    # Verify 0 rows returned
    assert arrow_table.num_rows == 0

    # Verify schema is still correct
    assert "split" in arrow_table.schema.names
    assert "text" in arrow_table.schema.names


def test_multiple_scans_same_table(catalog):
    """Test multiple independent scans from the same table."""
    table = catalog.load_table("default.imdb_plain_text")

    # Create two independent scans
    scan1 = table.scan().filter("split = 'train'")
    scan2 = table.scan().filter("split = 'test'")

    # Materialize both scans
    df1 = scan1.to_pandas().head(5)
    df2 = scan2.to_pandas().head(3)

    # Verify they don't interfere with each other
    assert len(df1) == 5
    assert all(df1["split"] == "train")

    assert len(df2) == 3
    assert all(df2["split"] == "test")

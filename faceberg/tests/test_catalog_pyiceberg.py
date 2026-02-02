"""Tests for reading catalogs using PyIceberg.

These tests verify that PyIceberg can properly read and scan tables
created by the Faceberg catalog from HuggingFace datasets.

PyIceberg's FsspecFileIO provides full support for the hf:// protocol,
enabling both metadata reading and data scanning from HuggingFace datasets.
"""

import uuid

import pyarrow as pa
import pytest
from pandas.api.types import is_string_dtype
from pyiceberg.catalog.rest import RestCatalog
from pyiceberg.transforms import IdentityTransform

# =============================================================================
# A. Table Scanning Tests
# =============================================================================


def test_scan_basic(synced_catalog):
    """Test creating a basic scan object."""
    catalog = synced_catalog
    table = catalog.load_table("default.imdb_plain_text")
    scan = table.scan()

    # Verify scan object is created
    assert scan is not None

    # Verify scan has expected methods
    assert hasattr(scan, "to_arrow")
    assert hasattr(scan, "to_pandas")
    assert hasattr(scan, "to_arrow_batch_reader")


def test_scan_to_arrow(synced_catalog):
    """Test scanning table to Arrow table."""
    catalog = synced_catalog
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


def test_scan_to_pandas(synced_catalog):
    """Test scanning table to Pandas DataFrame."""
    catalog = synced_catalog
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


def test_scan_with_selected_fields(synced_catalog):
    """Test scanning with column projection."""
    catalog = synced_catalog
    table = catalog.load_table("default.imdb_plain_text")

    # Scan with only specific columns selected
    scan = table.scan().select("text", "label")
    arrow_table = scan.to_arrow()

    # Verify only selected columns are present
    column_names = arrow_table.schema.names
    assert "text" in column_names
    assert "label" in column_names
    assert "split" not in column_names


def test_scan_limit(synced_catalog):
    """Test scanning with row limit."""
    catalog = synced_catalog
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


def test_read_schema(synced_catalog):
    """Test reading table schema."""
    catalog = synced_catalog
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


def test_read_partition_spec(synced_catalog):
    """Test reading partition specification."""
    catalog = synced_catalog
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


def test_read_properties(synced_catalog):
    """Test reading table properties."""
    catalog = synced_catalog
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


def test_read_snapshots(synced_catalog):
    """Test reading table snapshots."""
    catalog = synced_catalog
    table = catalog.load_table("default.imdb_plain_text")
    snapshots = list(table.snapshots())

    # Verify at least one snapshot exists
    assert len(snapshots) > 0

    # Verify snapshot has expected attributes
    snapshot = snapshots[0]
    assert hasattr(snapshot, "snapshot_id")
    assert hasattr(snapshot, "manifest_list")
    assert snapshot.snapshot_id > 0


def test_current_snapshot(synced_catalog):
    """Test reading current snapshot."""
    catalog = synced_catalog
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


def test_partition_filter_single_split(synced_catalog):
    """Test partition pruning with single split filter."""
    catalog = synced_catalog
    table = catalog.load_table("default.imdb_plain_text")

    # Scan with split filter
    scan = table.scan().filter("split = 'train'")
    arrow_table = scan.to_arrow()

    # Verify all rows have split == "train"
    split_values = arrow_table["split"].unique().to_pylist()
    assert split_values == ["train"]

    # Verify we got some rows (not empty result)
    assert arrow_table.num_rows > 0


def test_partition_filter_multiple_splits(synced_catalog):
    """Test partition pruning with multiple split filter."""
    catalog = synced_catalog
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


def test_scan_all_partitions(synced_catalog):
    """Test scanning all partitions without filter."""
    catalog = synced_catalog
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


def test_scan_empty_result(synced_catalog):
    """Test scanning with filter that returns no rows."""
    catalog = synced_catalog
    table = catalog.load_table("default.imdb_plain_text")

    # Scan with impossible filter
    scan = table.scan().filter("split = 'nonexistent_split'")
    arrow_table = scan.to_arrow()

    # Verify 0 rows returned
    assert arrow_table.num_rows == 0

    # Verify schema is still correct
    assert "split" in arrow_table.schema.names
    assert "text" in arrow_table.schema.names


def test_multiple_scans_same_table(synced_catalog):
    """Test multiple independent scans from the same table."""
    catalog = synced_catalog
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


# =============================================================================
# E. REST Catalog Tests
# =============================================================================


@pytest.fixture
def rest_catalog(rest_server):
    """Create PyIceberg RestCatalog connected to test server.

    Configures the catalog to use HfFileIO for handling hf:// URIs.
    """
    catalog = RestCatalog(
        name="faceberg_rest",
        uri=rest_server,
        **{
            "py-io-impl": "faceberg.catalog.HfFileIO",
        },
    )
    return catalog


def test_rest_list_namespaces(rest_catalog):
    """Test listing namespaces via REST catalog."""
    namespaces = rest_catalog.list_namespaces()

    # Verify we got namespaces
    assert len(namespaces) > 0

    # Verify default namespace exists
    namespace_strs = [".".join(ns) if isinstance(ns, tuple) else ns for ns in namespaces]
    assert "default" in namespace_strs


def test_rest_list_tables(rest_catalog):
    """Test listing tables via REST catalog."""
    tables = rest_catalog.list_tables("default")

    # Verify we got tables
    assert len(tables) > 0

    # Verify imdb_plain_text table exists
    table_names = [t[1] if isinstance(t, tuple) and len(t) > 1 else str(t) for t in tables]
    assert "imdb_plain_text" in table_names


def test_rest_load_table(rest_catalog):
    """Test loading a table via REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")

    # Verify table loaded successfully
    assert table is not None

    # Verify table has schema
    schema = table.schema()
    assert schema is not None
    assert len(schema.fields) > 0


def test_rest_scan_to_arrow(rest_catalog):
    """Test scanning table to Arrow via REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")
    scan = table.scan()

    # Convert to Arrow table
    arrow_table = scan.to_arrow()

    # Verify it's an Arrow table
    assert isinstance(arrow_table, pa.Table)

    # Verify we have rows
    assert arrow_table.num_rows > 0

    # Verify expected columns
    column_names = arrow_table.schema.names
    assert "split" in column_names
    assert "text" in column_names
    assert "label" in column_names


def test_rest_scan_to_pandas(rest_catalog):
    """Test scanning table to Pandas via REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")
    scan = table.scan()

    # Convert to Pandas DataFrame
    df = scan.to_pandas()

    # Verify DataFrame shape
    assert len(df) > 0
    assert len(df.columns) > 0

    # Verify split column exists
    assert "split" in df.columns


def test_rest_partition_filter(rest_catalog):
    """Test partition filtering via REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")

    # Scan with split filter
    scan = table.scan().filter("split = 'train'")
    arrow_table = scan.to_arrow()

    # Verify all rows have split == "train"
    split_values = arrow_table["split"].unique().to_pylist()
    assert split_values == ["train"]

    # Verify we got some rows
    assert arrow_table.num_rows > 0


def test_rest_read_schema(rest_catalog):
    """Test reading table schema via REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")
    schema = table.schema()

    # Verify schema has expected fields
    field_names = [field.name for field in schema.fields]
    assert "split" in field_names
    assert "text" in field_names
    assert "label" in field_names


def test_rest_read_properties(rest_catalog):
    """Test reading table properties via REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")
    properties = table.properties

    # Verify HuggingFace properties exist
    assert "huggingface.dataset.repo" in properties
    assert properties["huggingface.dataset.repo"] == "stanfordnlp/imdb"


def test_rest_read_snapshots(rest_catalog):
    """Test reading table snapshots via REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")
    snapshots = list(table.snapshots())

    # Verify at least one snapshot exists
    assert len(snapshots) > 0

    # Verify snapshot has expected attributes
    snapshot = snapshots[0]
    assert hasattr(snapshot, "snapshot_id")
    assert snapshot.snapshot_id > 0


def test_rest_column_projection(rest_catalog):
    """Test column projection via REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")

    # Scan with only specific columns selected
    scan = table.scan().select("text", "label")
    arrow_table = scan.to_arrow()

    # Verify only selected columns are present
    column_names = arrow_table.schema.names
    assert "text" in column_names
    assert "label" in column_names
    assert "split" not in column_names


# =============================================================================
# F. Write Operations (Local Catalog)
# =============================================================================


def test_append_data_basic(writable_catalog):
    """Test basic append operation."""
    table = writable_catalog.load_table("default.test_table")

    # Create test data matching schema
    test_data = pa.Table.from_pydict(
        {
            "split": ["test", "test"],
            "text": ["Test review 1", "Test review 2"],
            "label": [1, 0],
        }
    )

    # Append data - should complete without error
    table.append(test_data)

    # Verify operation completed
    assert table is not None


def test_append_data_verify_count(writable_catalog):
    """Test row count increases correctly after append."""
    table = writable_catalog.load_table("default.test_table")

    # Record count before append
    before_count = table.scan().to_arrow().num_rows

    # Create and append test data
    test_data = pa.Table.from_pydict(
        {
            "split": ["test", "test"],
            "text": ["Count test review 1", "Count test review 2"],
            "label": [1, 0],
        }
    )
    table.append(test_data)

    # Verify count increased by expected amount
    after_count = table.scan().to_arrow().num_rows
    assert after_count == before_count + len(test_data)


def test_append_data_verify_scan(writable_catalog):
    """Test appended data is readable via scan."""
    table = writable_catalog.load_table("default.test_table")

    # Create test data with unique text for verification
    unique_text = f"Unique test review {uuid.uuid4()}"
    test_data = pa.Table.from_pydict(
        {
            "split": ["test"],
            "text": [unique_text],
            "label": [1],
        }
    )

    # Append data
    table.append(test_data)

    # Scan for appended data
    scan = table.scan().filter(f"text = '{unique_text}'")
    result = scan.to_arrow()

    # Verify appended data is present
    assert result.num_rows == 1
    assert result["text"][0].as_py() == unique_text


def test_append_data_snapshot_history(writable_catalog):
    """Test snapshot history is updated after append."""
    table = writable_catalog.load_table("default.test_table")

    # Record snapshot count before append
    snapshots_before = list(table.snapshots())
    snapshot_count_before = len(snapshots_before)

    # Create and append test data
    test_data = pa.Table.from_pydict(
        {
            "split": ["test"],
            "text": ["Snapshot test review"],
            "label": [1],
        }
    )
    table.append(test_data)

    # Reload table to get updated snapshots
    table = writable_catalog.load_table("default.test_table")
    snapshots_after = list(table.snapshots())

    # Verify new snapshot was created
    assert len(snapshots_after) == snapshot_count_before + 1

    # Verify latest snapshot has append operation
    latest_snapshot = snapshots_after[-1]
    assert latest_snapshot.summary is not None
    # Summary.operation is an enum, not a string
    from pyiceberg.table.snapshots import Operation

    assert latest_snapshot.summary.operation == Operation.APPEND


def test_append_data_partition_integrity(writable_catalog):
    """Test partition integrity is maintained after append."""
    table = writable_catalog.load_table("default.test_table")

    # Record partition spec before append
    spec_before = table.spec()

    # Create test data for specific partition
    test_data = pa.Table.from_pydict(
        {
            "split": ["test", "test"],
            "text": ["Partition test review 1", "Partition test review 2"],
            "label": [1, 0],
        }
    )
    table.append(test_data)

    # Reload table and verify partition spec unchanged
    table = writable_catalog.load_table("default.test_table")
    spec_after = table.spec()
    assert len(spec_before.fields) == len(spec_after.fields)

    # Verify partition filtering still works
    scan = table.scan().filter("split = 'test'")
    result = scan.to_arrow()

    # All rows should have split == 'test'
    split_values = result["split"].unique().to_pylist()
    assert split_values == ["test"]
    assert result.num_rows > 0


# =============================================================================
# G. Write Operations (REST Catalog)
# =============================================================================


@pytest.mark.skip(reason="REST server write operations not yet implemented")
def test_rest_append_data_basic(rest_catalog):
    """Test basic append operation via REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")

    # Create test data matching schema
    test_data = pa.Table.from_pydict(
        {
            "split": ["test", "test"],
            "text": ["REST test review 1", "REST test review 2"],
            "label": [1, 0],
        }
    )

    # Append data - should complete without error
    table.append(test_data)

    # Verify operation completed
    assert table is not None


@pytest.mark.skip(reason="REST server write operations not yet implemented")
def test_rest_append_data_verify_count(rest_catalog):
    """Test row count increases correctly after append via REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")

    # Record count before append
    before_count = table.scan().to_arrow().num_rows

    # Create and append test data
    test_data = pa.Table.from_pydict(
        {
            "split": ["test", "test"],
            "text": ["REST count test review 1", "REST count test review 2"],
            "label": [1, 0],
        }
    )
    table.append(test_data)

    # Verify count increased by expected amount
    after_count = table.scan().to_arrow().num_rows
    assert after_count == before_count + len(test_data)


@pytest.mark.skip(reason="REST server write operations not yet implemented")
def test_rest_append_data_verify_scan(rest_catalog):
    """Test appended data is readable via scan through REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")

    # Create test data with unique text for verification
    unique_text = f"REST unique test review {uuid.uuid4()}"
    test_data = pa.Table.from_pydict(
        {
            "split": ["test"],
            "text": [unique_text],
            "label": [1],
        }
    )

    # Append data
    table.append(test_data)

    # Scan for appended data
    scan = table.scan().filter(f"text = '{unique_text}'")
    result = scan.to_arrow()

    # Verify appended data is present
    assert result.num_rows == 1
    assert result["text"][0].as_py() == unique_text


@pytest.mark.skip(reason="REST server write operations not yet implemented")
def test_rest_append_data_snapshot_history(rest_catalog):
    """Test snapshot history is updated after append via REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")

    # Record snapshot count before append
    snapshots_before = list(table.snapshots())
    snapshot_count_before = len(snapshots_before)

    # Create and append test data
    test_data = pa.Table.from_pydict(
        {
            "split": ["test"],
            "text": ["REST snapshot test review"],
            "label": [1],
        }
    )
    table.append(test_data)

    # Reload table to get updated snapshots
    table = rest_catalog.load_table("default.imdb_plain_text")
    snapshots_after = list(table.snapshots())

    # Verify new snapshot was created
    assert len(snapshots_after) == snapshot_count_before + 1

    # Verify latest snapshot has append operation
    latest_snapshot = snapshots_after[-1]
    assert latest_snapshot.summary is not None
    # Summary.operation is an enum, not a string
    from pyiceberg.table.snapshots import Operation

    assert latest_snapshot.summary.operation == Operation.APPEND


@pytest.mark.skip(reason="REST server write operations not yet implemented")
def test_rest_append_data_partition_integrity(rest_catalog):
    """Test partition integrity is maintained after append via REST catalog."""
    table = rest_catalog.load_table("default.imdb_plain_text")

    # Record partition spec before append
    spec_before = table.spec()

    # Create test data for specific partition
    test_data = pa.Table.from_pydict(
        {
            "split": ["test", "test"],
            "text": ["REST partition test review 1", "REST partition test review 2"],
            "label": [1, 0],
        }
    )
    table.append(test_data)

    # Reload table and verify partition spec unchanged
    table = rest_catalog.load_table("default.imdb_plain_text")
    spec_after = table.spec()
    assert len(spec_before.fields) == len(spec_after.fields)

    # Verify partition filtering still works
    scan = table.scan().filter("split = 'test'")
    result = scan.to_arrow()

    # All rows should have split == 'test'
    split_values = result["split"].unique().to_pylist()
    assert split_values == ["test"]
    assert result.num_rows > 0

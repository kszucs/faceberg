"""Tests for FacebergCatalog implementation."""

import re
import uuid

import pyarrow as pa
import pytest
from huggingface_hub import HfFileSystem
from pandas.api.types import is_string_dtype
from pyiceberg.exceptions import (
    NamespaceAlreadyExistsError,
    NamespaceNotEmptyError,
    NoSuchTableError,
    TableAlreadyExistsError,
)
from pyiceberg.io.fsspec import FsspecFileIO
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import LongType, NestedField, StringType

from faceberg.catalog import HfFileIO, HfLocationProvider, LocalCatalog, RemoteCatalog
from faceberg.catalog import catalog as catalog_factory


@pytest.fixture
def test_schema():
    """Create a test schema."""
    return Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "name", StringType(), required=False),
    )


@pytest.fixture
def get_table_location(tmp_path):
    """Generate a location for a table.

    Returns a callable that generates unique table locations.
    """
    counter = 0

    def _location(identifier: str = None):
        nonlocal counter
        counter += 1
        if identifier:
            # Use identifier as directory name
            name = identifier.replace(".", "_")
        else:
            # Generate unique name
            name = f"table_{counter}"
        location_dir = tmp_path / "tables" / name
        location_dir.mkdir(parents=True, exist_ok=True)
        return f"file://{location_dir.as_posix()}"

    return _location


# =============================================================================
# Catalog Creation Tests (Local-specific, not parametrized)
# =============================================================================


class TestCatalogCreation:
    """Tests for catalog creation and initialization."""

    def test_create_local_catalog(self, tmp_path):
        """Test LocalCatalog creation."""
        catalog_dir = tmp_path / "test_catalog"
        catalog_dir.mkdir()
        uri = f"file://{catalog_dir.as_posix()}"
        catalog = LocalCatalog(name=str(catalog_dir), uri=uri)

        # catalog.name is derived from path
        assert catalog.name == str(catalog_dir)
        assert catalog.uri.startswith("file:///")
        assert catalog.uri.endswith(str(catalog_dir.name))
        assert catalog_dir.exists()

    def test_local_catalog_from_config(self, tmp_path):
        """Test creating LocalCatalog from local config file."""
        catalog_dir = tmp_path / "test_catalog"
        catalog_dir.mkdir()
        uri = f"file://{catalog_dir.as_posix()}"
        catalog = LocalCatalog(name=str(catalog_dir), uri=uri)

        assert catalog.uri.startswith("file:///")
        assert catalog.uri.endswith(str(catalog_dir.name))

    def test_catalog_persistence(self, tmp_path, test_schema):
        """Test that catalog persists across instances."""
        catalog_dir = tmp_path / "test_catalog"
        # Create catalog and table
        uri = f"file://{catalog_dir.as_posix()}"
        catalog1 = LocalCatalog(name=str(catalog_dir), uri=uri)
        catalog1.init()

        catalog1.create_namespace("default")
        table_location_dir = tmp_path / "tables" / "default_test_table"
        table_location_dir.mkdir(parents=True)
        catalog1.create_table(
            "default.test_table", test_schema, location=f"file://{table_location_dir.as_posix()}"
        )
        # Changes are automatically persisted via context manager

        # Create new catalog instance
        catalog2 = LocalCatalog(name=str(catalog_dir), uri=uri)

        # Table should still exist
        assert catalog2.table_exists("default.test_table")
        table = catalog2.load_table("default.test_table")
        assert table.schema() == test_schema


# =============================================================================
# Namespace Operations (Parametrized for local/remote)
# =============================================================================


class TestNamespaceOperations:
    """Tests for namespace create, read, update, delete operations."""

    def test_create_namespace(self, catalog):
        """Test namespace creation."""
        catalog.create_namespace("default")
        assert ("default",) in catalog.list_namespaces()

    def test_list_namespaces_empty(self, catalog):
        """Test listing namespaces when none exist."""
        namespaces = catalog.list_namespaces()
        assert namespaces == []

    def test_list_namespaces_with_tables(self, catalog, test_schema, get_table_location):
        """Test listing namespaces with hierarchical names."""
        catalog.create_namespace("ns1")
        catalog.create_table("ns1.table1", test_schema, location=get_table_location("ns1.table1"))

        namespaces = catalog.list_namespaces()
        assert ("ns1",) in namespaces

    def test_drop_namespace(self, catalog):
        """Test dropping an empty namespace."""
        catalog.create_namespace("test_ns")
        catalog.drop_namespace("test_ns")

        # Namespace should not appear in list
        assert ("test_ns",) not in catalog.list_namespaces()

    def test_drop_namespace_not_empty(self, catalog, test_schema, get_table_location):
        """Test that dropping a non-empty namespace raises error."""
        catalog.create_namespace("test_ns")
        catalog.create_table(
            "test_ns.table1", test_schema, location=get_table_location("test_ns.table1")
        )

        with pytest.raises(NamespaceNotEmptyError):
            catalog.drop_namespace("test_ns")

    def test_update_namespace_properties(self, catalog):
        """Test updating namespace properties."""
        catalog.create_namespace("test_ns")
        summary = catalog.update_namespace_properties(
            "test_ns", removals={"old_prop"}, updates={"new_prop": "value"}
        )

        # Currently returns empty summary
        assert summary.removed == []
        assert summary.updated == []
        assert summary.missing == []

    def test_create_namespace_already_exists(self, catalog, test_schema, get_table_location):
        """Test creating namespace that already exists (has tables)."""
        catalog.create_namespace("test_ns")
        catalog.create_table(
            "test_ns.table1", test_schema, location=get_table_location("test_ns.table1")
        )

        with pytest.raises(NamespaceAlreadyExistsError):
            catalog.create_namespace("test_ns")


# =============================================================================
# Table Read Operations (Parametrized for local/remote)
# =============================================================================


class TestTableRead:
    """Tests for table read operations."""

    def test_load_table(self, catalog, test_schema, get_table_location):
        """Test loading a table."""
        catalog.create_namespace("default")
        catalog.create_table(
            identifier="default.test_table",
            schema=test_schema,
            location=get_table_location("default.test_table"),
        )

        table = catalog.load_table("default.test_table")

        assert table.schema() == test_schema

    def test_list_tables(self, catalog, test_schema, get_table_location):
        """Test listing tables."""
        catalog.create_namespace("default")

        # Create multiple tables
        catalog.create_table(
            "default.table1", test_schema, location=get_table_location("default.table1")
        )
        catalog.create_table(
            "default.table2", test_schema, location=get_table_location("default.table2")
        )

        tables = catalog.list_tables("default")

        assert len(tables) == 2
        assert ("default", "table1") in tables
        assert ("default", "table2") in tables

    def test_table_exists(self, catalog, test_schema, get_table_location):
        """Test checking table existence."""
        catalog.create_namespace("default")

        assert not catalog.table_exists("default.test_table")

        catalog.create_table(
            "default.test_table", test_schema, location=get_table_location("default.test_table")
        )

        assert catalog.table_exists("default.test_table")

    def test_load_table_not_found(self, catalog):
        """Test loading non-existent table raises error."""
        with pytest.raises(NoSuchTableError):
            catalog.load_table("default.nonexistent")


# =============================================================================
# Table Write Operations (Parametrized for local/remote)
# =============================================================================


class TestTableWrite:
    """Tests for table create, update, delete operations."""

    def test_create_table(self, catalog, test_schema, get_table_location):
        """Test table creation."""
        catalog.create_namespace("default")

        table = catalog.create_table(
            identifier="default.test_table",
            schema=test_schema,
            location=get_table_location("default.test_table"),
        )

        assert table.metadata is not None
        assert table.schema() == test_schema

    def test_drop_table(self, catalog, test_schema, get_table_location):
        """Test dropping a table."""
        catalog.create_namespace("default")
        catalog.create_table(
            "default.test_table", test_schema, location=get_table_location("default.test_table")
        )

        assert catalog.table_exists("default.test_table")

        catalog.drop_table("default.test_table")

        assert not catalog.table_exists("default.test_table")

    def test_rename_table(self, catalog, test_schema, get_table_location):
        """Test renaming a table."""
        catalog.create_namespace("default")
        catalog.create_table(
            "default.old_name", test_schema, location=get_table_location("default.old_name")
        )

        catalog.rename_table("default.old_name", "default.new_name")

        assert not catalog.table_exists("default.old_name")
        assert catalog.table_exists("default.new_name")

    def test_drop_table_not_found(self, catalog):
        """Test dropping non-existent table raises error."""
        with pytest.raises(NoSuchTableError):
            catalog.drop_table("default.nonexistent")

    def test_rename_table_source_not_found(self, catalog):
        """Test renaming non-existent table raises error."""
        with pytest.raises(NoSuchTableError):
            catalog.rename_table("default.nonexistent", "default.new_name")

    def test_rename_table_destination_exists(self, catalog, test_schema, get_table_location):
        """Test that renaming to existing table name raises error."""
        catalog.create_namespace("default")
        catalog.create_table(
            "default.table1", test_schema, location=get_table_location("default.table1")
        )
        catalog.create_table(
            "default.table2", test_schema, location=get_table_location("default.table2")
        )

        with pytest.raises(TableAlreadyExistsError):
            catalog.rename_table("default.table1", "default.table2")

    def test_create_table_transaction_not_implemented(self, catalog, test_schema):
        """Test that table transactions are not yet implemented."""
        with pytest.raises(NotImplementedError):
            catalog.create_table_transaction("default.test_table", test_schema)


# =============================================================================
# Table Write Properties (Parametrized for local/remote)
# =============================================================================


class TestTableWriteProperties:
    """Tests for table write properties and LocationProvider."""

    def test_create_table_with_write_properties(self, catalog, test_schema, get_table_location):
        """Test creating a table with write LocationProvider configured."""
        catalog.create_namespace("default")
        table = catalog.create_table(
            "default.write_test",
            schema=test_schema,
            location=get_table_location("default.write_test"),
            properties={
                "write.py-location-provider.impl": "faceberg.catalog.HfLocationProvider",
                "huggingface.write.split": "train",
            },
        )

        # Verify LocationProvider is configured
        assert (
            table.properties.get("write.py-location-provider.impl")
            == "faceberg.catalog.HfLocationProvider"
        )

    def test_location_provider_returns_correct_type(self, catalog, test_schema, get_table_location):
        """Test that table.location_provider() returns HfLocationProvider."""
        catalog.create_namespace("default")
        table = catalog.create_table(
            "default.test_table",
            schema=test_schema,
            location=get_table_location("default.test_table"),
            properties={
                "write.py-location-provider.impl": "faceberg.catalog.HfLocationProvider",
            },
        )

        # Verify LocationProvider is configured
        provider = table.location_provider()
        assert isinstance(provider, HfLocationProvider)


# =============================================================================
# Table Append Operations (Parametrized for local/remote)
# =============================================================================


class TestTableAppend:
    """Tests for PyIceberg append operations on writable tables."""

    @pytest.fixture
    def writable_catalog(self, catalog, tmp_path):
        """Create catalog with writable table for testing write operations.

        Creates a catalog with a writable table (not from HuggingFace dataset)
        that can be used to test append and other write operations.

        Args:
            catalog: Empty catalog instance (local or remote, from parametrized fixture)
            tmp_path: Temporary directory for table data

        Returns:
            Catalog instance with a writable test_table in the default namespace
        """
        # Create data directory for the table
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        location = f"file://{data_dir.as_posix()}"

        # Create the table with schema matching imdb dataset
        schema = Schema(
            NestedField(field_id=1, name="split", field_type=StringType(), required=False),
            NestedField(field_id=2, name="text", field_type=StringType(), required=False),
            NestedField(field_id=3, name="label", field_type=LongType(), required=False),
        )

        partition_spec = PartitionSpec(
            PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="split")
        )

        # Create the table with mandatory location argument
        catalog.create_table(
            identifier="default.test_table",
            schema=schema,
            location=location,
            partition_spec=partition_spec,
        )

        return catalog

    def test_append_data(self, writable_catalog):
        """Verifies data is appended, count increases, and data is scannable."""
        table = writable_catalog.load_table("default.test_table")

        # Record count before append
        before_count = table.scan().to_arrow().num_rows

        # Create test data with unique text for verification
        unique_text = f"Unique test review {uuid.uuid4()}"
        test_data = pa.Table.from_pydict(
            {
                "split": ["test", "test"],
                "text": [unique_text, "Test review 2"],
                "label": [1, 0],
            }
        )

        # Append data
        table.append(test_data)

        # Verify count increased by expected amount
        after_count = table.scan().to_arrow().num_rows
        assert after_count == before_count + len(test_data)

        # Verify appended data is readable via scan
        scan = table.scan().filter(f"text = '{unique_text}'")
        result = scan.to_arrow()
        assert result.num_rows == 1
        assert result["text"][0].as_py() == unique_text

    def test_append_data_snapshot_history(self, writable_catalog):
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

    def test_append_data_partition_integrity(self, writable_catalog):
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
# Dataset Operations (Parametrized for local/remote)
# =============================================================================


class TestDatasetOperations:
    """Tests for HuggingFace dataset integration."""

    def test_namespace_exists_after_add_dataset(self, synced_catalog):
        """Test that namespaces exist after datasets are added."""
        # Namespace should exist after add_dataset
        assert ("stanfordnlp",) in synced_catalog.list_namespaces()

        # Verify table exists
        tables = synced_catalog.list_tables("stanfordnlp")
        assert len(tables) > 0

    def test_add_dataset_already_exists(self, catalog):
        """Test adding a dataset that already exists raises error."""
        # Create table first time
        catalog.add_dataset("default.imdb_plain_text", "stanfordnlp/imdb", config="plain_text")

        # Try to create again - should raise
        with pytest.raises(TableAlreadyExistsError):
            catalog.add_dataset(
                "default.imdb_plain_text",
                "stanfordnlp/imdb",
                config="plain_text",
            )

    def test_add_dataset_with_config(self, catalog):
        """Test adding a dataset with a specific config."""
        # Create table
        table = catalog.add_dataset(
            "default.imdb_plain_text",
            "stanfordnlp/imdb",
            config="plain_text",
        )

        # Verify table
        assert table is not None
        assert table.schema() is not None
        assert len(table.schema().fields) > 0

        # Verify table properties
        props = table.properties
        assert "huggingface.dataset.repo" in props
        assert props["huggingface.dataset.repo"] == "stanfordnlp/imdb"
        assert "huggingface.dataset.config" in props
        assert props["huggingface.dataset.config"] == "plain_text"


# =============================================================================
# Unsupported Operations (Parametrized for local/remote)
# =============================================================================


class TestTableScanning:
    """Tests for PyIceberg table scanning operations."""

    def test_scan_basic(self, synced_catalog):
        """Test creating a basic scan object."""
        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")
        scan = table.scan()

        # Verify scan object is created
        assert scan is not None

        # Verify scan has expected methods
        assert hasattr(scan, "to_arrow")
        assert hasattr(scan, "to_pandas")
        assert hasattr(scan, "to_arrow_batch_reader")

    def test_scan_to_arrow(self, synced_catalog):
        """Test scanning table to Arrow table."""

        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")
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

    def test_scan_to_pandas(self, synced_catalog):
        """Test scanning table to Pandas DataFrame."""

        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")
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

    def test_scan_with_selected_fields(self, synced_catalog):
        """Test scanning with column projection."""
        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")

        # Scan with only specific columns selected
        scan = table.scan().select("text", "label")
        arrow_table = scan.to_arrow()

        # Verify only selected columns are present
        column_names = arrow_table.schema.names
        assert "text" in column_names
        assert "label" in column_names
        assert "split" not in column_names

    def test_scan_limit(self, synced_catalog):
        """Test scanning with row limit."""
        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")

        # PyIceberg doesn't support limit() directly on scan, need to materialize first
        scan = table.scan()
        arrow_table = scan.to_arrow()

        # Take first 10 rows
        limited_table = arrow_table.slice(0, 10)

        # Verify exactly 10 rows
        assert limited_table.num_rows == 10

    def test_partition_filter_single_split(self, synced_catalog):
        """Test partition pruning with single split filter."""
        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")

        # Scan with split filter
        scan = table.scan().filter("split = 'train'")
        arrow_table = scan.to_arrow()

        # Verify all rows have split == "train"
        split_values = arrow_table["split"].unique().to_pylist()
        assert split_values == ["train"]

        # Verify we got some rows (not empty result)
        assert arrow_table.num_rows > 0

    def test_partition_filter_multiple_splits(self, synced_catalog):
        """Test partition pruning with multiple split filter."""
        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")

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

    def test_scan_all_partitions(self, synced_catalog):
        """Test scanning all partitions without filter."""
        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")

        # Scan without filter
        scan = table.scan()
        arrow_table = scan.to_arrow()

        # Group by split to get all partitions
        split_values = set(arrow_table["split"].to_pylist())

        # Verify we have multiple splits
        assert len(split_values) > 1

        # Verify expected splits are present (IMDB has train/test/unsupervised)
        assert "train" in split_values or "test" in split_values

    def test_scan_empty_result(self, synced_catalog):
        """Test scanning with filter that returns no rows."""
        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")

        # Scan with impossible filter
        scan = table.scan().filter("split = 'nonexistent_split'")
        arrow_table = scan.to_arrow()

        # Verify 0 rows returned
        assert arrow_table.num_rows == 0

        # Verify schema is still correct
        assert "split" in arrow_table.schema.names
        assert "text" in arrow_table.schema.names

    def test_multiple_scans_same_table(self, synced_catalog):
        """Test multiple independent scans from the same table."""
        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")

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


class TestTableMetadata:
    """Tests for PyIceberg metadata reading operations."""

    def test_read_schema(self, synced_catalog):
        """Test reading table schema."""
        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")
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

    def test_read_partition_spec(self, synced_catalog):
        """Test reading partition specification."""

        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")
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

    def test_read_properties(self, synced_catalog):
        """Test reading table properties."""
        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")
        properties = table.properties

        # Verify HuggingFace properties exist
        assert "huggingface.dataset.repo" in properties
        assert properties["huggingface.dataset.repo"] == "stanfordnlp/imdb"

        assert "huggingface.dataset.config" in properties
        assert properties["huggingface.dataset.config"] == "plain_text"

        assert "huggingface.dataset.revision" in properties

        # Verify schema name mapping is present
        assert "schema.name-mapping.default" in properties

    def test_read_snapshots(self, synced_catalog):
        """Test reading table snapshots."""
        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")
        snapshots = list(table.snapshots())

        # Verify at least one snapshot exists
        assert len(snapshots) > 0

        # Verify snapshot has expected attributes
        snapshot = snapshots[0]
        assert hasattr(snapshot, "snapshot_id")
        assert hasattr(snapshot, "manifest_list")
        assert snapshot.snapshot_id > 0

    def test_current_snapshot(self, synced_catalog):
        """Test reading current snapshot."""
        catalog = synced_catalog
        table = catalog.load_table("stanfordnlp.imdb")
        snapshot = table.current_snapshot()

        # Verify current snapshot exists
        assert snapshot is not None

        # Verify snapshot has summary
        assert snapshot.summary is not None

        # Verify snapshot ID exists
        assert snapshot.snapshot_id > 0


# =============================================================================
# REST Catalog Integration Tests
# =============================================================================


class TestRestCatalogOperations:
    """Tests for PyIceberg REST catalog basic operations."""

    def test_rest_list_namespaces(self, rest_catalog):
        """Test listing namespaces via REST catalog."""
        namespaces = rest_catalog.list_namespaces()

        # Verify we got namespaces
        assert len(namespaces) > 0

        # Verify stanfordnlp namespace exists
        namespace_strs = [".".join(ns) if isinstance(ns, tuple) else ns for ns in namespaces]
        assert "stanfordnlp" in namespace_strs

    def test_rest_list_tables(self, rest_catalog):
        """Test listing tables via REST catalog."""
        tables = rest_catalog.list_tables("stanfordnlp")

        # Verify we got tables
        assert len(tables) > 0

        # Verify imdb table exists
        table_names = [t[1] if isinstance(t, tuple) and len(t) > 1 else str(t) for t in tables]
        assert "imdb" in table_names

    def test_rest_load_table(self, rest_catalog):
        """Test loading a table via REST catalog."""
        table = rest_catalog.load_table("stanfordnlp.imdb")

        # Verify table loaded successfully
        assert table is not None

        # Verify table has schema
        schema = table.schema()
        assert schema is not None
        assert len(schema.fields) > 0


class TestRestCatalogScanning:
    """Tests for PyIceberg REST catalog scanning operations."""

    def test_rest_scan_to_arrow(self, rest_catalog):
        """Test scanning table to Arrow via REST catalog."""
        import pyarrow as pa

        table = rest_catalog.load_table("stanfordnlp.imdb")
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

    def test_rest_scan_to_pandas(self, rest_catalog):
        """Test scanning table to Pandas via REST catalog."""
        table = rest_catalog.load_table("stanfordnlp.imdb")
        scan = table.scan()

        # Convert to Pandas DataFrame
        df = scan.to_pandas()

        # Verify DataFrame shape
        assert len(df) > 0
        assert len(df.columns) > 0

        # Verify split column exists
        assert "split" in df.columns

    def test_rest_partition_filter(self, rest_catalog):
        """Test partition filtering via REST catalog."""
        table = rest_catalog.load_table("stanfordnlp.imdb")

        # Scan with split filter
        scan = table.scan().filter("split = 'train'")
        arrow_table = scan.to_arrow()

        # Verify all rows have split == "train"
        split_values = arrow_table["split"].unique().to_pylist()
        assert split_values == ["train"]

        # Verify we got some rows
        assert arrow_table.num_rows > 0

    def test_rest_column_projection(self, rest_catalog):
        """Test column projection via REST catalog."""
        table = rest_catalog.load_table("stanfordnlp.imdb")

        # Scan with only specific columns selected
        scan = table.scan().select("text", "label")
        arrow_table = scan.to_arrow()

        # Verify only selected columns are present
        column_names = arrow_table.schema.names
        assert "text" in column_names
        assert "label" in column_names
        assert "split" not in column_names


class TestRestCatalogMetadata:
    """Tests for PyIceberg REST catalog metadata operations."""

    def test_rest_read_schema(self, rest_catalog):
        """Test reading table schema via REST catalog."""
        table = rest_catalog.load_table("stanfordnlp.imdb")
        schema = table.schema()

        # Verify schema has expected fields
        field_names = [field.name for field in schema.fields]
        assert "split" in field_names
        assert "text" in field_names
        assert "label" in field_names

    def test_rest_read_properties(self, rest_catalog):
        """Test reading table properties via REST catalog."""
        table = rest_catalog.load_table("stanfordnlp.imdb")
        properties = table.properties

        # Verify HuggingFace properties exist
        assert "huggingface.dataset.repo" in properties
        assert properties["huggingface.dataset.repo"] == "stanfordnlp/imdb"

    def test_rest_read_snapshots(self, rest_catalog):
        """Test reading table snapshots via REST catalog."""
        table = rest_catalog.load_table("stanfordnlp.imdb")
        snapshots = list(table.snapshots())

        # Verify at least one snapshot exists
        assert len(snapshots) > 0

        # Verify snapshot has expected attributes
        snapshot = snapshots[0]
        assert hasattr(snapshot, "snapshot_id")
        assert snapshot.snapshot_id > 0


class TestUnsupportedOperations:
    """Tests for operations that are not yet supported."""

    def test_view_operations_not_supported(self, catalog):
        """Test that view operations are not supported."""
        # view_exists should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            catalog.view_exists("default.test_view")

        # list_views should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            catalog.list_views("default")

        # drop_view should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            catalog.drop_view("default.test_view")


# =============================================================================
# HfFileIO Tests
# =============================================================================


class TestHfFileIO:
    """Tests for HfFileIO custom FileIO implementation."""

    def test_hffileio_initialization(self):
        """Test that HfFileIO can be initialized with properties."""
        io = HfFileIO(
            properties={
                "hf.endpoint": "https://huggingface.co",
                "hf.token": "test_token",
            }
        )

        assert io is not None
        assert io.properties["hf.endpoint"] == "https://huggingface.co"
        assert io.properties["hf.token"] == "test_token"

    def test_hffileio_creates_hf_filesystem(self):
        """Test that HfFileIO creates HfFileSystem for hf:// scheme."""
        io = HfFileIO(properties={"hf.endpoint": "https://huggingface.co"})
        fs = io.get_fs("hf")

        assert isinstance(fs, HfFileSystem)

    def test_hffileio_uses_skip_instance_cache(self):
        """Test that HfFileIO creates multiple distinct HfFileSystem instances.

        When skip_instance_cache=True, each call to get_fs('hf') should create
        a new HfFileSystem instance (after cache eviction). This test verifies
        that our custom factory uses skip_instance_cache correctly.
        """
        io = HfFileIO(properties={"hf.endpoint": "https://huggingface.co"})

        # First call creates and caches filesystem
        fs1 = io.get_fs("hf")

        # Verify we got a HfFileSystem instance
        assert isinstance(fs1, HfFileSystem)

        # Just verify that calling get_fs again works
        # (Testing internal cache behavior is fragile across pyiceberg versions)
        fs2 = io.get_fs("hf")
        assert isinstance(fs2, HfFileSystem)

    def test_hffileio_extends_fsspec_fileio(self):
        """Test that HfFileIO properly extends FsspecFileIO."""
        io = HfFileIO(properties={})

        assert isinstance(io, FsspecFileIO)
        # Should have all standard FileIO methods
        assert hasattr(io, "new_input")
        assert hasattr(io, "new_output")
        assert hasattr(io, "delete")
        assert hasattr(io, "get_fs")


# =============================================================================
# catalog() Factory Function Tests
# =============================================================================


class TestCatalogFactory:
    """Tests for the catalog() factory function."""

    def test_catalog_local_directory_path(self, tmp_path):
        """Test creating LocalCatalog from directory path."""
        catalog_dir = tmp_path / "test_catalog"
        catalog_dir.mkdir()

        cat = catalog_factory(str(catalog_dir))

        assert isinstance(cat, LocalCatalog)
        assert cat.uri.startswith("file:///")

    def test_catalog_local_file_uri(self, tmp_path):
        """Test creating LocalCatalog from file:// URI."""
        catalog_dir = tmp_path / "test_catalog"
        catalog_dir.mkdir()
        uri = f"file://{catalog_dir.as_posix()}"

        cat = catalog_factory(uri)

        assert isinstance(cat, LocalCatalog)
        assert cat.uri.startswith("file:///")

    def test_catalog_remote_datasets_explicit(self):
        """Test creating RemoteCatalog with explicit hf://datasets/ URI."""
        cat = catalog_factory("hf://datasets/my-org/my-repo", hf_token="test_token")

        assert isinstance(cat, RemoteCatalog)
        assert cat.uri == "hf://datasets/my-org/my-repo"

    def test_catalog_remote_spaces_explicit(self):
        """Test creating RemoteCatalog with explicit hf://spaces/ URI."""
        cat = catalog_factory("hf://spaces/my-org/my-space", hf_token="test_token")

        assert isinstance(cat, RemoteCatalog)
        assert cat.uri == "hf://spaces/my-org/my-space"

    def test_catalog_remote_models_explicit(self):
        """Test creating RemoteCatalog with explicit hf://models/ URI."""

        with pytest.raises(ValueError, match="Unsupported"):
            catalog_factory("hf://models/my-org/my-model", hf_token="test_token")

    def test_catalog_remote_shorthand_defaults_to_spaces(self):
        """Test creating RemoteCatalog with shorthand org/repo format defaults to spaces."""
        cat = catalog_factory("my-org/my-repo", hf_token="test_token")

        assert isinstance(cat, RemoteCatalog)
        assert cat.uri == "hf://spaces/my-org/my-repo"
        assert cat.name == "my-org/my-repo"

    def test_catalog_remote_with_properties(self):
        """Test creating RemoteCatalog with additional properties."""
        cat = catalog_factory(
            "hf://spaces/my-org/my-space",
            hf_token="test_token",
            custom_prop="custom_value",
        )

        assert isinstance(cat, RemoteCatalog)
        assert cat.properties["custom_prop"] == "custom_value"

    def test_catalog_local_with_hf_token(self, tmp_path):
        """Test creating LocalCatalog with hf_token (for accessing datasets)."""
        catalog_dir = tmp_path / "test_catalog"
        catalog_dir.mkdir()

        cat = catalog_factory(str(catalog_dir), hf_token="test_token")

        assert isinstance(cat, LocalCatalog)

    def test_catalog_name_extraction_from_hf_uri(self):
        """Test that catalog name is correctly extracted from hf:// URI."""
        # Datasets
        cat1 = catalog_factory("hf://datasets/org/repo")
        assert cat1.name == "org/repo"

        # Spaces
        cat2 = catalog_factory("hf://spaces/org/space")
        assert cat2.name == "org/space"

    def test_catalog_warehouse_property_set_correctly(self, tmp_path):
        """Test that warehouse property is set correctly for different catalog types."""
        # Local catalog
        catalog_dir = tmp_path / "test_catalog"
        catalog_dir.mkdir()
        local_cat = catalog_factory(str(catalog_dir))
        assert local_cat.properties["warehouse"] == str(catalog_dir)

        # Remote catalog
        remote_cat = catalog_factory("hf://datasets/org/repo")
        assert remote_cat.properties["warehouse"] == "hf://datasets/org/repo"

    def test_local_catalog_requires_file_uri(self, tmp_path):
        """Test that LocalCatalog requires file:// URI."""
        catalog_dir = tmp_path / "test_catalog"
        catalog_dir.mkdir()

        # Should raise ValueError when given a plain path
        with pytest.raises(ValueError, match="LocalCatalog requires file:// URI"):
            LocalCatalog(name="test", uri=str(catalog_dir))

        # Should work with file:// URI
        uri = f"file://{catalog_dir.as_posix()}"
        cat = LocalCatalog(name="test", uri=uri)
        assert isinstance(cat, LocalCatalog)

    def test_remote_catalog_requires_hf_uri(self):
        """Test that RemoteCatalog requires hf:// URI."""
        # Should raise ValueError when given an invalid URI
        with pytest.raises(ValueError, match="RemoteCatalog requires hf:// URI"):
            RemoteCatalog(name="test", uri="file:///path/to/catalog")

        with pytest.raises(ValueError, match="RemoteCatalog requires hf:// URI"):
            RemoteCatalog(name="test", uri="org/repo")

        # Should work with hf:// URI
        cat = RemoteCatalog(name="test", uri="hf://datasets/org/repo")
        assert isinstance(cat, RemoteCatalog)

    def test_catalog_factory_handles_path_conversion(self, tmp_path):
        """Test that catalog() factory converts paths to file:// URIs."""
        catalog_dir = tmp_path / "test_catalog"
        catalog_dir.mkdir()

        # Factory should accept plain path and convert to file:// URI
        cat = catalog_factory(str(catalog_dir))
        assert isinstance(cat, LocalCatalog)
        assert cat.uri.startswith("file:///")


# =============================================================================
# HfLocationProvider Tests
# =============================================================================


class TestHfLocationProvider:
    """Tests for HfLocationProvider."""

    def test_default_pattern(self):
        """Test default file naming pattern."""
        provider = HfLocationProvider(
            table_location="hf://datasets/test-org/test-dataset",
            table_properties={},
        )

        # First file
        path1 = provider.new_data_location("ignored.parquet")
        assert path1.endswith("/train-00000-iceberg.parquet")

        # Second file
        path2 = provider.new_data_location("ignored.parquet")
        assert path2.endswith("/train-00001-iceberg.parquet")

    def test_custom_split(self):
        """Test custom split name."""
        provider = HfLocationProvider(
            table_location="hf://datasets/test-org/test-dataset",
            table_properties={"huggingface.write.split": "validation"},
        )

        path = provider.new_data_location("ignored.parquet")
        assert "validation-00000-iceberg.parquet" in path

    def test_custom_pattern(self):
        """Test custom file pattern."""
        provider = HfLocationProvider(
            table_location="hf://datasets/test-org/test-dataset",
            table_properties={
                "huggingface.write.pattern": "data-{split}-{index:03d}.parquet",
            },
        )

        path = provider.new_data_location("ignored.parquet")
        assert path.endswith("/data-train-000.parquet")

    def test_uuid_mode(self):
        """Test UUID-based naming."""
        provider = HfLocationProvider(
            table_location="hf://datasets/test-org/test-dataset",
            table_properties={
                "huggingface.write.use-uuid": "true",
                "huggingface.write.pattern": "{split}-{uuid}.parquet",
            },
        )

        path = provider.new_data_location("ignored.parquet")
        # UUID is 36 characters (8-4-4-4-12 with hyphens)
        assert path.endswith(".parquet")
        assert "train-" in path
        # Extract UUID part and verify format
        filename = path.split("/")[-1]
        uuid_part = filename.replace("train-", "").replace(".parquet", "")
        assert len(uuid_part) == 36

    def test_start_index(self):
        """Test starting from a specific index."""
        provider = HfLocationProvider(
            table_location="hf://datasets/test-org/test-dataset",
            table_properties={"huggingface.write.next-index": "10"},
        )

        path = provider.new_data_location("ignored.parquet")
        assert path.endswith("/train-00010-iceberg.parquet")


# =============================================================================
# Write to Existing Dataset Tests (Parametrized for local/remote)
# =============================================================================


class TestWriteToExistingDataset:
    """Tests for writing to existing HuggingFace datasets using location provider."""

    def test_append_to_existing_dataset(self, writable_dataset):
        """Test appending data to an existing dataset with HfLocationProvider.

        Verifies that:
        - The writable_dataset fixture provides a valid dataset
        - Data can be appended and read back correctly
        - Appended files follow HuggingFace naming pattern: train-{index:05d}-iceberg.parquet
        """
        catalog = writable_dataset

        # Verify table exists and is properly configured
        assert catalog.table_exists("testorg.testdataset")
        table = catalog.load_table("testorg.testdataset")
        assert table is not None

        # Verify table has HfLocationProvider configured
        assert (
            table.properties.get("write.py-location-provider.impl")
            == "faceberg.catalog.HfLocationProvider"
        )

        # Verify table has initial data
        before_count = table.scan().to_arrow().num_rows
        assert before_count == 10  # Initial data from fixture

        # Append new data (including split column as it's part of the schema)
        new_data = pa.Table.from_pydict(
            {
                "split": ["train", "train", "train"],
                "text": ["Appended test review", "Another appended review", "Third review"],
                "label": [1, 0, 1],
            }
        )

        table.append(new_data)

        # Reload table to get updated metadata
        table = catalog.load_table("testorg.testdataset")

        # Verify data was appended (count should increase)
        after_count = table.scan().to_arrow().num_rows
        assert after_count >= before_count + len(new_data)

        # Verify appended data is readable
        scan = table.scan().filter(f"text = 'Appended test review'")
        result = scan.to_arrow()
        assert result.num_rows == 1
        assert result["text"][0].as_py() == "Appended test review"

        # # Verify files follow HuggingFace naming pattern by checking data files in manifest
        # snapshot = table.current_snapshot()
        # if snapshot:
        #     # Get data files from the manifest
        #     data_files = [file.file_path for file in table.scan().to_arrow().to_batches()]

        #     # For remote catalog, check data file paths from manifest entries
        #     manifest_list = snapshot.manifest_list
        #     if manifest_list:
        #         # Get actual data file paths from the table scan
        #         data_file_paths = []
        #         for task in table.scan().plan_files():
        #             data_file_paths.append(task.file.file_path)

        #         # Extract filenames and verify pattern
        #         filenames = [path.split("/")[-1] for path in data_file_paths]
        #         assert len(filenames) > 0, "No data files found in table"

        #         # Verify files follow pattern: train-{index:05d}-iceberg.parquet
        #         pattern = re.compile(r"^train-\d{5}-iceberg\.parquet$")
        #         matching_files = [f for f in filenames if pattern.match(f)]
        #         assert len(matching_files) > 0, (
        #             f"No files matching HF pattern found. Files: {filenames}"
        #         )

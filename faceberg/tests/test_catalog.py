"""Tests for FacebergCatalog implementation."""

import json

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from pyiceberg.exceptions import (
    NamespaceAlreadyExistsError,
    NamespaceNotEmptyError,
    NoSuchTableError,
    TableAlreadyExistsError,
)
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType

from faceberg.catalog import LocalCatalog
from faceberg.config import CatalogConfig, NamespaceConfig, TableConfig


@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary test directory."""
    catalog_dir = tmp_path / "test_catalog"
    catalog_dir.mkdir()
    yield catalog_dir
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def catalog(test_dir):
    """Create a test catalog."""
    from faceberg.config import CatalogConfig

    # Create minimal config for testing basic catalog functionality
    minimal_config = CatalogConfig(
        name="test_catalog",
        location=str(test_dir),
        namespaces={},
    )
    return LocalCatalog(
        name="test_catalog",
        location=str(test_dir),
        config=minimal_config,
    )


@pytest.fixture
def test_schema():
    """Create a test schema."""
    return Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "name", StringType(), required=False),
    )


def test_create_catalog(test_dir):
    """Test catalog creation."""
    from faceberg.config import CatalogConfig

    minimal_config = CatalogConfig(
        name="test",
        location=str(test_dir),
        namespaces={},
    )
    catalog = LocalCatalog(name="test", location=str(test_dir), config=minimal_config)

    assert catalog.name == "test"
    assert catalog.catalog_dir == test_dir
    assert test_dir.exists()
    # Note: catalog.json is created after first operation, not at init


def test_create_namespace(catalog):
    """Test namespace creation."""
    catalog.create_namespace("default")

    # Note: Namespace directories are implicit in JSON catalog (from table names)
    # Empty namespaces don't create physical directories until tables are added


def test_list_namespaces_empty(catalog):
    """Test listing namespaces when none exist."""
    namespaces = catalog.list_namespaces()
    assert namespaces == []


def test_create_table(catalog, test_schema):
    """Test table creation."""
    catalog.create_namespace("default")

    table = catalog.create_table(
        identifier="default.test_table",
        schema=test_schema,
    )

    assert table.metadata is not None
    assert table.schema() == test_schema


def test_load_table(catalog, test_schema):
    """Test loading a table."""
    catalog.create_namespace("default")
    catalog.create_table(
        identifier="default.test_table",
        schema=test_schema,
    )

    table = catalog.load_table("default.test_table")

    assert table.schema() == test_schema


def test_list_tables(catalog, test_schema):
    """Test listing tables."""
    catalog.create_namespace("default")

    # Create multiple tables
    catalog.create_table("default.table1", test_schema)
    catalog.create_table("default.table2", test_schema)

    tables = catalog.list_tables("default")

    assert len(tables) == 2
    assert ("default", "table1") in tables
    assert ("default", "table2") in tables


def test_table_exists(catalog, test_schema):
    """Test checking table existence."""
    catalog.create_namespace("default")

    assert not catalog.table_exists("default.test_table")

    catalog.create_table("default.test_table", test_schema)

    assert catalog.table_exists("default.test_table")


def test_drop_table(catalog, test_schema):
    """Test dropping a table."""
    catalog.create_namespace("default")
    catalog.create_table("default.test_table", test_schema)

    assert catalog.table_exists("default.test_table")

    catalog.drop_table("default.test_table")

    assert not catalog.table_exists("default.test_table")


def test_rename_table(catalog, test_schema):
    """Test renaming a table."""
    catalog.create_namespace("default")
    catalog.create_table("default.old_name", test_schema)

    catalog.rename_table("default.old_name", "default.new_name")

    assert not catalog.table_exists("default.old_name")
    assert catalog.table_exists("default.new_name")


def test_catalog_persistence(test_dir, test_schema):
    """Test that catalog persists across instances."""
    from faceberg.config import CatalogConfig

    minimal_config = CatalogConfig(
        name="test",
        location=str(test_dir),
        namespaces={},
    )

    # Create catalog and table
    catalog1 = LocalCatalog("test", str(test_dir), config=minimal_config)
    catalog1.create_namespace("default")
    catalog1.create_table("default.test_table", test_schema)
    # Changes are automatically persisted via context manager

    # Create new catalog instance
    catalog2 = LocalCatalog("test", str(test_dir), config=minimal_config)

    # Table should still exist
    assert catalog2.table_exists("default.test_table")
    table = catalog2.load_table("default.test_table")
    assert table.schema() == test_schema


def test_catalog_json_format(catalog, test_schema):
    """Test catalog.json format."""
    catalog.create_namespace("default")
    catalog.create_table("default.test_table", test_schema)

    catalog_file = catalog.catalog_dir / "catalog.json"
    assert catalog_file.exists()

    with open(catalog_file) as f:
        data = json.load(f)

    # Check new catalog.json format
    assert "type" in data
    assert data["type"] == "local"
    assert "uri" in data
    assert data["uri"].startswith("file:///")
    assert "tables" in data
    assert "default.test_table" in data["tables"]
    assert isinstance(data["tables"]["default.test_table"], str)


# =============================================================================
# FacebergCatalog Tests
# =============================================================================


@pytest.fixture
def faceberg_test_dir(tmp_path):
    """Create temporary test directory for FacebergCatalog."""
    return tmp_path / "faceberg_test"


@pytest.fixture
def faceberg_config(faceberg_test_dir):
    """Create test CatalogConfig."""
    return CatalogConfig(
        name="test_catalog",
        location=str(faceberg_test_dir),
        namespaces=[
            NamespaceConfig(
                name="default",
                tables=[
                    TableConfig(
                        name="imdb_plain_text", dataset="stanfordnlp/imdb", config="plain_text"
                    ),
                ]
            )
        ],
    )


@pytest.fixture
def faceberg_config_file(tmp_path, faceberg_test_dir):
    """Create test config YAML file."""
    config_file = tmp_path / "test_faceberg.yml"
    config_content = f"""catalog:
  name: test_catalog
  location: {faceberg_test_dir}

default:
  imdb_plain_text:
    dataset: stanfordnlp/imdb
    config: plain_text
"""
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def faceberg_catalog(faceberg_config_file):
    """Create test LocalCatalog for Faceberg tests."""
    config = CatalogConfig.from_yaml(faceberg_config_file)
    return LocalCatalog(
        name=config.name,
        location=config.location,
        config=config,
    )


def test_faceberg_from_local(faceberg_config_file, faceberg_test_dir):
    """Test creating LocalCatalog from local config file."""
    config = CatalogConfig.from_yaml(faceberg_config_file)
    catalog = LocalCatalog(
        name=config.name,
        location=config.location,
        config=config,
    )

    assert catalog.name == "test_catalog"
    assert catalog.catalog_dir == faceberg_test_dir


def test_faceberg_lazy_namespace_creation(faceberg_catalog):
    """Test that namespaces are created on-demand when tables are created."""
    # Namespace should not exist yet
    assert ("default",) not in faceberg_catalog.list_namespaces()

    # Sync will create namespace on-demand
    synced_tables = faceberg_catalog.sync(token=None, table_name=None)

    # Namespace should now exist
    assert ("default",) in faceberg_catalog.list_namespaces()
    assert len(synced_tables) > 0


def test_faceberg_create_tables_from_datasets(faceberg_catalog, faceberg_config):
    """Test creating tables from datasets in FacebergCatalog."""
    # Sync tables (token=None works for public datasets, namespaces created on-demand)
    synced_tables = faceberg_catalog.sync(
        token=None,
        table_name=None,
    )

    # Verify tables were created
    assert len(synced_tables) > 0

    # Verify table was created in catalog
    tables = faceberg_catalog.list_tables("default")
    assert len(tables) > 0

    # Should have table for imdb dataset
    table_names = [f"{ns}.{table}" for ns, table in tables]
    assert any("imdb" in name for name in table_names)


def test_faceberg_create_specific_table(faceberg_catalog, faceberg_config):
    """Test creating a specific table in FacebergCatalog."""
    # Sync specific table (token=None works for public datasets, namespace created on-demand)
    synced_tables = faceberg_catalog.sync(
        token=None,
        table_name="default.imdb_plain_text",
    )

    # Verify table was synced
    assert len(synced_tables) == 1

    # Verify table exists
    assert faceberg_catalog.table_exists("default.imdb_plain_text")


def test_faceberg_create_table_already_exists(faceberg_catalog, faceberg_config):
    """Test creating a table that already exists raises error in FacebergCatalog."""
    from faceberg.bridge import DatasetInfo

    # Discover dataset (token=None works for public datasets)
    dataset_info = DatasetInfo.discover(
        repo_id="stanfordnlp/imdb",
        configs=["plain_text"],
        token=None,
    )

    # Convert to TableInfo
    table_info = dataset_info.to_table_info(
        namespace="default",
        table_name="imdb_plain_text",
        config="plain_text",
        token=None,
    )

    # Create table first time
    faceberg_catalog._create_table(table_info)

    # Try to create again - should raise
    with pytest.raises(TableAlreadyExistsError):
        faceberg_catalog._create_table(table_info)


def test_faceberg_create_table_for_config(faceberg_catalog, faceberg_config):
    """Test creating a table for a specific config in FacebergCatalog."""
    from faceberg.bridge import DatasetInfo

    # Discover dataset (token=None works for public datasets, namespace created on-demand)
    dataset_info = DatasetInfo.discover(
        repo_id="stanfordnlp/imdb",
        configs=["plain_text"],
        token=None,
    )

    # Convert to TableInfo
    table_info = dataset_info.to_table_info(
        namespace="default",
        table_name="imdb_plain_text",
        config="plain_text",
        token=None,
    )

    # Create table
    table = faceberg_catalog._create_table(table_info)

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


def test_faceberg_invalid_table_name_format(faceberg_catalog, faceberg_config):
    """Test invalid table name format raises error in FacebergCatalog."""
    with pytest.raises(ValueError, match="Invalid table name"):
        faceberg_catalog.sync(
            token=None,
            table_name="invalid_format",  # Missing namespace
        )


def test_faceberg_dataset_not_found_in_config(faceberg_catalog):
    """Test error when dataset not found in config in FacebergCatalog."""
    # Catalog config has "imdb" dataset, so "nonexistent" should fail
    with pytest.raises(ValueError, match="not found in config"):
        faceberg_catalog.sync(
            token=None,
            table_name="default.nonexistent_default",
        )


# =============================================================================
# Additional LocalCatalog Tests for Better Coverage
# =============================================================================


def test_drop_namespace(catalog, test_schema):
    """Test dropping an empty namespace."""
    catalog.create_namespace("test_ns")
    catalog.drop_namespace("test_ns")

    # Namespace should not appear in list
    assert ("test_ns",) not in catalog.list_namespaces()


def test_drop_namespace_not_empty(catalog, test_schema):
    """Test that dropping a non-empty namespace raises error."""
    catalog.create_namespace("test_ns")
    catalog.create_table("test_ns.table1", test_schema)

    with pytest.raises(NamespaceNotEmptyError):
        catalog.drop_namespace("test_ns")


def test_load_namespace_properties(catalog):
    """Test loading namespace properties."""
    catalog.create_namespace("test_ns")
    props = catalog.load_namespace_properties("test_ns")

    # Currently returns empty dict
    assert props == {}


def test_update_namespace_properties(catalog):
    """Test updating namespace properties."""
    catalog.create_namespace("test_ns")
    summary = catalog.update_namespace_properties(
        "test_ns",
        removals={"old_prop"},
        updates={"new_prop": "value"}
    )

    # Currently returns empty summary
    assert summary.removed == []
    assert summary.updated == []
    assert summary.missing == []


def test_register_table(catalog, test_schema):
    """Test registering an existing table."""
    # Create a table first
    catalog.create_namespace("default")
    table = catalog.create_table("default.source_table", test_schema)

    # Get the metadata location
    metadata_location = table.metadata_location

    # Register it under a different name
    registered_table = catalog.register_table("default.registered_table", metadata_location)

    assert registered_table.metadata_location == metadata_location
    assert catalog.table_exists("default.registered_table")


def test_register_table_already_exists(catalog, test_schema):
    """Test that registering a table that already exists raises error."""
    catalog.create_namespace("default")
    table = catalog.create_table("default.test_table", test_schema)

    with pytest.raises(TableAlreadyExistsError):
        catalog.register_table("default.test_table", table.metadata_location)


def test_purge_table(catalog, test_schema):
    """Test purging a table."""
    catalog.create_namespace("default")
    catalog.create_table("default.test_table", test_schema)

    # Purge should remove the table
    catalog.purge_table("default.test_table")

    assert not catalog.table_exists("default.test_table")


def test_view_operations(catalog):
    """Test that view operations are not supported."""
    # view_exists should return False
    assert not catalog.view_exists("default.test_view")

    # list_views should return empty list
    assert catalog.list_views("default") == []

    # drop_view should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        catalog.drop_view("default.test_view")


def test_create_table_transaction_not_implemented(catalog, test_schema):
    """Test that table transactions are not yet implemented."""
    with pytest.raises(NotImplementedError):
        catalog.create_table_transaction("default.test_table", test_schema)


def test_commit_table_not_implemented(catalog):
    """Test that commit_table is not yet implemented."""
    from unittest.mock import MagicMock

    from pyiceberg.table import CommitTableRequest

    # Use a mock request - we just need to test that NotImplementedError is raised
    mock_request = MagicMock(spec=CommitTableRequest)

    with pytest.raises(NotImplementedError):
        catalog.commit_table(mock_request)


def test_identifier_to_str_with_tuple(catalog):
    """Test converting tuple identifier to string."""
    result = catalog._identifier_to_str(("namespace", "table"))
    assert result == "namespace.table"


def test_identifier_to_str_with_string(catalog):
    """Test that string identifiers pass through unchanged."""
    result = catalog._identifier_to_str("namespace.table")
    assert result == "namespace.table"


def test_get_metadata_location_outside_staging_context(catalog):
    """Test that _get_metadata_location raises error outside staging context."""
    with pytest.raises(RuntimeError, match="must be called within _staging\\(\\) context"):
        catalog._get_metadata_location("default.test_table")


def test_save_catalog_outside_staging_context(catalog):
    """Test that _save_catalog raises error outside staging context."""
    with pytest.raises(RuntimeError, match="must be called within _staging\\(\\) context"):
        catalog._save_catalog()


def test_gather_changes_outside_staging_context(catalog):
    """Test that _gather_changes raises error outside staging context."""
    with pytest.raises(RuntimeError, match="must be called within _staging\\(\\) context"):
        catalog._gather_changes()


def test_persist_changes_outside_staging_context(catalog):
    """Test that _persist_changes raises error outside staging context."""
    with pytest.raises(RuntimeError, match="must be called within _staging\\(\\) context"):
        catalog._persist_changes()


def test_invalid_table_identifier_format(catalog):
    """Test that invalid table identifier raises error."""
    with pytest.raises(ValueError, match="Invalid table identifier"):
        with catalog._staging():
            catalog._get_metadata_location("invalid_no_dot")


def test_load_table_not_found(catalog):
    """Test loading non-existent table raises error."""
    with pytest.raises(NoSuchTableError):
        catalog.load_table("default.nonexistent")


def test_load_table_metadata_file_not_found(catalog, test_schema, test_dir):
    """Test error when metadata file is missing."""
    catalog.create_namespace("default")
    catalog.create_table("default.test_table", test_schema)

    # Manually corrupt the catalog to point to non-existent file
    catalog_file = catalog.catalog_dir / "catalog.json"
    with open(catalog_file) as f:
        data = json.load(f)

    # Point to non-existent metadata file (update tables dict)
    data["tables"]["default.test_table"] = "default/test_table/metadata/nonexistent.metadata.json"

    with open(catalog_file, "w") as f:
        json.dump(data, f)

    # Reload catalog to pick up the corrupted data
    catalog._load_catalog()

    # Should raise NoSuchTableError
    with pytest.raises(NoSuchTableError, match="metadata file not found"):
        catalog.load_table("default.test_table")


def test_create_namespace_already_exists(catalog, test_schema):
    """Test creating namespace that already exists (has tables)."""
    catalog.create_namespace("test_ns")
    catalog.create_table("test_ns.table1", test_schema)

    with pytest.raises(NamespaceAlreadyExistsError):
        catalog.create_namespace("test_ns")


def test_drop_table_not_found(catalog):
    """Test dropping non-existent table raises error."""
    with pytest.raises(NoSuchTableError):
        catalog.drop_table("default.nonexistent")


def test_rename_table_source_not_found(catalog):
    """Test renaming non-existent table raises error."""
    with pytest.raises(NoSuchTableError):
        catalog.rename_table("default.nonexistent", "default.new_name")


def test_rename_table_destination_exists(catalog, test_schema):
    """Test that renaming to existing table name raises error."""
    catalog.create_namespace("default")
    catalog.create_table("default.table1", test_schema)
    catalog.create_table("default.table2", test_schema)

    with pytest.raises(TableAlreadyExistsError):
        catalog.rename_table("default.table1", "default.table2")


def test_list_namespaces_with_multi_level(catalog, test_schema):
    """Test listing namespaces with hierarchical names."""
    catalog.create_namespace("ns1")
    catalog.create_table("ns1.table1", test_schema)

    namespaces = catalog.list_namespaces()
    assert ("ns1",) in namespaces


def test_staging_context_cleanup_on_error(catalog, test_schema):
    """Test that staging context cleans up even on error."""
    catalog.create_namespace("default")

    try:
        with catalog._staging():
            # Try to create duplicate table (will fail)
            catalog._catalog["default.test_table"] = "some/path"
            raise ValueError("Simulated error")
    except ValueError:
        pass

    # Staging should be cleaned up
    assert catalog._staging_dir is None
    assert catalog._catalog is None
    assert catalog._old_catalog is None


def test_gather_changes_with_new_and_deleted_tables(catalog, test_schema):
    """Test _gather_changes detects both additions and deletions."""
    catalog.create_namespace("default")
    catalog.create_table("default.table1", test_schema)
    catalog.create_table("default.table2", test_schema)

    # Now drop table1 and add table3
    with catalog._staging():
        # Simulate old catalog having table1 and table2
        catalog._old_catalog = {"default.table1": "path1", "default.table2": "path2"}
        # New catalog has table2 and table3
        catalog._catalog = {"default.table2": "path2", "default.table3": "path3"}

        # Create dummy metadata file for table3 in staging
        table3_path = catalog._staging_dir / "default" / "table3" / "metadata"
        table3_path.mkdir(parents=True, exist_ok=True)
        metadata_file = table3_path / "v0.metadata.json"
        metadata_file.write_text('{"test": "data"}')

        # Save catalog to staging
        catalog._save_catalog()

        # Gather changes
        changes = catalog._gather_changes()

        # Should have:
        # 1. catalog.json (Add)
        # 2. table3 metadata file (Add)
        # 3. table1 directory (Delete)
        add_ops = [op for op in changes if isinstance(op, CommitOperationAdd)]
        delete_ops = [op for op in changes if isinstance(op, CommitOperationDelete)]

        assert len(add_ops) >= 2  # catalog.json + table3 metadata
        assert len(delete_ops) == 1  # table1 deleted
        assert delete_ops[0].path_in_repo == "default/table1/"

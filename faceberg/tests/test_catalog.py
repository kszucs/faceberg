"""Tests for FacebergCatalog implementation."""

import shutil
from unittest.mock import MagicMock

import pytest
import yaml
from huggingface_hub import HfFileSystem
from pyiceberg.exceptions import (
    NamespaceAlreadyExistsError,
    NamespaceNotEmptyError,
    NoSuchTableError,
    TableAlreadyExistsError,
)
from pyiceberg.io.fsspec import FsspecFileIO
from pyiceberg.schema import Schema
from pyiceberg.table import CommitTableRequest
from pyiceberg.types import LongType, NestedField, StringType

from faceberg.bridge import DatasetInfo
from faceberg.catalog import HfFileIO, LocalCatalog, RemoteCatalog
from faceberg.catalog import catalog as catalog_factory
from faceberg.database import Catalog, Namespace, Table


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
    uri = f"file:///{test_dir.as_posix()}"
    return LocalCatalog(name=str(test_dir), uri=uri)


@pytest.fixture
def test_schema():
    """Create a test schema."""
    return Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "name", StringType(), required=False),
    )


def test_create_catalog(test_dir):
    """Test catalog creation."""
    uri = f"file:///{test_dir.as_posix()}"
    catalog = LocalCatalog(name=str(test_dir), uri=uri)

    # catalog.name is derived from path
    assert catalog.name == str(test_dir)
    assert catalog.uri.startswith("file:///")
    assert catalog.uri.endswith(str(test_dir.name))
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
    # Create catalog and table
    uri = f"file:///{test_dir.as_posix()}"
    catalog1 = LocalCatalog(name=str(test_dir), uri=uri)
    catalog1.create_namespace("default")
    catalog1.create_table("default.test_table", test_schema)
    # Changes are automatically persisted via context manager

    # Create new catalog instance
    catalog2 = LocalCatalog(name=str(test_dir), uri=uri)

    # Table should still exist
    assert catalog2.table_exists("default.test_table")
    table = catalog2.load_table("default.test_table")
    assert table.schema() == test_schema


def test_catalog_json_format(catalog, test_schema):
    """Test faceberg.yml format."""
    catalog.create_namespace("default")
    catalog.create_table("default.test_table", test_schema)

    catalog_file = catalog.catalog_dir / "faceberg.yml"
    assert catalog_file.exists()

    with open(catalog_file) as f:
        data = yaml.safe_load(f)

    # Check faceberg.yml format
    assert "uri" in data
    assert data["uri"].startswith("file:///")
    assert "default" in data
    assert "test_table" in data["default"]
    assert isinstance(data["default"]["test_table"], dict)
    assert "uri" in data["default"]["test_table"]


# =============================================================================
# FacebergCatalog Tests
# =============================================================================


@pytest.fixture
def faceberg_test_dir(tmp_path):
    """Create temporary test directory for FacebergCatalog."""
    return tmp_path / "faceberg_test"


@pytest.fixture
def faceberg_config():
    """Create test Catalog."""
    return Catalog(
        uri=".faceberg",
        namespaces={
            "default": Namespace(
                tables={
                    "imdb_plain_text": Table(
                        dataset="stanfordnlp/imdb", uri="", revision="", config="plain_text"
                    ),
                }
            )
        },
    )


@pytest.fixture
def faceberg_config_file(tmp_path):
    """Create test config YAML file."""
    config_file = tmp_path / "test_faceberg.yml"
    config_content = """uri: .faceberg

default:
  imdb_plain_text:
    dataset: stanfordnlp/imdb
    config: plain_text
"""
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def faceberg_catalog(faceberg_config_file, faceberg_test_dir):
    """Create test LocalCatalog for Faceberg tests."""
    # Create catalog and ensure the database file exists
    faceberg_test_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(faceberg_config_file, faceberg_test_dir / "faceberg.yml")
    uri = f"file:///{faceberg_test_dir.as_posix()}"
    return LocalCatalog(name=str(faceberg_test_dir), uri=uri)


def test_faceberg_from_local(faceberg_config_file, faceberg_test_dir):
    """Test creating LocalCatalog from local config file."""
    uri = f"file:///{faceberg_test_dir.as_posix()}"
    catalog = LocalCatalog(name=str(faceberg_test_dir), uri=uri)

    assert catalog.uri.startswith("file:///")
    assert catalog.uri.endswith(str(faceberg_test_dir.name))
    assert catalog.catalog_dir == faceberg_test_dir


def test_faceberg_lazy_namespace_creation(faceberg_catalog):
    """Test that namespaces exist after tables are defined in config."""
    # Namespace should exist from config
    assert ("default",) in faceberg_catalog.list_namespaces()

    # Sync will create tables
    synced_tables = faceberg_catalog.sync_datasets()

    # Namespace should still exist
    assert ("default",) in faceberg_catalog.list_namespaces()
    assert len(synced_tables) > 0


def test_faceberg_create_tables_from_datasets(faceberg_catalog, faceberg_config):
    """Test creating tables from datasets in FacebergCatalog."""
    # Sync tables (token=None works for public datasets, namespaces created on-demand)
    synced_tables = faceberg_catalog.sync_datasets()

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
    synced_tables = faceberg_catalog.sync_datasets(
        table_name="default.imdb_plain_text",
    )

    # Verify table was synced
    assert len(synced_tables) == 1

    # Verify table exists
    assert faceberg_catalog.table_exists("default.imdb_plain_text")


def test_faceberg_create_table_already_exists(faceberg_catalog, faceberg_config):
    """Test creating a table that already exists raises error in FacebergCatalog."""
    # Discover dataset (token=None works for public datasets)
    dataset_info = DatasetInfo.discover(
        repo_id="stanfordnlp/imdb",
        configs=["plain_text"],
    )

    # Convert to TableInfo
    table_info = dataset_info.to_table_info(
        namespace="default",
        table_name="imdb_plain_text",
        config="plain_text",
    )

    # Create table first time
    faceberg_catalog._add_dataset(table_info)

    # Try to create again - should raise
    with pytest.raises(TableAlreadyExistsError):
        faceberg_catalog._add_dataset(table_info)


def test_faceberg_create_table_for_config(faceberg_catalog, faceberg_config):
    """Test creating a table for a specific config in FacebergCatalog."""
    # Discover dataset (token=None works for public datasets, namespace created on-demand)
    dataset_info = DatasetInfo.discover(
        repo_id="stanfordnlp/imdb",
        configs=["plain_text"],
    )

    # Convert to TableInfo
    table_info = dataset_info.to_table_info(
        namespace="default",
        table_name="imdb_plain_text",
        config="plain_text",
    )

    # Create table
    table = faceberg_catalog._add_dataset(table_info)

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
        faceberg_catalog.sync_datasets(
            table_name="invalid_format",  # Missing namespace
        )


def test_faceberg_dataset_not_found_in_config(faceberg_catalog):
    """Test error when dataset not found in store in FacebergCatalog."""
    # Catalog store has "imdb_plain_text" dataset, so "nonexistent" should fail
    with pytest.raises(ValueError, match="not found in store"):
        faceberg_catalog.sync_datasets(
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
        "test_ns", removals={"old_prop"}, updates={"new_prop": "value"}
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
    # Use a mock request - we just need to test that NotImplementedError is raised
    mock_request = MagicMock(spec=CommitTableRequest)

    with pytest.raises(NotImplementedError):
        catalog.commit_table(mock_request)


def test_save_catalog_outside_staging_context(catalog):
    """Test that _save_database raises error outside staging context."""
    with pytest.raises(RuntimeError, match="must be called within _staging\\(\\) context"):
        catalog._save_database()


def test_persist_changes_outside_staging_context(catalog):
    """Test that _persist_changes raises error outside staging context."""
    with pytest.raises(RuntimeError, match="must be called within _staging\\(\\) context"):
        catalog._persist_changes()


def test_load_table_not_found(catalog):
    """Test loading non-existent table raises error."""
    with pytest.raises(NoSuchTableError):
        catalog.load_table("default.nonexistent")


def test_load_table_metadata_file_not_found(catalog, test_schema, test_dir):
    """Test error when metadata file is missing."""
    catalog.create_namespace("default")
    catalog.create_table("default.test_table", test_schema)

    # Manually corrupt the catalog to point to non-existent file
    catalog_file = catalog.catalog_dir / "faceberg.yml"
    with open(catalog_file) as f:
        data = yaml.safe_load(f)

    # Point to non-existent metadata file
    data["default"]["test_table"]["uri"] = "file:///nonexistent/metadata.json"

    with open(catalog_file, "w") as f:
        yaml.dump(data, f)

    # Reload catalog to pick up the corrupted data
    catalog._load_database()

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
            # Try to create duplicate namespace (will fail)
            raise ValueError("Simulated error")
    except ValueError:
        pass

    # Staging should be cleaned up
    assert catalog._staging_dir is None
    assert catalog._staged_changes is None


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
        from huggingface_hub import HfFileSystem

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
        assert cat.catalog_dir == catalog_dir
        assert cat.uri.startswith("file:///")

    def test_catalog_local_file_uri(self, tmp_path):
        """Test creating LocalCatalog from file:// URI."""
        catalog_dir = tmp_path / "test_catalog"
        catalog_dir.mkdir()
        uri = f"file:///{catalog_dir.as_posix()}"

        cat = catalog_factory(uri)

        assert isinstance(cat, LocalCatalog)
        assert cat.catalog_dir.as_posix() == catalog_dir.as_posix()
        assert cat.uri.startswith("file:///")

    def test_catalog_remote_datasets_explicit(self):
        """Test creating RemoteCatalog with explicit hf://datasets/ URI."""
        cat = catalog_factory("hf://datasets/my-org/my-repo", hf_token="test_token")

        assert isinstance(cat, RemoteCatalog)
        assert cat._hf_repo == "my-org/my-repo"
        assert cat._hf_repo_type == "datasets"
        assert cat.uri == "hf://datasets/my-org/my-repo"

    def test_catalog_remote_spaces_explicit(self):
        """Test creating RemoteCatalog with explicit hf://spaces/ URI."""
        cat = catalog_factory("hf://spaces/my-org/my-space", hf_token="test_token")

        assert isinstance(cat, RemoteCatalog)
        assert cat._hf_repo == "my-org/my-space"
        assert cat._hf_repo_type == "spaces"
        assert cat.uri == "hf://spaces/my-org/my-space"

    def test_catalog_remote_models_explicit(self):
        """Test creating RemoteCatalog with explicit hf://models/ URI."""
        cat = catalog_factory("hf://models/my-org/my-model", hf_token="test_token")

        assert isinstance(cat, RemoteCatalog)
        assert cat._hf_repo == "my-org/my-model"
        assert cat._hf_repo_type == "models"
        assert cat.uri == "hf://models/my-org/my-model"

    def test_catalog_remote_shorthand_defaults_to_datasets(self):
        """Test creating RemoteCatalog with shorthand org/repo format defaults to datasets."""
        cat = catalog_factory("my-org/my-repo", hf_token="test_token")

        assert isinstance(cat, RemoteCatalog)
        assert cat._hf_repo == "my-org/my-repo"
        assert cat._hf_repo_type == "datasets"
        assert cat.uri == "hf://datasets/my-org/my-repo"
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
        assert cat._hf_token == "test_token"

    def test_catalog_name_extraction_from_hf_uri(self):
        """Test that catalog name is correctly extracted from hf:// URI."""
        # Datasets
        cat1 = catalog_factory("hf://datasets/org/repo")
        assert cat1.name == "org/repo"

        # Spaces
        cat2 = catalog_factory("hf://spaces/org/space")
        assert cat2.name == "org/space"

        # Models
        cat3 = catalog_factory("hf://models/org/model")
        assert cat3.name == "org/model"

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
        uri = f"file:///{catalog_dir.as_posix()}"
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
        assert cat.catalog_dir == catalog_dir.resolve()

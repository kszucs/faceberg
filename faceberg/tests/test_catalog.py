"""Tests for JsonCatalog and FacebergCatalog implementations."""

import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pytest
from pyiceberg.exceptions import TableAlreadyExistsError
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType

from faceberg.catalog import FacebergCatalog, JsonCatalog
from faceberg.config import CatalogConfig, DatasetConfig, FacebergConfig


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
    return JsonCatalog(
        name="test_catalog",
        warehouse=str(test_dir),
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
    catalog = JsonCatalog(name="test", warehouse=str(test_dir))

    assert catalog.name == "test"
    assert catalog.warehouse == test_dir
    assert test_dir.exists()
    assert (test_dir / "catalog.json").exists()


def test_create_namespace(catalog):
    """Test namespace creation."""
    catalog.create_namespace("default")

    # Namespace directory should exist
    ns_dir = catalog.warehouse / "metadata" / "default"
    assert ns_dir.exists()


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
    catalog1 = JsonCatalog("test", str(test_dir))
    catalog1.create_namespace("default")
    catalog1.create_table("default.test_table", test_schema)

    # Create new catalog instance
    catalog2 = JsonCatalog("test", str(test_dir))

    # Table should still exist
    assert catalog2.table_exists("default.test_table")
    table = catalog2.load_table("default.test_table")
    assert table.schema() == test_schema


def test_catalog_json_format(catalog, test_schema):
    """Test catalog.json format."""
    catalog.create_namespace("default")
    catalog.create_table("default.test_table", test_schema)

    catalog_file = catalog.warehouse / "catalog.json"
    assert catalog_file.exists()

    with open(catalog_file) as f:
        data = json.load(f)

    assert "default.test_table" in data
    assert isinstance(data["default.test_table"], str)


# =============================================================================
# FacebergCatalog Tests
# =============================================================================


@pytest.fixture
def faceberg_test_dir(tmp_path):
    """Create temporary test directory for FacebergCatalog."""
    return tmp_path / "faceberg_test"


@pytest.fixture
def faceberg_config(faceberg_test_dir):
    """Create test FacebergConfig."""
    return FacebergConfig(
        catalog=CatalogConfig(name="test_catalog", location=str(faceberg_test_dir)),
        datasets=[
            DatasetConfig(name="imdb", repo="stanfordnlp/imdb", configs=None),
        ],
    )


@pytest.fixture
def faceberg_catalog(faceberg_config):
    """Create test FacebergCatalog."""
    return FacebergCatalog.from_config(faceberg_config)


def test_faceberg_from_config(faceberg_config, faceberg_test_dir):
    """Test creating FacebergCatalog from config."""
    catalog = FacebergCatalog.from_config(faceberg_config)

    assert catalog.name == "test_catalog"
    assert catalog.warehouse == faceberg_test_dir


def test_faceberg_initialize(faceberg_catalog):
    """Test initializing FacebergCatalog doesn't raise."""
    # Initialize should not raise even if namespace doesn't exist yet
    # (namespace is created implicitly when first table is created)
    faceberg_catalog.initialize()


def test_faceberg_initialize_idempotent(faceberg_catalog):
    """Test FacebergCatalog initialize can be called multiple times."""
    faceberg_catalog.initialize()
    faceberg_catalog.initialize()  # Should not raise


def test_faceberg_create_tables_from_datasets(faceberg_catalog, faceberg_config):
    """Test creating tables from datasets in FacebergCatalog."""
    # Skip if no HF_TOKEN (e.g., in CI)
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")

    # Initialize catalog
    faceberg_catalog.initialize()

    # Create tables
    faceberg_catalog.create_tables(
        token=os.getenv("HF_TOKEN"),
        table_name=None,
    )

    # Verify table was created
    tables = faceberg_catalog.list_tables("default")
    assert len(tables) > 0

    # Should have table for imdb dataset
    table_names = [f"{ns}.{table}" for ns, table in tables]
    assert any("imdb" in name for name in table_names)


def test_faceberg_create_specific_table(faceberg_catalog, faceberg_config):
    """Test creating a specific table in FacebergCatalog."""
    # Skip if no HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")

    faceberg_catalog.initialize()

    # Create specific table
    faceberg_catalog.create_tables(
        token=os.getenv("HF_TOKEN"),
        table_name="default.imdb_plain_text",
    )

    # Verify table exists
    assert faceberg_catalog.table_exists("default.imdb_plain_text")


def test_faceberg_create_table_already_exists(faceberg_catalog, faceberg_config):
    """Test creating a table that already exists raises error in FacebergCatalog."""
    # Skip if no HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")

    from faceberg.bridge import DatasetInfo

    faceberg_catalog.initialize()

    # Discover dataset
    dataset_info = DatasetInfo.discover(
        repo_id="stanfordnlp/imdb",
        token=os.getenv("HF_TOKEN"),
    )

    # Convert to TableInfo
    table_infos = dataset_info.to_table_infos(
        namespace="default",
        table_name_prefix="imdb",
        token=os.getenv("HF_TOKEN"),
    )

    # Get the plain_text config table info
    table_info = next((ti for ti in table_infos if ti.source_config == "plain_text"), None)
    assert table_info is not None

    # Create table first time
    faceberg_catalog._create_table_from_table_info(table_info)

    # Try to create again - should raise
    with pytest.raises(TableAlreadyExistsError):
        faceberg_catalog._create_table_from_table_info(table_info)


def test_faceberg_create_table_for_config(faceberg_catalog, faceberg_config):
    """Test creating a table for a specific config in FacebergCatalog."""
    # Skip if no HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")

    from faceberg.bridge import DatasetInfo

    faceberg_catalog.initialize()

    # Discover dataset
    dataset_info = DatasetInfo.discover(
        repo_id="stanfordnlp/imdb",
        token=os.getenv("HF_TOKEN"),
    )

    # Convert to TableInfo
    table_infos = dataset_info.to_table_infos(
        namespace="default",
        table_name_prefix="imdb",
        token=os.getenv("HF_TOKEN"),
    )

    # Get the plain_text config table info
    table_info = next((ti for ti in table_infos if ti.source_config == "plain_text"), None)
    assert table_info is not None

    # Create table
    table = faceberg_catalog._create_table_from_table_info(table_info)

    # Verify table
    assert table is not None
    assert table.schema() is not None
    assert len(table.schema().fields) > 0

    # Verify table properties
    props = table.properties
    assert "faceberg.source.repo" in props
    assert props["faceberg.source.repo"] == "stanfordnlp/imdb"
    assert "faceberg.source.config" in props
    assert props["faceberg.source.config"] == "plain_text"


def test_faceberg_invalid_table_name_format(faceberg_catalog, faceberg_config):
    """Test invalid table name format raises error in FacebergCatalog."""
    with pytest.raises(ValueError, match="Invalid table name"):
        faceberg_catalog.create_tables(
            token=None,
            table_name="invalid_format",  # Missing namespace
        )


def test_faceberg_dataset_not_found_in_config(faceberg_catalog):
    """Test error when dataset not found in config in FacebergCatalog."""
    # Catalog config has "imdb" dataset, so "nonexistent" should fail
    with pytest.raises(ValueError, match="not found in config"):
        faceberg_catalog.create_tables(
            token=None,
            table_name="default.nonexistent_default",
        )

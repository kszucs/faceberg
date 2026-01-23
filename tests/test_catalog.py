"""Tests for JsonCatalog implementation."""

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pytest
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType

from faceberg.catalog import JsonCatalog


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


# Quick manual test if run directly
if __name__ == "__main__":
    from pathlib import Path
    import shutil

    # Clean up any previous test data
    test_dir = Path(__file__).parent / ".test_catalog"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    print("Creating JsonCatalog...")
    catalog = JsonCatalog(
        name="test_catalog",
        warehouse=str(test_dir),
    )

    print("Creating namespace...")
    catalog.create_namespace("default")

    print("Defining schema...")
    schema = Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "name", StringType(), required=False),
    )

    print("Creating table...")
    table = catalog.create_table(
        identifier="default.test_table",
        schema=schema,
    )

    print(f"✓ Table created!")
    print(f"✓ Table location: {table.metadata.location}")

    print("\nListing tables...")
    tables = catalog.list_tables("default")
    print(f"✓ Tables in 'default': {tables}")

    print("\nLoading table...")
    loaded_table = catalog.load_table("default.test_table")
    print(f"✓ Table loaded!")
    print(f"✓ Schema: {loaded_table.schema()}")

    print("\nChecking catalog.json...")
    catalog_json = test_dir / "catalog.json"
    if catalog_json.exists():
        with open(catalog_json) as f:
            data = json.load(f)
        print(f"✓ Catalog contents: {json.dumps(data, indent=2)}")

    print("\n✅ All basic catalog operations work!")

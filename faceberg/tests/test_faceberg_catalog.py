"""Tests for FacebergCatalog."""

import os
from pathlib import Path

import pytest
from pyiceberg.exceptions import TableAlreadyExistsError
from pyiceberg.schema import Schema
from pyiceberg.types import IntegerType, LongType, NestedField, StringType

from faceberg.catalog import FacebergCatalog
from faceberg.config import CatalogConfig, DatasetConfig, FacebergConfig


@pytest.fixture
def test_dir(tmp_path):
    """Create temporary test directory."""
    return tmp_path / "test_catalog"


@pytest.fixture
def test_config(test_dir):
    """Create test FacebergConfig."""
    return FacebergConfig(
        catalog=CatalogConfig(name="test_catalog", location=str(test_dir)),
        datasets=[
            DatasetConfig(name="imdb", repo="stanfordnlp/imdb", configs=None),
        ],
    )


@pytest.fixture
def catalog(test_config):
    """Create test FacebergCatalog."""
    return FacebergCatalog.from_config(test_config)


def test_from_config(test_config, test_dir):
    """Test creating catalog from config."""
    catalog = FacebergCatalog.from_config(test_config)

    assert catalog.name == "test_catalog"
    assert catalog.warehouse == test_dir


def test_initialize(catalog):
    """Test initializing catalog doesn't raise."""
    # Initialize should not raise even if namespace doesn't exist yet
    # (namespace is created implicitly when first table is created)
    catalog.initialize()


def test_initialize_idempotent(catalog):
    """Test initialize can be called multiple times."""
    catalog.initialize()
    catalog.initialize()  # Should not raise


def test_create_tables_from_datasets(catalog, test_config):
    """Test creating tables from datasets."""
    # Skip if no HF_TOKEN (e.g., in CI)
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")

    # Initialize catalog
    catalog.initialize()

    # Create tables
    catalog.create_tables(
        token=os.getenv("HF_TOKEN"),
        table_name=None,
    )

    # Verify table was created
    tables = catalog.list_tables("default")
    assert len(tables) > 0

    # Should have table for imdb dataset
    table_names = [f"{ns}.{table}" for ns, table in tables]
    assert any("imdb" in name for name in table_names)


def test_create_specific_table(catalog, test_config):
    """Test creating a specific table."""
    # Skip if no HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")

    catalog.initialize()

    # Create specific table
    catalog.create_tables(
        token=os.getenv("HF_TOKEN"),
        table_name="default.imdb_plain_text",
    )

    # Verify table exists
    assert catalog.table_exists("default.imdb_plain_text")


def test_create_table_already_exists(catalog, test_config):
    """Test creating a table that already exists raises error."""
    # Skip if no HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")

    from faceberg.discovery import DatasetInfo

    catalog.initialize()

    # Discover dataset
    dataset_info = DatasetInfo.discover(
        repo_id="stanfordnlp/imdb",
        token=os.getenv("HF_TOKEN"),
    )

    # Create table first time
    catalog._create_table_for_config(
        dataset_config=test_config.datasets[0],
        dataset_info=dataset_info,
        config_name="plain_text",
    )

    # Try to create again - should raise
    with pytest.raises(TableAlreadyExistsError):
        catalog._create_table_for_config(
            dataset_config=test_config.datasets[0],
            dataset_info=dataset_info,
            config_name="plain_text",
        )


def test_create_table_for_config(catalog, test_config):
    """Test creating a table for a specific config."""
    # Skip if no HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")

    from faceberg.discovery import DatasetInfo

    catalog.initialize()

    # Discover dataset
    dataset_info = DatasetInfo.discover(
        repo_id="stanfordnlp/imdb",
        token=os.getenv("HF_TOKEN"),
    )

    # Create table
    table = catalog._create_table_for_config(
        dataset_config=test_config.datasets[0],
        dataset_info=dataset_info,
        config_name="plain_text",
    )

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


def test_invalid_table_name_format(catalog, test_config):
    """Test invalid table name format raises error."""
    with pytest.raises(ValueError, match="Invalid table name"):
        catalog.create_tables(
            token=None,
            table_name="invalid_format",  # Missing namespace
        )


def test_dataset_not_found_in_config(catalog):
    """Test error when dataset not found in config."""
    # Catalog config has "imdb" dataset, so "nonexistent" should fail
    with pytest.raises(ValueError, match="not found in config"):
        catalog.create_tables(
            token=None,
            table_name="default.nonexistent_default",
        )


if __name__ == "__main__":
    # Run basic smoke test
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Running basic FacebergCatalog test...")

        config = FacebergConfig(
            catalog=CatalogConfig(name="test", location=tmpdir),
            datasets=[DatasetConfig(name="imdb", repo="stanfordnlp/imdb", configs=None)],
        )

        catalog = FacebergCatalog.from_config(config)
        print("✓ Created catalog from config")

        catalog.initialize()
        print("✓ Initialized catalog")

        print("\n✓ All basic tests passed!")

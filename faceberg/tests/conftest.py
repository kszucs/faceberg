"""Shared pytest fixtures for catalog tests."""

import pytest

from faceberg.catalog import LocalCatalog
from faceberg.config import CatalogConfig, NamespaceConfig, TableConfig


@pytest.fixture(scope="session")
def synced_catalog_dir(tmp_path_factory):
    """Create temp directory for synced catalog (session-scoped)."""
    return tmp_path_factory.mktemp("synced_catalog")


@pytest.fixture(scope="session")
def synced_catalog(synced_catalog_dir):
    """Create and sync catalog with test dataset (session-scoped).

    Syncs stanfordnlp/imdb (plain_text config) - a small public dataset with
    org prefix compatible with DuckDB's httpfs hf:// URL requirements.

    This fixture is session-scoped to minimize API calls and improve test speed.
    The catalog is synced once and shared across all tests.
    """
    # Create config with test dataset
    config = CatalogConfig(
        namespaces=[
            NamespaceConfig(
                name="default",
                tables=[
                    TableConfig(
                        name="imdb_plain_text",
                        dataset="stanfordnlp/imdb",
                        config="plain_text",
                    ),
                ],
            )
        ],
    )

    # Create catalog instance (hf:// protocol support is built-in)
    catalog = LocalCatalog(location=str(synced_catalog_dir))

    # Sync all tables (token=None works for public datasets)
    synced_tables = catalog.sync(config)

    # Verify sync was successful
    assert len(synced_tables) == 1, f"Expected 1 table, got {len(synced_tables)}"

    return catalog


@pytest.fixture
def catalog(synced_catalog):
    """Provide catalog instance (function-scoped wrapper).

    This is a function-scoped fixture that provides access to the session-scoped
    synced catalog. The catalog has built-in hf:// protocol support via HfFileIO.
    """
    return synced_catalog

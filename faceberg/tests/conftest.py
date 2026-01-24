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
    """Create and sync catalog with test datasets (session-scoped).

    Syncs two small public datasets:
    - stanfordnlp/imdb (plain_text config)
    - rotten_tomatoes (default config)

    This fixture is session-scoped to minimize API calls and improve test speed.
    The catalog is synced once and shared across all tests.
    """
    # Create config with two test datasets
    config = CatalogConfig(
        name="test_catalog",
        location=str(synced_catalog_dir),
        namespaces=[
            NamespaceConfig(
                name="default",
                tables=[
                    TableConfig(
                        name="imdb_plain_text",
                        dataset="stanfordnlp/imdb",
                        config="plain_text",
                    ),
                    TableConfig(
                        name="rotten_tomatoes",
                        dataset="rotten_tomatoes",
                        config="default",
                    ),
                ],
            )
        ],
    )

    # Create catalog instance (hf:// protocol support is built-in)
    catalog = LocalCatalog(config=config)

    # Sync all tables (token=None works for public datasets)
    synced_tables = catalog.sync(token=None, table_name=None)

    # Verify sync was successful
    assert len(synced_tables) == 2, f"Expected 2 tables, got {len(synced_tables)}"

    return catalog


@pytest.fixture
def catalog(synced_catalog):
    """Provide catalog instance (function-scoped wrapper).

    This is a function-scoped fixture that provides access to the session-scoped
    synced catalog. The catalog has built-in hf:// protocol support via HfFileIO.
    """
    return synced_catalog

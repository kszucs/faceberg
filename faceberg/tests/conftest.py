"""Shared pytest fixtures for catalog tests."""

import os

import pytest

from faceberg.catalog import LocalCatalog
from faceberg.config import CatalogConfig, NamespaceConfig, TableConfig


# Workaround for HfFileSystem threading issue
# HfFileSystem has a cache that can include exceptions that can't be deep copied
# when used in multi-threaded contexts. We disable instance caching to avoid this.
def _patch_hffilesystem():
    """Patch HfFileSystem to disable instance caching for thread safety."""
    try:
        from huggingface_hub import HfFileSystem

        original_call = HfFileSystem.__call__

        def patched_call(cls, *args, **kwargs):
            # Disable instance caching to avoid deep copy issues with cached exceptions
            kwargs['skip_instance_cache'] = True
            return original_call(*args, **kwargs)

        HfFileSystem.__call__ = classmethod(patched_call)
    except ImportError:
        pass  # HuggingFace not installed


_patch_hffilesystem()


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

    # Create catalog instance with FsspecFileIO to support hf:// URIs
    catalog = LocalCatalog(
        name=config.name,
        location=config.location,
        config=config,
        # Enable FsspecFileIO for hf:// protocol support
        **{
            "py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO",
            "hf.endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
            # Disable fsspec caching to avoid threading issues
            "fsspec.cache_type": "none",
            # Use single worker to avoid threading issues with HfFileSystem
            "max-workers": "1",
        },
    )

    # Sync all tables (token=None works for public datasets)
    synced_tables = catalog.sync(token=None, table_name=None)

    # Verify sync was successful
    assert len(synced_tables) == 2, f"Expected 2 tables, got {len(synced_tables)}"

    return catalog


@pytest.fixture
def catalog(synced_catalog):
    """Provide catalog instance (function-scoped wrapper).

    This is a function-scoped fixture that provides access to the session-scoped
    synced catalog. The catalog is configured with FsspecFileIO to support hf:// URIs.
    """
    # Clear HfFileSystem cache before each test to avoid threading issues
    # The cache can contain exceptions that can't be deep copied
    try:
        from huggingface_hub import HfFileSystem

        HfFileSystem._cache.clear()
    except (ImportError, AttributeError):
        pass  # HuggingFace not installed or cache structure changed

    return synced_catalog

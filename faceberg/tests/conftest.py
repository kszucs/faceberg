"""Shared pytest fixtures for catalog tests."""

import socket
import threading
import time

import pytest
import requests
import uvicorn

from faceberg.catalog import LocalCatalog
from faceberg.database import Catalog, Namespace, Table
from faceberg.server import create_app


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
    # Create catalog directory
    synced_catalog_dir.mkdir(parents=True, exist_ok=True)

    # Create store with test dataset
    catalog_uri = f"file:///{synced_catalog_dir.as_posix()}"
    store_obj = Catalog(
        uri=catalog_uri,
        namespaces={
            "default": Namespace(
                tables={
                    "imdb_plain_text": Table(
                        dataset="stanfordnlp/imdb",
                        uri="",  # Empty until synced
                        revision="",  # Empty until synced
                        config="plain_text",
                    ),
                }
            )
        },
    )

    # Write config to faceberg.yml
    config_file = synced_catalog_dir / "faceberg.yml"
    store_obj.to_yaml(config_file)

    # Create catalog instance (hf:// protocol support is built-in)
    catalog = LocalCatalog(name=str(synced_catalog_dir), uri=catalog_uri)

    # Sync all tables (token=None works for public datasets)
    synced_tables = catalog.sync_datasets()

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


@pytest.fixture(scope="session")
def rest_server(synced_catalog):
    """Start REST catalog server for testing (session-scoped).

    Returns the base URL of the server (e.g., http://localhost:8181).
    The server runs in a background thread and is shared across all tests.
    """
    # Find available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    base_url = f"http://127.0.0.1:{port}"
    app = create_app(str(synced_catalog.catalog_dir))

    # Start server in background thread
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    for _ in range(50):
        try:
            if requests.get(f"{base_url}/v1/config", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("REST server failed to start")

    yield base_url

    # Cleanup
    server.should_exit = True
    thread.join(timeout=5)

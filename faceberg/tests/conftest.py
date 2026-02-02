"""Shared pytest fixtures for catalog tests."""

import os
import shutil
import socket
import threading
import time
from contextlib import contextmanager

import pytest
import requests
import uvicorn
from pyiceberg.catalog.rest import RestCatalog

from faceberg.catalog import LocalCatalog, RemoteCatalog
from faceberg.server import create_app


def pytest_addoption(parser):
    """Add custom pytest command line options."""
    parser.addoption(
        "--hf-live",
        action="store_true",
        default=False,
        help="Enable live HuggingFace Hub testing (requires HF_TOKEN)",
    )


@contextmanager
def local_catalog(path):
    uri = f"file://{path.as_posix()}"
    catalog = LocalCatalog(name="test-local-catalog", uri=uri)
    try:
        catalog.init()
        yield catalog
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def remote_catalog(hf_org, hf_token):
    # Create unique repo name for this test session
    uri = f"hf://datasets/{hf_org}/faceberg-test"
    catalog = RemoteCatalog(name="faceberg-test", uri=uri, hf_token=hf_token)

    # Remove the testing repo if it exists from previous runs
    catalog.hf_api.delete_repo(catalog.hf_repo, repo_type=catalog.hf_repo_type, missing_ok=True)
    try:
        catalog.init()
        yield catalog
    finally:
        # Do not cleanup to allow inspection of test artifacts
        pass


@pytest.fixture(params=["local", "remote"])
def catalog(request, tmp_path):
    """Parametrized empty catalog fixture for both local and remote catalogs.

    This fixture provides an empty LocalCatalog or RemoteCatalog for testing
    catalog operations.

    For remote catalogs:
    - Requires --hf-live flag and HF_TOKEN environment variable
    - Creates a unique repository per test session to avoid conflicts
    - Repository format: hf://datasets/{org}/faceberg-test-{session_id}

    Args:
        request: Pytest request object with param ("local" or "remote")
        tmp_path: Temporary directory for this test

    Returns:
        Either an empty LocalCatalog or RemoteCatalog instance
    """
    if request.param == "local":
        # Create local catalog within context manager
        with local_catalog(tmp_path) as catalog:
            yield catalog
    elif request.param == "remote":
        # Check if live HF testing is enabled
        hf_live = request.config.getoption("--hf-live")
        hf_org = os.environ.get("FACEBERG_TEST_ORG")
        hf_token = os.environ.get("FACEBERG_TEST_TOKEN")
        # TODO(kszucs): hf_repo should be the unique repo name per test session
        # maybe also check the token scopes
        if not hf_live:
            pytest.skip("Live HF testing not enabled (use --hf-live)")
        if not (hf_org and hf_token):
            pytest.skip(
                "FACEBERG_TEST_ORG and FACEBERG_TEST_TOKEN environment variables must be set"
            )
        # Create remote catalog within context manager
        with remote_catalog(hf_org, hf_token) as catalog:
            yield catalog
    else:
        raise ValueError(f"Unknown catalog type: {request.param}")


@pytest.fixture
def synced_catalog(catalog):
    """Catalog with synced test dataset (inherits parametrization from catalog).

    Syncs stanfordnlp/imdb (plain_text config) - a small public dataset with
    org prefix compatible with DuckDB's httpfs hf:// URL requirements.

    Note: Tests that access table data or metadata (schema, snapshots, scanning)
    with hf:// URLs will be skipped for remote catalog because PyIceberg's HfFileSystem
    tries to validate repository existence which requires network access.

    Args:
        catalog: Empty catalog instance (local or remote, from parametrized fixture)

    Returns:
        Catalog instance with synced imdb dataset
    """
    catalog.add_dataset("stanfordnlp.imdb", repo="stanfordnlp/imdb", config="plain_text")

    assert ("stanfordnlp",) in catalog.list_namespaces()

    return catalog


@pytest.fixture
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
    app = create_app(synced_catalog.uri)

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


@pytest.fixture
def rest_catalog(rest_server):
    """Create PyIceberg RestCatalog connected to test server.

    Configures the catalog to use HfFileIO for handling hf:// URIs.
    """
    return RestCatalog(
        name="faceberg_rest",
        uri=rest_server,
        **{
            "py-io-impl": "faceberg.catalog.HfFileIO",
        },
    )

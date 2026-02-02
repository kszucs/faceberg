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

from faceberg.catalog import LocalCatalog, RemoteCatalog
from faceberg.config import Config, Namespace
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
def writable_catalog(tmp_path):
    """Create catalog with writable table for testing write operations.

    Creates a catalog with a writable table (not from HuggingFace dataset)
    that can be used to test append and other write operations.
    """
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField, StringType

    # Create catalog directory
    catalog_dir = tmp_path / "writable_catalog"
    catalog_dir.mkdir()

    # Define table data URI
    table_data_uri = f"file://{(catalog_dir / 'data').as_posix()}"

    # Create config with empty default namespace
    catalog_uri = f"file://{catalog_dir.as_posix()}"
    store_obj = Config(default=Namespace())

    # Write config to faceberg.yml
    config_file = catalog_dir / "faceberg.yml"
    store_obj.to_yaml(config_file)

    # Create catalog instance
    catalog = LocalCatalog(name=str(catalog_dir), uri=catalog_uri)

    # Create the table with schema matching imdb dataset
    schema = Schema(
        NestedField(field_id=1, name="split", field_type=StringType(), required=False),
        NestedField(field_id=2, name="text", field_type=StringType(), required=False),
        NestedField(field_id=3, name="label", field_type=LongType(), required=False),
    )

    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    partition_spec = PartitionSpec(
        PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="split")
    )

    # Create the table
    catalog.create_table(
        identifier="default.test_table",
        schema=schema,
        partition_spec=partition_spec,
        properties={"write.data.path": table_data_uri},
    )

    return catalog


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

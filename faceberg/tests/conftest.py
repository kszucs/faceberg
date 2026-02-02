"""Shared pytest fixtures for catalog tests."""

import os
import shutil
from contextlib import contextmanager

import pytest
from datasets import Dataset
from huggingface_hub import HfApi
from pyiceberg.catalog.rest import RestCatalog

from faceberg.catalog import LocalCatalog, RemoteCatalog
from faceberg.server import serve_app


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
    catalog = LocalCatalog(name="local", uri=uri)
    try:
        catalog.init()
        yield catalog
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def remote_catalog():
    hf_org = os.environ.get("FACEBERG_TEST_ORG")
    hf_token = os.environ.get("FACEBERG_TEST_TOKEN")
    if not (hf_org and hf_token):
        pytest.skip("FACEBERG_TEST_ORG and FACEBERG_TEST_TOKEN environment variables must be set")

    # Create unique repo name for this test session
    uri = f"hf://datasets/{hf_org}/faceberg-catalog"
    catalog = RemoteCatalog(name="remote", uri=uri, hf_token=hf_token)

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
        if not request.config.getoption("--hf-live"):
            pytest.skip("Live HF testing not enabled (use --hf-live)")
        with remote_catalog() as catalog:
            yield catalog
    else:
        raise ValueError(f"Unknown catalog type: {request.param}")


@pytest.fixture(params=["local", "remote"], scope="session")
def session_catalog(request, tmp_path_factory):
    """Session-scoped parametrized catalog fixture (local and remote).

    Similar to `catalog` fixture but with session scope for expensive setup.

    Args:
        request: Pytest request object with param ("local" or "remote")
        tmp_path_factory: Factory for creating temporary directories

    Returns:
        Either an empty LocalCatalog or RemoteCatalog instance
    """
    if request.param == "local":
        tmp_path = tmp_path_factory.mktemp("local_catalog_session")
        with local_catalog(tmp_path) as catalog:
            yield catalog
    elif request.param == "remote":
        if not request.config.getoption("--hf-live"):
            pytest.skip("Live HF testing not enabled (use --hf-live)")
        with remote_catalog() as catalog:
            yield catalog
    else:
        raise ValueError(f"Unknown catalog type: {request.param}")


@pytest.fixture
def mbpp(catalog):
    """Catalog with small test dataset (mbpp - ~1000 examples)."""
    catalog.add_dataset(
        "google-research-datasets.mbpp", repo="google-research-datasets/mbpp", config="sanitized"
    )
    assert ("google-research-datasets",) in catalog.list_namespaces()
    return catalog


@pytest.fixture(scope="session")
def session_mbpp(session_catalog):
    """Session-scoped catalog with small test dataset (mbpp - ~1000 examples)."""
    session_catalog.add_dataset(
        "google-research-datasets.mbpp", repo="google-research-datasets/mbpp", config="sanitized"
    )
    assert ("google-research-datasets",) in session_catalog.list_namespaces()
    return session_catalog


@pytest.fixture(scope="session")
def session_rest_server(session_mbpp):
    """Start REST catalog server for testing (session-scoped).

    Returns the base URL of the server (e.g., http://localhost:8181).
    The server runs in a background thread and is shared across all tests.
    """
    with serve_app(session_mbpp.uri) as base_url:
        yield base_url


@pytest.fixture(scope="session")
def session_rest_catalog(session_rest_server):
    """Create PyIceberg RestCatalog connected to test server (session-scoped).

    Configures the catalog to use HfFileIO for handling hf:// URIs.
    """
    return RestCatalog(
        name="faceberg_rest_session",
        uri=session_rest_server,
        **{
            "py-io-impl": "faceberg.catalog.HfFileIO",
        },
    )


@contextmanager
def remote_dataset(postfix):
    """Create a small synthetic dataset for testing writes.

    Creates and publishes a small synthetic dataset to HuggingFace Hub that can
    be used for testing write operations.

    Args:
        hf_org: Organization to create the dataset in
        hf_token: HuggingFace authentication token

    Yields:
        Dataset repo ID
    """
    hf_org = os.environ.get("FACEBERG_TEST_ORG")
    hf_token = os.environ.get("FACEBERG_TEST_TOKEN")
    if not (hf_org and hf_token):
        pytest.skip("FACEBERG_TEST_ORG and FACEBERG_TEST_TOKEN environment variables must be set")

    hf_repo = f"{hf_org}/faceberg-dataset-{postfix}"
    hf_api = HfApi(token=hf_token)
    hf_api.delete_repo(repo_id=hf_repo, repo_type="dataset", missing_ok=True)
    try:
        # Create small synthetic dataset
        # Note: "split" column is added automatically by add_dataset, not in the data
        dataset = Dataset.from_dict(
            {
                "text": [f"Test review {i}" for i in range(10)],
                "label": [i % 2 for i in range(10)],
            }
        )

        # Push dataset to hub (creates repo and uploads data)
        dataset.push_to_hub(
            hf_repo,
            token=hf_token,
            private=False,
        )

        yield hf_repo
    finally:
        pass


@pytest.fixture
def writable_dataset(catalog, request):
    """Parametrized writable dataset for both local and remote catalogs.

    Creates a small test dataset on HuggingFace Hub that can be written to.
    The dataset is always created on HF Hub regardless of catalog type.
    Inherits parametrization from catalog fixture.

    Requires:
    - --hf-live flag
    - FACEBERG_TEST_ORG and FACEBERG_TEST_TOKEN environment variables

    Returns:
        Catalog instance with writable testorg.testdataset table
    """
    # Create a small synthetic test dataset on HuggingFace Hub
    with remote_dataset(postfix=catalog.name) as dataset_repo:
        # Add dataset to catalog - this discovers the uploaded files and creates Iceberg metadata
        # No config specified - let it auto-detect from parquet files
        catalog.add_dataset(
            identifier="testorg.testdataset",
            repo=dataset_repo,
        )
        yield catalog

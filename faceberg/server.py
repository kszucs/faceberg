"""REST catalog server implementation for Faceberg.

This module implements a REST API server that exposes Faceberg catalogs
(LocalCatalog and RemoteCatalog) via HTTP endpoints following the Apache
Iceberg REST catalog specification.

Initial implementation supports read-only operations:
- GET /v1/config - Get catalog configuration
- GET /v1/namespaces - List namespaces
- GET /v1/namespaces/{namespace} - Load namespace properties
- HEAD /v1/namespaces/{namespace} - Check namespace exists
- GET /v1/namespaces/{namespace}/tables - List tables
- GET /v1/namespaces/{namespace}/tables/{table} - Load table
- HEAD /v1/namespaces/{namespace}/tables/{table} - Check table exists
"""

from __future__ import annotations

import os
from pathlib import Path
from http import HTTPStatus
from typing import Any, Dict, Optional

# Disable Litestar warnings about sync handlers - all catalog operations are blocking I/O
os.environ.setdefault("LITESTAR_WARN_IMPLICIT_SYNC_TO_THREAD", "0")

from huggingface_hub import HfApi
from litestar import Controller, Litestar, Request, get, head
from litestar.config.cors import CORSConfig
from litestar.datastructures import State
from litestar.params import Parameter
from litestar.response import Response
from pyiceberg.catalog import URI, WAREHOUSE_LOCATION
from pyiceberg.catalog.rest import (
    PREFIX,
    ConfigResponse,
    ListNamespaceResponse,
    ListTablesResponse,
    NamespaceResponse,
    TableResponse,
)
from pyiceberg.exceptions import (
    CommitFailedException,
    NamespaceAlreadyExistsError,
    NamespaceNotEmptyError,
    NoSuchNamespaceError,
    NoSuchTableError,
    NoSuchViewError,
    TableAlreadyExistsError,
)

from .catalog import catalog as create_catalog


# =========================================================================
# Error Handling
# =========================================================================


def _handle_exception(_request: Request, exc: Exception) -> Response:
    """Map Iceberg exceptions to HTTP responses with error details."""
    status_map = {
        NoSuchNamespaceError: HTTPStatus.NOT_FOUND,
        NoSuchTableError: HTTPStatus.NOT_FOUND,
        NoSuchViewError: HTTPStatus.NOT_FOUND,
        NamespaceAlreadyExistsError: HTTPStatus.CONFLICT,
        TableAlreadyExistsError: HTTPStatus.CONFLICT,
        NamespaceNotEmptyError: HTTPStatus.CONFLICT,
        CommitFailedException: HTTPStatus.CONFLICT,
    }

    status = status_map.get(type(exc), HTTPStatus.INTERNAL_SERVER_ERROR)
    message = str(exc) or exc.__class__.__name__

    content = {
        "error": {
            "message": message,
            "type": exc.__class__.__name__,
            "code": status,
        }
    }

    return Response(status_code=status, content=content)


# Exception handlers for Litestar
_EXCEPTION_HANDLERS: Dict[int | type[Exception], Any] = {
    NoSuchNamespaceError: _handle_exception,
    NoSuchTableError: _handle_exception,
    NoSuchViewError: _handle_exception,
    NamespaceAlreadyExistsError: _handle_exception,
    TableAlreadyExistsError: _handle_exception,
    NamespaceNotEmptyError: _handle_exception,
    CommitFailedException: _handle_exception,
    Exception: _handle_exception,  # Catch-all for consistent error format
}


# =========================================================================
# Config Endpoint
# =========================================================================


@get("/v1/config", sync_to_thread=False)
def get_config_handler(
    state: State,
    request: Request,
    warehouse_param: Optional[str] = Parameter(default=None, query="warehouse"),
) -> Dict[str, Any]:
    """Get catalog configuration including warehouse location and prefix."""
    catalog = state["catalog"]
    warehouse = warehouse_param or catalog.properties.get(WAREHOUSE_LOCATION, "")

    overrides: Dict[str, Any] = {
        URI: str(request.base_url),
        WAREHOUSE_LOCATION: warehouse,
    }

    prefix = state.get("prefix", "")
    if prefix:
        overrides[PREFIX] = prefix

    return ConfigResponse(defaults={}, overrides=overrides).model_dump()


# =========================================================================
# Namespace Endpoints (READ-ONLY)
# =========================================================================


class NamespaceController(Controller):
    """Controller for namespace operations (read-only)."""

    path = "/v1/namespaces"
    sync_to_thread = False

    @get("")
    def list_namespaces(
        self, state: State, parent: Optional[str] = Parameter(default=None, query="parent")
    ) -> Dict[str, Any]:
        """List all namespaces or namespaces under a parent."""
        catalog = state["catalog"]
        namespaces = catalog.list_namespaces(parent if parent else ())
        return ListNamespaceResponse(namespaces=namespaces).model_dump()

    @get("/{namespace:str}")
    def load_namespace(self, state: State, namespace: str) -> Dict[str, Any]:
        """Load namespace properties."""
        catalog = state["catalog"]
        props = catalog.load_namespace_properties(namespace)
        ns = catalog.identifier_to_tuple(namespace)
        return NamespaceResponse(namespace=ns, properties=props).model_dump()

    @head("/{namespace:str}", status_code=HTTPStatus.NO_CONTENT)
    def namespace_exists(self, state: State, namespace: str) -> None:
        """Check if namespace exists (returns 204 if exists, 404 if not)."""
        catalog = state["catalog"]
        catalog.load_namespace_properties(namespace)  # Raises NoSuchNamespaceError if not exists


# =========================================================================
# Table Endpoints (READ-ONLY)
# =========================================================================


class TablesController(Controller):
    """Controller for table operations (read-only)."""

    path = "/v1/namespaces"
    sync_to_thread = False

    @get("/{namespace:str}/tables")
    def list_tables(self, state: State, namespace: str) -> Dict[str, Any]:
        """List all tables in a namespace."""
        catalog = state["catalog"]
        identifiers = catalog.list_tables(namespace)

        # Convert identifiers to ListTablesResponse format
        from pyiceberg.catalog.rest import ListTableResponseEntry

        entries = []
        for identifier in identifiers:
            # identifier is a tuple like ('namespace', 'table')
            if len(identifier) == 2:
                ns, table_name = identifier
                entries.append(ListTableResponseEntry(namespace=(ns,), name=table_name))
            else:
                # Handle case where namespace has multiple parts
                *ns_parts, table_name = identifier
                entries.append(
                    ListTableResponseEntry(namespace=tuple(ns_parts), name=table_name)
                )

        return ListTablesResponse(identifiers=entries).model_dump()

    @get("/{namespace:str}/tables/{table:str}")
    def load_table(self, state: State, namespace: str, table: str) -> Dict[str, Any]:
        """Load table metadata."""
        catalog = state["catalog"]
        identifier = f"{namespace}.{table}"
        table_obj = catalog.load_table(identifier)

        # Convert Table to TableResponse
        return TableResponse(
            metadata_location=table_obj.metadata_location,
            metadata=table_obj.metadata,
            config=getattr(table_obj, "config", {}) or {},
        ).model_dump(by_alias=True)

    @head("/{namespace:str}/tables/{table:str}", status_code=HTTPStatus.NO_CONTENT)
    def table_exists(self, state: State, namespace: str, table: str) -> None:
        """Check if table exists (returns 204 if exists, 404 if not)."""
        catalog = state["catalog"]
        identifier = f"{namespace}.{table}"
        if not catalog.table_exists(identifier):
            raise NoSuchTableError(f"Table does not exist: {identifier}")


# =========================================================================
# Application Factory
# =========================================================================


def create_app(
    catalog_uri: str,
    hf_token: Optional[str] = None,
    prefix: str = "",
) -> Litestar:
    """Create a Litestar application exposing a REST catalog.

    Args:
        catalog_uri: Catalog URI - file:// for LocalCatalog or hf:// for RemoteCatalog
        hf_token: HuggingFace API token (required for RemoteCatalog)
        prefix: URL prefix for the REST API (optional)

    Returns:
        Configured Litestar application

    Examples:
        >>> # Local catalog
        >>> app = create_app("/path/to/catalog")

        >>> # Remote catalog on HuggingFace Hub
        >>> app = create_app("hf://datasets/org/repo", hf_token="hf_...")

        >>> # With URL prefix
        >>> app = create_app("/path/to/catalog", prefix="my-catalog")
    """
    # Initialize catalog using factory function
    catalog = create_catalog(catalog_uri, hf_token=hf_token)

    # Application state
    app_state = State({
        "catalog": catalog,
        "prefix": prefix.strip("/"),
    })

    # Create Litestar app
    app = Litestar(
        route_handlers=[get_config_handler, NamespaceController, TablesController],
        state=app_state,
        exception_handlers=_EXCEPTION_HANDLERS,
        cors_config=CORSConfig(allow_origins=["*"]),  # Allow CORS for all origins
    )

    return app


def deploy_app(
    space_name: str,
    catalog_uri: str,
    hf_token: Optional[str] = None,
    github_repo: str = "kszucs/faceberg",
) -> str:
    """Deploy the catalog server to HF Spaces from GitHub.

    The deployed server will listen on host 0.0.0.0 and port 7860 (HF Spaces default).
    This installs faceberg from the specified GitHub repository.

    Args:
        space_name: Space name in format "username/space-name"
        catalog_uri: Catalog URI to serve
        hf_token: HuggingFace API token (or use HF_TOKEN env var)
        github_repo: GitHub repository in format "owner/repo" (default: kszucs/faceberg)

    Returns:
        Space URL (e.g., "https://huggingface.co/spaces/user/my-catalog")

    Examples:
        >>> deploy_spaces_app("user/my-catalog", "hf://datasets/org/repo")
        'https://huggingface.co/spaces/user/my-catalog'

        Connect to the deployed catalog:
        >>> from pyiceberg.catalog.rest import RestCatalog
        >>> catalog = RestCatalog(
        ...     name="my-catalog",
        ...     uri="https://user-my-catalog.hf.space"
        ... )
    """
    import tempfile
    from huggingface_hub import CommitOperationAdd

    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=space_name,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    # Get paths
    spaces_dir = Path(__file__).parent / "spaces"

    # Read templates
    dockerfile_template = (spaces_dir / "Dockerfile").read_text()
    readme_template = (spaces_dir / "README.md").read_text()

    # Format templates
    dockerfile_content = dockerfile_template.format(github_repo=github_repo)

    space_display_name = space_name.split("/")[1].replace("-", " ").title()
    api_url = space_name.replace("/", "-")
    readme_content = readme_template.format(
        space_display_name=space_display_name,
        catalog_uri=catalog_uri,
        api_url=api_url,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write Dockerfile
        dockerfile_path = tmpdir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)

        # Write README
        readme_path = tmpdir / "README.md"
        readme_path.write_text(readme_content)

        # Upload files
        operations = [
            CommitOperationAdd(path_in_repo="Dockerfile", path_or_fileobj=str(dockerfile_path)),
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme_path)),
        ]

        api.create_commit(
            repo_id=space_name,
            repo_type="space",
            operations=operations,
            commit_message="Deploy Faceberg catalog server",
        )

    # Set environment variables
    api.add_space_secret(repo_id=space_name, key="CATALOG_URI", value=catalog_uri)
    if catalog_uri.startswith("hf://") and hf_token:
        api.add_space_secret(repo_id=space_name, key="HF_TOKEN", value=hf_token)

    return f"https://huggingface.co/spaces/{space_name}"


__all__ = ["create_app", "deploy_app"]

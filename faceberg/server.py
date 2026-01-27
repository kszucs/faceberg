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
from http import HTTPStatus
from typing import Any, Dict, Optional

# Disable Litestar warnings about sync handlers - all catalog operations are blocking I/O
os.environ.setdefault("LITESTAR_WARN_IMPLICIT_SYNC_TO_THREAD", "0")

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
                entries.append(ListTableResponseEntry(namespace=tuple(ns_parts), name=table_name))

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
    app_state = State(
        {
            "catalog": catalog,
            "prefix": prefix.strip("/"),
        }
    )

    # Create Litestar app
    app = Litestar(
        route_handlers=[get_config_handler, NamespaceController, TablesController],
        state=app_state,
        exception_handlers=_EXCEPTION_HANDLERS,
        cors_config=CORSConfig(allow_origins=["*"]),  # Allow CORS for all origins
    )

    return app


__all__ = ["create_app"]

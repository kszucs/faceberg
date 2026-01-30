"""Tests for REST catalog server."""

from litestar.testing import TestClient

from faceberg.config import Config, Namespace
from faceberg.server import create_app


class TestConfigEndpoint:
    """Tests for /v1/config endpoint."""

    def test_get_config(self, synced_catalog):
        """Test getting catalog configuration."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/config")
            assert response.status_code == 200

            data = response.json()
            assert "defaults" in data
            assert "overrides" in data
            assert "uri" in data["overrides"]

    def test_get_config_with_warehouse_param(self, synced_catalog):
        """Test config endpoint with warehouse parameter."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/config?warehouse=/my/warehouse")
            assert response.status_code == 200

            data = response.json()
            assert data["overrides"]["warehouse"] == "/my/warehouse"


class TestNamespaceEndpoints:
    """Tests for namespace-related endpoints."""

    def test_list_namespaces(self, synced_catalog):
        """Test listing all namespaces."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/namespaces")
            assert response.status_code == 200

            data = response.json()
            assert "namespaces" in data
            namespaces = data["namespaces"]
            assert len(namespaces) > 0
            # Check that 'default' namespace exists
            assert ["default"] in namespaces or ("default",) in namespaces

    def test_load_namespace(self, synced_catalog):
        """Test loading namespace properties."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/namespaces/default")
            assert response.status_code == 200

            data = response.json()
            assert "namespace" in data
            assert "properties" in data
            # Namespace should be a list/tuple
            assert data["namespace"] == ["default"] or data["namespace"] == ("default",)

    def test_namespace_exists_head(self, synced_catalog):
        """Test checking namespace existence with HEAD request."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            # Existing namespace should return 204
            response = client.head("/v1/namespaces/default")
            assert response.status_code == 204

    def test_namespace_not_exists(self, synced_catalog):
        """Test loading non-existent namespace returns empty properties.

        Note: The catalog behavior is to return empty properties rather than
        raising an error for non-existent namespaces.
        """
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/namespaces/nonexistent")
            assert response.status_code == 200

            data = response.json()
            assert "namespace" in data
            assert "properties" in data
            # Empty properties for non-existent namespace
            assert data["properties"] == {}


class TestTableEndpoints:
    """Tests for table-related endpoints."""

    def test_list_tables(self, synced_catalog):
        """Test listing tables in a namespace."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/namespaces/default/tables")
            assert response.status_code == 200

            data = response.json()
            assert "identifiers" in data
            tables = data["identifiers"]
            assert len(tables) > 0
            # Check structure of table entries
            assert all("namespace" in t and "name" in t for t in tables)

    def test_load_table(self, synced_catalog):
        """Test loading a table."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/namespaces/default/tables/imdb_plain_text")
            assert response.status_code == 200

            data = response.json()
            # TableResponse uses snake_case in JSON (by_alias=True converts to kebab-case)
            assert "metadata-location" in data or "metadata_location" in data
            assert "metadata" in data
            # Verify metadata structure
            metadata = data["metadata"]
            assert "format-version" in metadata or "format_version" in metadata

    def test_table_exists_head(self, synced_catalog):
        """Test checking table existence with HEAD request."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            # Existing table should return 204
            response = client.head("/v1/namespaces/default/tables/imdb_plain_text")
            assert response.status_code == 204

    def test_table_not_exists(self, synced_catalog):
        """Test loading non-existent table returns 404."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/namespaces/default/tables/nonexistent")
            assert response.status_code == 404

            data = response.json()
            assert "error" in data
            assert data["error"]["type"] == "NoSuchTableError"

    def test_table_exists_wrong_namespace(self, synced_catalog):
        """Test checking table in wrong namespace returns 404."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            response = client.head("/v1/namespaces/nonexistent/tables/imdb_plain_text")
            assert response.status_code == 404


class TestErrorHandling:
    """Tests for error handling."""

    def test_error_response_format(self, synced_catalog):
        """Test that errors follow Iceberg REST spec format."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            # Use table not found as the test case for error format
            response = client.get("/v1/namespaces/default/tables/nonexistent")
            assert response.status_code == 404

            data = response.json()
            assert "error" in data
            error = data["error"]
            assert "message" in error
            assert "type" in error
            assert "code" in error
            assert error["type"] == "NoSuchTableError"
            assert error["code"] == 404

    def test_internal_error_handling(self, synced_catalog):
        """Test that unexpected errors are caught and formatted properly."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            # Try to cause an error by requesting invalid paths
            response = client.get("/v1/config/invalid")
            # Should not crash, should return proper error format
            assert response.status_code in [404, 405, 500]


class TestEmptyCatalogBehavior:
    """Test server behavior with empty catalogs.

    These tests verify that the server correctly handles catalogs that have
    no tables or namespaces, which is important for new catalog initialization.
    """

    def test_empty_catalog_list_namespaces(self, tmp_path):
        """Test that an empty catalog returns empty namespace list."""
        # Create empty catalog
        catalog_dir = tmp_path / "empty_catalog"
        catalog_dir.mkdir()
        catalog_config = Config()
        catalog_config.to_yaml(catalog_dir / "faceberg.yml")

        # Create server
        app = create_app(str(catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/namespaces")
            assert response.status_code == 200

            data = response.json()
            assert "namespaces" in data
            # Empty catalog should return empty list, not error
            assert data["namespaces"] == []

    def test_empty_catalog_list_tables_in_default(self, tmp_path):
        """Test that listing tables in non-existent namespace returns 404."""
        # Create empty catalog
        catalog_dir = tmp_path / "empty_catalog"
        catalog_dir.mkdir()
        catalog_config = Config()
        catalog_config.to_yaml(catalog_dir / "faceberg.yml")

        # Create server
        app = create_app(str(catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/namespaces/default/tables")
            # Non-existent namespace should return 404
            assert response.status_code == 404

    def test_catalog_with_namespace_but_no_tables(self, tmp_path):
        """Test catalog with defined namespace but no tables."""
        # Create catalog with empty namespace
        catalog_dir = tmp_path / "catalog_with_empty_ns"
        catalog_dir.mkdir()
        catalog_config = Config({"default": Namespace()})
        catalog_config.to_yaml(catalog_dir / "faceberg.yml")

        # Create server
        app = create_app(str(catalog_dir))

        with TestClient(app=app) as client:
            # Should list the namespace
            response = client.get("/v1/namespaces")
            assert response.status_code == 200
            data = response.json()
            assert "namespaces" in data
            assert ["default"] in data["namespaces"] or ("default",) in data["namespaces"]

            # But tables should be empty
            response = client.get("/v1/namespaces/default/tables")
            assert response.status_code == 200
            data = response.json()
            assert "identifiers" in data
            assert data["identifiers"] == []


class TestSyncedCatalogDataIntegrity:
    """Test that synced catalog data is correctly exposed via REST API.

    These tests ensure that after syncing datasets, the REST API correctly
    exposes the namespace and table information.
    """

    def test_synced_catalog_has_default_namespace(self, synced_catalog):
        """Verify synced catalog exposes the default namespace."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/namespaces")
            assert response.status_code == 200

            data = response.json()
            namespaces = data["namespaces"]

            # Must have at least one namespace
            assert len(namespaces) > 0, "Synced catalog should have namespaces"

            # Should include default namespace
            namespace_list = [list(ns) if isinstance(ns, tuple) else ns for ns in namespaces]
            assert ["default"] in namespace_list or ("default",) in namespaces, (
                f"Expected 'default' namespace in {namespace_list}"
            )

    def test_synced_catalog_has_imdb_table(self, synced_catalog):
        """Verify synced catalog exposes the imdb_plain_text table."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/namespaces/default/tables")
            assert response.status_code == 200

            data = response.json()
            tables = data["identifiers"]

            # Must have at least one table
            assert len(tables) > 0, "Synced catalog should have tables in default namespace"

            # Should include imdb_plain_text table
            table_names = [t["name"] if isinstance(t, dict) else str(t) for t in tables]
            assert "imdb_plain_text" in table_names, f"Expected 'imdb_plain_text' in {table_names}"

    def test_synced_catalog_table_has_valid_metadata(self, synced_catalog):
        """Verify synced table returns valid Iceberg metadata."""
        app = create_app(str(synced_catalog.catalog_dir))

        with TestClient(app=app) as client:
            response = client.get("/v1/namespaces/default/tables/imdb_plain_text")
            assert response.status_code == 200

            data = response.json()

            # Must have metadata location and metadata
            assert "metadata" in data, "Table response must include metadata"
            assert "metadata-location" in data or "metadata_location" in data, (
                "Table response must include metadata location"
            )

            metadata = data["metadata"]

            # Verify metadata structure
            assert "format-version" in metadata or "format_version" in metadata
            assert "schemas" in metadata, "Metadata must include schemas"
            assert "current-schema-id" in metadata or "current_schema_id" in metadata

            # Verify schema has fields
            schemas = metadata.get("schemas", [])
            assert len(schemas) > 0, "Table must have at least one schema"

            current_schema = schemas[0]
            fields = current_schema.get("fields", [])
            assert len(fields) > 0, "Schema must have fields"

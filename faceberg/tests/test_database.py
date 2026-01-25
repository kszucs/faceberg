"""Tests for faceberg.store module."""

import pytest
import yaml

from faceberg.database import Catalog, Namespace, Table


class TestTable:
    """Tests for Table dataclass."""

    def test_table_config_defaults(self):
        """Test Table with default config value."""
        table = Table(dataset="org/repo", uri="file:///path/to/metadata")
        assert table.dataset == "org/repo"
        assert table.config == "default"
        assert table.uri == "file:///path/to/metadata"

    def test_table_config_explicit(self):
        """Test Table with explicit config value."""
        table = Table(dataset="org/repo", uri="file:///path/to/metadata", config="custom")
        assert table.dataset == "org/repo"
        assert table.config == "custom"
        assert table.uri == "file:///path/to/metadata"


class TestNamespace:
    """Tests for Namespace dataclass."""

    def test_namespace_config(self):
        """Test Namespace creation."""
        tables = {
            "table1": Table(dataset="org/repo1", uri="file:///path1"),
            "table2": Table(dataset="org/repo2", uri="file:///path2", config="custom"),
        }
        namespace = Namespace(tables=tables)

        assert len(namespace.tables) == 2
        assert "table1" in namespace.tables
        assert "table2" in namespace.tables
        assert namespace.tables["table1"].dataset == "org/repo1"
        assert namespace.tables["table2"].dataset == "org/repo2"


class TestCatalog:
    """Tests for Catalog dataclass and parsing."""

    def test_catalog_config_creation(self):
        """Test Catalog creation."""
        tables = {"table1": Table(dataset="org/repo", uri="file:///path")}
        namespaces = {"ns1": Namespace(tables=tables)}

        config = Catalog(uri=".faceberg", namespaces=namespaces)

        assert config.uri == ".faceberg"
        assert len(config.namespaces) == 1
        assert "ns1" in config.namespaces

    def test_from_yaml_valid_config(self, tmp_path):
        """Test parsing valid YAML config."""
        yaml_content = """
uri: .faceberg

namespace1:
  table1:
    dataset: org/repo1
    config: config1
  table2:
    dataset: org/repo2
    config: config2

namespace2:
  table3:
    dataset: org/repo3
    # config defaults to 'default'
"""
        config_file = tmp_path / "test_config.yml"
        config_file.write_text(yaml_content)

        config = Catalog.from_yaml(config_file)

        assert len(config.namespaces) == 2

        # Check namespace1
        assert "namespace1" in config.namespaces
        ns1 = config.namespaces["namespace1"]
        assert len(ns1.tables) == 2
        assert "table1" in ns1.tables
        assert ns1.tables["table1"].dataset == "org/repo1"
        assert ns1.tables["table1"].config == "config1"
        assert "table2" in ns1.tables

        # Check namespace2
        assert "namespace2" in config.namespaces
        ns2 = config.namespaces["namespace2"]
        assert len(ns2.tables) == 1
        assert "table3" in ns2.tables
        assert ns2.tables["table3"].config == "default"  # Default value

    def test_from_yaml_missing_file(self):
        """Test parsing non-existent file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Catalog.from_yaml("/nonexistent/path.yml")

    def test_from_yaml_empty_file(self, tmp_path):
        """Test parsing empty YAML file."""
        config_file = tmp_path / "empty.yml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="Config file is empty"):
            Catalog.from_yaml(config_file)

    def test_from_yaml_valid_config_simple(self, tmp_path):
        """Test parsing simple valid YAML config (no catalog section needed)."""
        yaml_content = """
uri: .faceberg

namespace1:
  table1:
    dataset: org/repo1
"""
        config_file = tmp_path / "simple.yml"
        config_file.write_text(yaml_content)

        config = Catalog.from_yaml(config_file)
        assert len(config.namespaces) == 1
        assert "namespace1" in config.namespaces

    def test_from_yaml_no_namespaces(self, tmp_path):
        """Test parsing YAML with no namespaces defined (only reserved catalog key)."""
        yaml_content = """
uri: .faceberg

catalog:
  name: test_catalog
  location: .faceberg/
"""
        config_file = tmp_path / "no_namespaces.yml"
        config_file.write_text(yaml_content)

        with pytest.raises(
            ValueError, match="Cannot use 'catalog' as namespace name \\(reserved\\)"
        ):
            Catalog.from_yaml(config_file)

    def test_from_yaml_invalid_namespace_name(self, tmp_path):
        """Test parsing YAML with invalid namespace name (special characters)."""
        yaml_content = """
uri: .faceberg

namespace@invalid:
  table1:
    dataset: org/repo1
"""
        config_file = tmp_path / "invalid_ns.yml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="Invalid namespace name"):
            Catalog.from_yaml(config_file)

    def test_from_yaml_namespace_not_dict(self, tmp_path):
        """Test parsing YAML where namespace value is not a dict."""
        yaml_content = """
uri: .faceberg

namespace1: not_a_dict
"""
        config_file = tmp_path / "ns_not_dict.yml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="must be a dict of tables"):
            Catalog.from_yaml(config_file)

    def test_from_yaml_empty_namespace(self, tmp_path):
        """Test parsing YAML with namespace that has no tables (allowed for new namespaces)."""
        yaml_content = """
uri: .faceberg

namespace1: {}
"""
        config_file = tmp_path / "empty_ns.yml"
        config_file.write_text(yaml_content)

        # Empty namespaces are now allowed (for newly created namespaces)
        config = Catalog.from_yaml(config_file)
        assert len(config.namespaces) == 1
        assert "namespace1" in config.namespaces
        assert len(config.namespaces["namespace1"].tables) == 0

    def test_from_yaml_missing_dataset(self, tmp_path):
        """Test parsing YAML with table missing 'dataset' field."""
        yaml_content = """
uri: .faceberg

namespace1:
  table1:
    config: config1
"""
        config_file = tmp_path / "missing_dataset.yml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="Missing 'dataset' in namespace1.table1"):
            Catalog.from_yaml(config_file)

    def test_from_yaml_valid_namespace_names(self, tmp_path):
        """Test parsing YAML with various valid namespace names."""
        yaml_content = """
uri: .faceberg

namespace_1:
  table1:
    dataset: org/repo1

namespace-2:
  table2:
    dataset: org/repo2

Namespace3:
  table3:
    dataset: org/repo3

ns123:
  table4:
    dataset: org/repo4
"""
        config_file = tmp_path / "valid_names.yml"
        config_file.write_text(yaml_content)

        config = Catalog.from_yaml(config_file)
        assert len(config.namespaces) == 4
        assert "namespace_1" in config.namespaces
        assert "namespace-2" in config.namespaces
        assert "Namespace3" in config.namespaces
        assert "ns123" in config.namespaces

    def test_to_yaml(self, tmp_path):
        """Test exporting config to YAML."""
        tables = {
            "table1": Table(dataset="org/repo1", uri="file:///path1", config="config1"),
            "table2": Table(dataset="org/repo2", uri="file:///path2", config="default"),
        }
        namespaces = {"test_ns": Namespace(tables=tables)}

        config = Catalog(uri=".faceberg", namespaces=namespaces)

        output_file = tmp_path / "output.yml"
        config.to_yaml(output_file)

        # Read back and verify
        with open(output_file) as f:
            data = yaml.safe_load(f)

        assert "test_ns" in data
        assert "table1" in data["test_ns"]
        assert data["test_ns"]["table1"]["dataset"] == "org/repo1"
        assert data["test_ns"]["table1"]["config"] == "config1"
        assert data["test_ns"]["table2"]["config"] == "default"

    def test_round_trip(self, tmp_path):
        """Test that config can be exported and re-imported correctly."""
        yaml_content = """
uri: .faceberg

namespace1:
  table1:
    dataset: org/repo1
    config: config1
  table2:
    dataset: org/repo2
    config: config2

namespace2:
  table3:
    dataset: org/repo3
"""
        input_file = tmp_path / "input.yml"
        input_file.write_text(yaml_content)

        output_file = tmp_path / "output.yml"

        # Load, export, and re-load
        config1 = Catalog.from_yaml(input_file)
        config1.to_yaml(output_file)
        config2 = Catalog.from_yaml(output_file)

        # Verify they're equivalent
        assert len(config1.namespaces) == len(config2.namespaces)
        assert set(config1.namespaces.keys()) == set(config2.namespaces.keys())

        for ns_name in config1.namespaces:
            ns1 = config1.namespaces[ns_name]
            ns2 = config2.namespaces[ns_name]
            assert len(ns1.tables) == len(ns2.tables)
            assert set(ns1.tables.keys()) == set(ns2.tables.keys())

            for table_name in ns1.tables:
                t1 = ns1.tables[table_name]
                t2 = ns2.tables[table_name]
                assert t1.dataset == t2.dataset
                assert t1.config == t2.config

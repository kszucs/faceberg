"""Tests for faceberg.config module."""

import pytest
import yaml

from faceberg.config import CatalogConfig, NamespaceConfig, TableConfig


class TestTableConfig:
    """Tests for TableConfig dataclass."""

    def test_table_config_defaults(self):
        """Test TableConfig with default config value."""
        table = TableConfig(name="my_table", dataset="org/repo")
        assert table.name == "my_table"
        assert table.dataset == "org/repo"
        assert table.config == "default"

    def test_table_config_explicit(self):
        """Test TableConfig with explicit config value."""
        table = TableConfig(name="my_table", dataset="org/repo", config="custom")
        assert table.name == "my_table"
        assert table.dataset == "org/repo"
        assert table.config == "custom"


class TestNamespaceConfig:
    """Tests for NamespaceConfig dataclass."""

    def test_namespace_config(self):
        """Test NamespaceConfig creation."""
        table1 = TableConfig(name="table1", dataset="org/repo1")
        table2 = TableConfig(name="table2", dataset="org/repo2", config="custom")

        namespace = NamespaceConfig(name="test_ns", tables=[table1, table2])

        assert namespace.name == "test_ns"
        assert len(namespace.tables) == 2
        assert namespace.tables[0].name == "table1"
        assert namespace.tables[1].name == "table2"


class TestCatalogConfig:
    """Tests for CatalogConfig dataclass and parsing."""

    def test_catalog_config_creation(self):
        """Test CatalogConfig creation."""
        table = TableConfig(name="table1", dataset="org/repo")
        namespace = NamespaceConfig(name="ns1", tables=[table])

        config = CatalogConfig(namespaces=[namespace])

        assert len(config.namespaces) == 1
        assert config.namespaces[0].name == "ns1"

    def test_from_yaml_valid_config(self, tmp_path):
        """Test parsing valid YAML config."""
        yaml_content = """
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

        config = CatalogConfig.from_yaml(config_file)

        assert len(config.namespaces) == 2

        # Check namespace1
        ns1 = config.namespaces[0]
        assert ns1.name == "namespace1"
        assert len(ns1.tables) == 2
        assert ns1.tables[0].name == "table1"
        assert ns1.tables[0].dataset == "org/repo1"
        assert ns1.tables[0].config == "config1"
        assert ns1.tables[1].name == "table2"

        # Check namespace2
        ns2 = config.namespaces[1]
        assert ns2.name == "namespace2"
        assert len(ns2.tables) == 1
        assert ns2.tables[0].name == "table3"
        assert ns2.tables[0].config == "default"  # Default value

    def test_from_yaml_missing_file(self):
        """Test parsing non-existent file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            CatalogConfig.from_yaml("/nonexistent/path.yml")

    def test_from_yaml_empty_file(self, tmp_path):
        """Test parsing empty YAML file."""
        config_file = tmp_path / "empty.yml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="Config file is empty"):
            CatalogConfig.from_yaml(config_file)

    def test_from_yaml_valid_config_simple(self, tmp_path):
        """Test parsing simple valid YAML config (no catalog section needed)."""
        yaml_content = """
namespace1:
  table1:
    dataset: org/repo1
"""
        config_file = tmp_path / "simple.yml"
        config_file.write_text(yaml_content)

        config = CatalogConfig.from_yaml(config_file)
        assert len(config.namespaces) == 1
        assert config.namespaces[0].name == "namespace1"

        with pytest.raises(ValueError, match="Missing 'location' in catalog config"):
            CatalogConfig.from_yaml(config_file)

    def test_from_yaml_no_namespaces(self, tmp_path):
        """Test parsing YAML with no namespaces defined."""
        yaml_content = """
catalog:
  name: test_catalog
  location: .faceberg/
"""
        config_file = tmp_path / "no_namespaces.yml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="No namespaces defined"):
            CatalogConfig.from_yaml(config_file)

    def test_from_yaml_invalid_namespace_name(self, tmp_path):
        """Test parsing YAML with invalid namespace name (special characters)."""
        yaml_content = """
catalog:
  name: test_catalog
  location: .faceberg/

namespace@invalid:
  table1:
    dataset: org/repo1
"""
        config_file = tmp_path / "invalid_ns.yml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="Invalid namespace name"):
            CatalogConfig.from_yaml(config_file)

    def test_from_yaml_namespace_not_dict(self, tmp_path):
        """Test parsing YAML where namespace value is not a dict."""
        yaml_content = """
namespace1: not_a_dict
"""
        config_file = tmp_path / "ns_not_dict.yml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="must be a dict of tables"):
            CatalogConfig.from_yaml(config_file)

    def test_from_yaml_empty_namespace(self, tmp_path):
        """Test parsing YAML with namespace that has no tables."""
        yaml_content = """
namespace1: {}
"""
        config_file = tmp_path / "empty_ns.yml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="has no tables defined"):
            CatalogConfig.from_yaml(config_file)

    def test_from_yaml_missing_dataset(self, tmp_path):
        """Test parsing YAML with table missing 'dataset' field."""
        yaml_content = """
namespace1:
  table1:
    config: config1
"""
        config_file = tmp_path / "missing_dataset.yml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="Missing 'dataset' in namespace1.table1"):
            CatalogConfig.from_yaml(config_file)

    def test_from_yaml_valid_namespace_names(self, tmp_path):
        """Test parsing YAML with various valid namespace names."""
        yaml_content = """
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

        config = CatalogConfig.from_yaml(config_file)
        assert len(config.namespaces) == 4
        namespace_names = [ns.name for ns in config.namespaces]
        assert "namespace_1" in namespace_names
        assert "namespace-2" in namespace_names
        assert "Namespace3" in namespace_names
        assert "ns123" in namespace_names

    def test_to_yaml(self, tmp_path):
        """Test exporting config to YAML."""
        table1 = TableConfig(name="table1", dataset="org/repo1", config="config1")
        table2 = TableConfig(name="table2", dataset="org/repo2", config="default")
        namespace = NamespaceConfig(name="test_ns", tables=[table1, table2])

        config = CatalogConfig(namespaces=[namespace])

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
        config1 = CatalogConfig.from_yaml(input_file)
        config1.to_yaml(output_file)
        config2 = CatalogConfig.from_yaml(output_file)

        # Verify they're equivalent
        assert len(config1.namespaces) == len(config2.namespaces)

        for ns1, ns2 in zip(config1.namespaces, config2.namespaces):
            assert ns1.name == ns2.name
            assert len(ns1.tables) == len(ns2.tables)

            for t1, t2 in zip(ns1.tables, ns2.tables):
                assert t1.name == t2.name
                assert t1.dataset == t2.dataset
                assert t1.config == t2.config

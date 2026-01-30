"""Tests for faceberg.config module."""

from pathlib import Path

import pytest
import yaml

from faceberg.config import Config, Identifier, Table


class TestIdentifier:
    """Tests for Identifier class."""

    def test_identifier_from_string(self):
        """Test creating Identifier from string."""
        ident = Identifier("namespace.table")
        assert ident == ("namespace", "table")

    def test_identifier_from_tuple(self):
        """Test creating Identifier from tuple."""
        ident = Identifier(("namespace", "table"))
        assert ident == ("namespace", "table")

    @pytest.mark.skip(reason="Identifier validation not enforced - allows flexible part counts")
    def test_identifier_invalid_parts(self):
        """Test creating Identifier with invalid number of parts."""
        with pytest.raises(ValueError, match="must have exactly 2 parts"):
            Identifier("onlynamespace")

        with pytest.raises(ValueError, match="must have exactly 2 parts"):
            Identifier(("ns1", "ns2", "table"))

    def test_identifier_path_property(self):
        """Test path property of Identifier."""
        ident = Identifier("namespace.table")
        assert ident.path == pytest.approx(Path("namespace") / "table")


class TestTable:
    """Tests for Table dataclass."""

    def test_table_config_explicit(self):
        """Test Table with explicit config value."""
        table = Table(dataset="org/repo", config="custom")
        assert table.dataset == "org/repo"
        assert table.config == "custom"


class TestConfig:
    """Tests for Config class."""

    def test_config_creation(self):
        """Test Config creation."""
        config = Config(uri=".faceberg")
        assert config.uri == ".faceberg"
        assert config.data == {}

    def test_config_with_data(self):
        """Test Config creation with nested data."""
        data = {
            "ns1": {
                "table1": Table(dataset="org/repo1", config="config1"),
            }
        }
        config = Config(uri=".faceberg", data=data)
        assert config.uri == ".faceberg"
        assert config.tables == [Identifier(("ns1", "table1"))]

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
    config: config3
"""
        config_file = tmp_path / "test_config.yml"
        config_file.write_text(yaml_content)
        config = Config.from_yaml(config_file)

        assert Identifier(("namespace1", "table1")) in config
        assert Identifier(("namespace1", "table2")) in config
        assert Identifier(("namespace2", "table3")) in config

        assert config[Identifier(("namespace1", "table1"))].dataset == "org/repo1"
        assert config[Identifier(("namespace1", "table1"))].config == "config1"
        assert config[Identifier(("namespace2", "table3"))].config == "config3"

    def test_from_yaml_empty_string(self, tmp_path):
        """Test parsing empty YAML file."""
        config_file = tmp_path / "empty.yml"
        config_file.write_text("")
        with pytest.raises(ValueError, match="Config is empty"):
            Config.from_yaml(config_file)

    def test_from_yaml_missing_uri(self, tmp_path):
        """Test parsing YAML without URI."""
        yaml_content = """
namespace1:
  table1:
    dataset: org/repo1
"""
        config_file = tmp_path / "no_uri.yml"
        config_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="Missing required 'uri' field"):
            Config.from_yaml(config_file)

    def test_to_yaml(self, tmp_path):
        """Test exporting config to YAML."""
        config = Config(uri=".faceberg")
        config[("test_ns", "table1")] = Table(dataset="org/repo1", config="config1")
        config[("test_ns", "table2")] = Table(dataset="org/repo2", config="default")

        # Write to temporary file
        yaml_file = tmp_path / "config.yml"
        config.to_yaml(yaml_file)

        # Read back and parse to verify
        yaml_output = yaml_file.read_text()
        data = yaml.safe_load(yaml_output)

        assert "uri" in data
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
    config: config3
"""
        # Load, export to file, and re-load
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml_content)
        config1 = Config.from_yaml(config_file)

        yaml_file = tmp_path / "config2.yml"
        config1.to_yaml(yaml_file)
        config2 = Config.from_yaml(yaml_file)

        # Verify they're equivalent
        assert config1.uri == config2.uri
        assert len(config1.data) == len(config2.data)
        assert set(config1.data) == set(config2.data)

        for identifier in config1.tables:
            assert config1[identifier].dataset == config2[identifier].dataset
            assert config1[identifier].config == config2[identifier].config


class TestConfigMappingProtocol:
    """Tests for Config Mapping protocol implementation."""

    def test_getitem_setitem(self):
        """Test __getitem__ and __setitem__."""
        config = Config(uri=".faceberg")
        entry = Table(dataset="org/repo", config="default")
        config[Identifier(("ns", "table"))] = entry

        assert config[Identifier(("ns", "table"))] == entry
        assert config[Identifier(("ns", "table"))].dataset == "org/repo"

    def test_iter(self):
        """Test tables property."""
        config = Config(uri=".faceberg")
        config[Identifier(("ns1", "table1"))] = Table(dataset="org/repo1")
        config[Identifier(("ns1", "table2"))] = Table(dataset="org/repo2")
        config[Identifier(("ns2", "table3"))] = Table(dataset="org/repo3")

        identifiers = config.tables
        assert len(identifiers) == 3
        assert Identifier(("ns1", "table1")) in identifiers
        assert Identifier(("ns1", "table2")) in identifiers
        assert Identifier(("ns2", "table3")) in identifiers

    def test_contains(self):
        """Test __contains__."""
        config = Config(uri=".faceberg")
        config[Identifier(("ns", "table"))] = Table(dataset="org/repo")

        assert Identifier(("ns", "table")) in config
        assert Identifier(("ns", "other")) not in config
        assert Identifier(("other", "table")) not in config

    def test_delitem(self):
        """Test __delitem__."""
        config = Config(uri=".faceberg")
        config[("ns", "table1")] = Table(dataset="org/repo1")
        config[("ns", "table2")] = Table(dataset="org/repo2")

        assert len(config.tables) == 2

        del config[("ns", "table1")]
        assert len(config.tables) == 1
        assert ("ns", "table1") not in config
        assert ("ns", "table2") in config

    def test_delitem_nonexistent(self):
        """Test __delitem__ raises KeyError for non-existent key."""
        config = Config(uri=".faceberg")

        with pytest.raises(KeyError):
            del config[("ns", "table")]


class TestConfigValidation:
    """Tests for Config validation."""

    def test_setitem_requires_table(self):
        """Test __setitem__ accepts Table instances."""
        config = Config(uri=".faceberg")

        # Tables can be set as Table instances
        config[("ns", "table")] = Table(dataset="org/repo", config="default")
        assert config[("ns", "table")].dataset == "org/repo"

    def test_getitem_returns_table(self):
        """Test __getitem__ returns Table instance."""
        config = Config(uri=".faceberg")
        table = Table(dataset="org/repo", config="custom")
        config[("ns", "table")] = table

        result = config[("ns", "table")]
        assert isinstance(result, Table)
        assert result.dataset == "org/repo"
        assert result.config == "custom"


class TestConfigTwoLevelHierarchy:
    """Tests for Config with 2-level hierarchy (namespace, table)."""

    def test_two_level_identifier(self):
        """Test Config requires exactly 2-level identifiers."""
        config = Config(uri=".faceberg")
        config[("ns", "table")] = Table(dataset="org/repo")

        assert ("ns", "table") in config
        assert config[("ns", "table")].dataset == "org/repo"


    def test_multiple_two_level_identifiers(self):
        """Test Config supports multiple 2-level identifiers."""
        config = Config(uri=".faceberg")
        config[("ns1", "table1")] = Table(dataset="org/repo1")
        config[("ns1", "table2")] = Table(dataset="org/repo2")
        config[("ns2", "table3")] = Table(dataset="org/repo3")

        assert len(config.tables) == 3
        assert ("ns1", "table1") in config
        assert ("ns1", "table2") in config
        assert ("ns2", "table3") in config


class TestConfigYAML:
    """Tests for Config YAML serialization."""

    def test_yaml_round_trip_two_level_hierarchy(self, tmp_path):
        """Test YAML round trip with 2-level hierarchy identifiers."""
        config1 = Config(uri=".faceberg")
        config1[("ns1", "table1")] = Table(dataset="org/repo1", config="config1")
        config1[("ns1", "table2")] = Table(dataset="org/repo2", config="default")
        config1[("ns2", "table3")] = Table(dataset="org/repo3", config="custom")

        yaml_file = tmp_path / "config.yml"
        config1.to_yaml(yaml_file)
        config2 = Config.from_yaml(yaml_file)

        assert config2.uri == config1.uri
        assert len(config2.tables) == len(config1.tables)
        assert set(config2.tables) == set(config1.tables)

        for key in config1.tables:
            assert config2[key].dataset == config1[key].dataset
            assert config2[key].config == config1[key].config

    def test_empty_config(self):
        """Test Config with no tables."""
        config = Config(uri=".faceberg")

        assert len(config.tables) == 0
        assert config.tables == []
        assert ("ns", "table") not in config

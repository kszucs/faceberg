"""Tests for faceberg.config module."""

from collections.abc import Mapping

import pytest
import yaml

from faceberg.config import Config, Entry


class TestEntry:
    """Tests for Entry dataclass."""

    def test_entry_config_defaults(self):
        """Test Entry with default config value."""
        entry = Entry(dataset="org/repo")
        assert entry.dataset == "org/repo"
        assert entry.config == "default"

    def test_entry_config_explicit(self):
        """Test Entry with explicit config value."""
        entry = Entry(dataset="org/repo", config="custom")
        assert entry.dataset == "org/repo"
        assert entry.config == "custom"

    def test_entry_minimal_fields(self):
        """Test Entry only requires dataset field."""
        entry = Entry(dataset="test")
        assert entry.dataset == "test"
        assert entry.config == "default"


class TestConfig:
    """Tests for Config class."""

    def test_config_creation(self):
        """Test Config creation."""
        config = Config(uri=".faceberg")
        assert config.uri == ".faceberg"
        assert len(config) == 0

    def test_config_with_data(self):
        """Test Config creation with nested data."""
        data = {
            "ns1": {
                "table1": Entry(dataset="org/repo1", config="config1"),
            }
        }
        config = Config(uri=".faceberg", data=data)
        assert config.uri == ".faceberg"
        assert len(config) == 1

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

        config = Config.from_yaml(config_file)

        assert len(config) == 3
        assert ("namespace1", "table1") in config
        assert ("namespace1", "table2") in config
        assert ("namespace2", "table3") in config

        assert config[("namespace1", "table1")].dataset == "org/repo1"
        assert config[("namespace1", "table1")].config == "config1"
        assert config[("namespace2", "table3")].config == "default"

    def test_from_yaml_missing_file(self):
        """Test parsing non-existent file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Config.from_yaml("/nonexistent/path.yml")

    def test_from_yaml_empty_file(self, tmp_path):
        """Test parsing empty YAML file."""
        config_file = tmp_path / "empty.yml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="Config file is empty"):
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
        config[("test_ns", "table1")] = Entry(dataset="org/repo1", config="config1")
        config[("test_ns", "table2")] = Entry(dataset="org/repo2", config="default")

        output_file = tmp_path / "output.yml"
        config.to_yaml(output_file)

        # Read back and verify
        with open(output_file) as f:
            data = yaml.safe_load(f)

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
"""
        input_file = tmp_path / "input.yml"
        input_file.write_text(yaml_content)

        output_file = tmp_path / "output.yml"

        # Load, export, and re-load
        config1 = Config.from_yaml(input_file)
        config1.to_yaml(output_file)
        config2 = Config.from_yaml(output_file)

        # Verify they're equivalent
        assert config1.uri == config2.uri
        assert len(config1) == len(config2)
        assert set(config1) == set(config2)

        for identifier in config1:
            assert config1[identifier].dataset == config2[identifier].dataset
            assert config1[identifier].config == config2[identifier].config


class TestConfigMappingProtocol:
    """Tests for Config Mapping protocol implementation."""

    def test_config_is_mapping(self):
        """Test Config implements mapping protocol."""
        config = Config(uri=".faceberg")
        assert isinstance(config, Mapping)

    def test_getitem_setitem(self):
        """Test __getitem__ and __setitem__."""
        config = Config(uri=".faceberg")
        entry = Entry(dataset="org/repo", config="default")
        config[("ns", "table")] = entry

        assert config[("ns", "table")] == entry
        assert config[("ns", "table")].dataset == "org/repo"

    def test_len(self):
        """Test __len__."""
        config = Config(uri=".faceberg")
        assert len(config) == 0

        config[("ns1", "table1")] = Entry(dataset="org/repo1")
        assert len(config) == 1

        config[("ns1", "table2")] = Entry(dataset="org/repo2")
        assert len(config) == 2

        config[("ns2", "table3")] = Entry(dataset="org/repo3")
        assert len(config) == 3

    def test_iter(self):
        """Test __iter__."""
        config = Config(uri=".faceberg")
        config[("ns1", "table1")] = Entry(dataset="org/repo1")
        config[("ns1", "table2")] = Entry(dataset="org/repo2")
        config[("ns2", "table3")] = Entry(dataset="org/repo3")

        identifiers = list(config)
        assert len(identifiers) == 3
        assert ("ns1", "table1") in identifiers
        assert ("ns1", "table2") in identifiers
        assert ("ns2", "table3") in identifiers

    def test_contains(self):
        """Test __contains__."""
        config = Config(uri=".faceberg")
        config[("ns", "table")] = Entry(dataset="org/repo")

        assert ("ns", "table") in config
        assert ("ns", "other") not in config
        assert ("other", "table") not in config

    def test_delitem(self):
        """Test __delitem__."""
        config = Config(uri=".faceberg")
        config[("ns", "table1")] = Entry(dataset="org/repo1")
        config[("ns", "table2")] = Entry(dataset="org/repo2")

        assert len(config) == 2

        del config[("ns", "table1")]
        assert len(config) == 1
        assert ("ns", "table1") not in config
        assert ("ns", "table2") in config

    def test_delitem_nonexistent(self):
        """Test __delitem__ raises KeyError for non-existent key."""
        config = Config(uri=".faceberg")

        with pytest.raises(KeyError):
            del config[("ns", "table")]


class TestConfigValidation:
    """Tests for Config validation."""

    def test_setitem_requires_entry(self):
        """Test __setitem__ requires Entry instance."""
        config = Config(uri=".faceberg")

        with pytest.raises(ValueError, match="Value must be an Entry"):
            config[("ns", "table")] = {"dataset": "org/repo", "config": "default"}

        with pytest.raises(ValueError, match="Value must be an Entry"):
            config[("ns", "table")] = "not an entry"

    def test_getitem_returns_entry(self):
        """Test __getitem__ returns Entry instance."""
        config = Config(uri=".faceberg")
        entry = Entry(dataset="org/repo", config="custom")
        config[("ns", "table")] = entry

        result = config[("ns", "table")]
        assert isinstance(result, Entry)
        assert result.dataset == "org/repo"
        assert result.config == "custom"


class TestConfigArbitraryDepth:
    """Tests for Config with arbitrary identifier depth."""

    def test_single_level_identifier(self):
        """Test Config supports single-level identifiers."""
        config = Config(uri=".faceberg")
        config[("table",)] = Entry(dataset="org/repo")

        assert ("table",) in config
        assert config[("table",)].dataset == "org/repo"

    def test_two_level_identifier(self):
        """Test Config supports two-level identifiers."""
        config = Config(uri=".faceberg")
        config[("ns", "table")] = Entry(dataset="org/repo")

        assert ("ns", "table") in config
        assert config[("ns", "table")].dataset == "org/repo"

    def test_three_level_identifier(self):
        """Test Config supports three-level identifiers."""
        config = Config(uri=".faceberg")
        config[("ns1", "ns2", "table")] = Entry(dataset="org/repo")

        assert ("ns1", "ns2", "table") in config
        assert config[("ns1", "ns2", "table")].dataset == "org/repo"

    def test_mixed_depth_identifiers(self):
        """Test Config supports mixed depth identifiers."""
        config = Config(uri=".faceberg")
        config[("table1",)] = Entry(dataset="org/repo1")
        config[("ns", "table2")] = Entry(dataset="org/repo2")
        config[("ns1", "ns2", "table3")] = Entry(dataset="org/repo3")

        assert len(config) == 3
        assert ("table1",) in config
        assert ("ns", "table2") in config
        assert ("ns1", "ns2", "table3") in config


class TestConfigYAML:
    """Tests for Config YAML serialization."""

    def test_yaml_round_trip_arbitrary_depth(self, tmp_path):
        """Test YAML round trip with arbitrary depth identifiers."""
        config1 = Config(uri=".faceberg")
        config1[("table1",)] = Entry(dataset="org/repo1", config="config1")
        config1[("ns", "table2")] = Entry(dataset="org/repo2", config="default")
        config1[("ns1", "ns2", "table3")] = Entry(dataset="org/repo3", config="custom")

        yaml_file = tmp_path / "config.yml"
        config1.to_yaml(yaml_file)

        config2 = Config.from_yaml(yaml_file)

        assert config2.uri == config1.uri
        assert len(config2) == len(config1)
        assert set(config2) == set(config1)

        for key in config1:
            assert config2[key].dataset == config1[key].dataset
            assert config2[key].config == config1[key].config

    def test_empty_config(self):
        """Test Config with no tables."""
        config = Config(uri=".faceberg")

        assert len(config) == 0
        assert list(config) == []
        assert ("ns", "table") not in config

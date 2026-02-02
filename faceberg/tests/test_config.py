"""Tests for faceberg.config module."""

import pytest

from faceberg.config import Config, Dataset, Namespace, Node, Table, View


@pytest.fixture
def sample_config():
    """Fixture that returns a fresh config dict for each test."""
    return {
        "ns1": {
            "table1": {
                "type": "dataset",
                "repo": "org/dataset1",
                "config": "config1",
            },
            "table2": {
                "type": "dataset",
                "repo": "org/dataset2",
                "config": "config2",
            },
        },
        "ns2": {
            "view1": {
                "type": "view",
                "query": "SELECT * FROM ns1.table1",
            },
        },
        "ns3": {
            "subns1": {
                "table3": {
                    "type": "dataset",
                    "repo": "org/dataset3",
                    "config": "config3",
                }
            }
        },
    }


def test_config(sample_config):
    cfg = Config.from_dict(sample_config)

    assert isinstance(cfg, Config)
    for k, v in cfg.items():
        assert isinstance(v, Namespace)

    # Verify to_dict includes type discriminators
    cfg_dict = cfg.to_dict()
    assert cfg_dict["ns1"]["table1"]["type"] == "dataset"
    assert cfg_dict["ns2"]["view1"]["type"] == "view"

    # Verify data access
    assert cfg[("ns1", "table1")].repo == "org/dataset1"
    assert cfg[("ns1", "table2")].config == "config2"
    assert cfg[("ns2", "view1")].query == "SELECT * FROM ns1.table1"
    assert cfg[("ns3", "subns1", "table3")].repo == "org/dataset3"

    # Verify mutation
    cfg[("ns1", "table1")].repo = "org/updated_dataset1"
    assert cfg[("ns1", "table1")].repo == "org/updated_dataset1"

    # Check datasets() method
    datasets = cfg.datasets()
    assert datasets == {
        ("ns1", "table1"): cfg[("ns1", "table1")],
        ("ns1", "table2"): cfg[("ns1", "table2")],
        ("ns3", "subns1", "table3"): cfg[("ns3", "subns1", "table3")],
    }


def test_config_root(sample_config):
    cfg = Config.from_dict(sample_config)
    assert cfg[()] is cfg


def test_config_contains(sample_config):
    """Test __contains__ method for membership testing."""
    cfg = Config.from_dict(sample_config)

    # Test with string keys
    assert "ns1" in cfg
    assert "ns2" in cfg
    assert "nonexistent" not in cfg
    assert ("ns1", "table1") in cfg
    assert ("ns1", "table999") not in cfg

    # Test with tuple keys
    assert ("ns1",) in cfg
    assert ("ns1", "table1") in cfg
    assert ("ns3", "subns1", "table3") in cfg
    assert ("ns1", "nonexistent") not in cfg
    assert ("nonexistent", "table") not in cfg


def test_config_setitem_creates_intermediate_namespaces():
    """Test that __setitem__ creates intermediate Namespace objects as needed."""
    cfg = Config()

    # Set nested item without creating intermediates first
    new_dataset = Dataset(repo="org/new_dataset", config="new_config")
    cfg[("analytics", "sales", "orders")] = new_dataset

    # Verify intermediate namespaces were created
    assert isinstance(cfg[("analytics",)], Namespace)
    assert isinstance(cfg[("analytics", "sales")], Namespace)
    assert cfg[("analytics", "sales", "orders")].repo == "org/new_dataset"


def test_config_setitem_overwrite(sample_config):
    """Test overwriting existing items."""
    cfg = Config.from_dict(sample_config)

    # Overwrite existing dataset
    new_dataset = Dataset(repo="org/replaced", config="new_config")
    cfg[("ns1", "table1")] = new_dataset
    assert cfg[("ns1", "table1")].repo == "org/replaced"

    # Overwrite with different node type
    new_view = View(query="SELECT * FROM replaced")
    cfg["ns1.table1"] = new_view
    assert isinstance(cfg["ns1.table1"], View)
    assert cfg["ns1.table1"].query == "SELECT * FROM replaced"


def test_config_getitem_keyerror(sample_config):
    """Test that accessing non-existent keys raises KeyError."""
    cfg = Config.from_dict(sample_config)

    with pytest.raises(KeyError):
        _ = cfg["nonexistent"]

    with pytest.raises(KeyError):
        _ = cfg["ns1.nonexistent"]

    with pytest.raises(KeyError):
        _ = cfg[("nonexistent", "nested")]


def test_config_invalid_key_type(sample_config):
    """Test that invalid key types raise TypeError."""
    cfg = Config.from_dict(sample_config)

    with pytest.raises(TypeError, match="Key must be a tuple or string"):
        _ = cfg[123]

    with pytest.raises(TypeError, match="Key must be a tuple or string"):
        _ = cfg[["list", "key"]]

    with pytest.raises(TypeError, match="Key must be a tuple or string"):
        cfg[123] = Dataset(repo="test", config="default")


def test_config_empty():
    """Test empty config initialization and operations."""
    cfg = Config()
    assert cfg.to_dict() == {}

    # Add first item
    cfg["first"] = Namespace()
    assert "first" in cfg
    assert isinstance(cfg["first"], Namespace)


def test_config_repr():
    """Test Config string representation."""
    cfg = Config()
    cfg["test"] = Namespace()
    repr_str = repr(cfg)
    assert "Config" in repr_str


def test_yaml_round_trip(tmp_path, sample_config):
    """Test saving and loading config from YAML file with type preservation."""
    cfg = Config.from_dict(sample_config)
    yaml_path = tmp_path / "test_config.yaml"

    # Save to YAML
    cfg.to_yaml(yaml_path)
    assert yaml_path.exists()

    # Verify YAML content has type discriminators
    yaml_content = yaml_path.read_text()
    assert "type: dataset" in yaml_content
    assert "type: view" in yaml_content

    # Load from YAML and verify round-trip
    loaded_cfg = Config.from_yaml(yaml_path)
    assert loaded_cfg.to_dict() == cfg.to_dict()

    # Verify data integrity and types
    assert isinstance(loaded_cfg[("ns1", "table1")], Dataset)
    assert loaded_cfg[("ns1", "table1")].repo == "org/dataset1"
    assert isinstance(loaded_cfg[("ns2", "view1")], View)
    assert loaded_cfg[("ns2", "view1")].query == "SELECT * FROM ns1.table1"


def test_yaml_empty_file(tmp_path):
    """Test loading from empty YAML file."""
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("")

    cfg = Config.from_yaml(yaml_path)
    assert cfg.to_dict() == {}


def test_yaml_preserves_all_types(tmp_path):
    """Test that YAML serialization preserves all node types."""
    cfg = Config()
    cfg[("data", "dataset1")] = Dataset(repo="org/repo1", config="cfg1")
    cfg[("data", "view1")] = View(query="SELECT 1")
    cfg[("data", "table1")] = Table(uri="")

    yaml_path = tmp_path / "types.yaml"
    cfg.to_yaml(yaml_path)

    # Verify YAML has all type discriminators
    yaml_content = yaml_path.read_text()
    assert "type: dataset" in yaml_content
    assert "type: view" in yaml_content
    assert "type: table" in yaml_content

    # Load and verify all types are preserved
    loaded = Config.from_yaml(yaml_path)
    assert isinstance(loaded[("data", "dataset1")], Dataset)
    assert isinstance(loaded[("data", "view1")], View)
    assert isinstance(loaded[("data", "table1")], Table)
    assert loaded[("data", "dataset1")].repo == "org/repo1"
    assert loaded[("data", "view1")].query == "SELECT 1"


def test_node_from_dict_table():
    """Test Node.from_dict for Table type."""
    data = {"type": "table", "uri": ""}
    node = Node.from_dict(data)
    assert isinstance(node, Table)


def test_node_from_dict_dataset():
    """Test Node.from_dict for Dataset type."""
    data = {"type": "dataset", "repo": "org/dataset", "config": "default"}
    node = Node.from_dict(data)
    assert isinstance(node, Dataset)
    assert node.repo == "org/dataset"
    assert node.config == "default"


def test_node_from_dict_dataset_default_config():
    """Test Dataset with default config value."""
    data = {"type": "dataset", "repo": "org/dataset"}
    node = Node.from_dict(data)
    assert isinstance(node, Dataset)
    assert node.repo == "org/dataset"
    assert node.config == "default"  # Default value


def test_node_from_dict_view():
    """Test Node.from_dict for View type."""
    data = {"type": "view", "query": "SELECT * FROM table"}
    node = Node.from_dict(data)
    assert isinstance(node, View)
    assert node.query == "SELECT * FROM table"


def test_node_from_dict_namespace():
    """Test Node.from_dict for Namespace type (explicit and implicit)."""
    # Explicit type
    data = {"type": "namespace", "child1": {"type": "table", "uri": ""}}
    node = Node.from_dict(data)
    assert isinstance(node, Namespace)
    assert isinstance(node["child1"], Table)

    # Implicit type (no type field defaults to namespace)
    data = {"child1": {"type": "table", "uri": ""}, "child2": {"type": "view", "query": "SELECT 1"}}
    node = Node.from_dict(data)
    assert isinstance(node, Namespace)
    assert isinstance(node["child1"], Table)
    assert isinstance(node["child2"], View)


def test_node_from_dict_unknown_type():
    """Test Node.from_dict raises ValueError for unknown types."""
    data = {"type": "unknown_type"}
    with pytest.raises(ValueError, match="Unknown node type: unknown_type"):
        Node.from_dict(data)


def test_node_from_dict_invalid_input():
    """Test Node.from_dict raises TypeError for non-dict input."""
    with pytest.raises(TypeError, match="Expected dict to deserialize"):
        Node.from_dict("not a dict")

    with pytest.raises(TypeError, match="Expected dict to deserialize"):
        Node.from_dict(123)

    with pytest.raises(TypeError, match="Expected dict to deserialize"):
        Node.from_dict(["list", "input"])


def test_node_to_dict():
    """Test Node.to_dict for different node types includes type discriminators."""
    dataset = Dataset(repo="org/repo", config="cfg")
    assert dataset.to_dict() == {"repo": "org/repo", "config": "cfg", "type": "dataset"}

    view = View(query="SELECT * FROM table")
    assert view.to_dict() == {"query": "SELECT * FROM table", "type": "view"}

    table = Table(uri="file://data/location")
    assert table.to_dict() == {"uri": "file://data/location", "type": "table"}


def test_namespace_repr():
    """Test Namespace string representation."""
    ns = Namespace()
    ns["child"] = Table(uri="")
    repr_str = repr(ns)
    assert "Namespace" in repr_str


def test_namespace_dict_behavior():
    """Test that Namespace behaves like a dict."""
    ns = Namespace()
    ns["key1"] = Dataset(repo="org/repo1", config="cfg1")
    ns["key2"] = View(query="SELECT 1")

    assert len(ns) == 2
    assert "key1" in ns
    assert "key2" in ns
    assert list(ns.keys()) == ["key1", "key2"]


def test_config_from_dict_with_empty():
    """Test Config.from_dict with empty dict creates empty config."""
    cfg = Config.from_dict({})
    assert cfg.to_dict() == {}


def test_complex_nested_structure():
    """Test deeply nested namespace structure."""
    cfg = Config()
    cfg[("level1", "level2", "level3", "level4", "dataset")] = Dataset(
        repo="org/deep", config="nested"
    )

    assert isinstance(cfg[("level1",)], Namespace)
    assert isinstance(cfg[("level1", "level2")], Namespace)
    assert isinstance(cfg[("level1", "level2", "level3")], Namespace)
    assert isinstance(cfg[("level1", "level2", "level3", "level4")], Namespace)
    assert cfg[("level1", "level2", "level3", "level4", "dataset")].repo == "org/deep"


def test_mixed_access_patterns(sample_config):
    """Test mixing tuple and string access in same config."""
    cfg = Config.from_dict(sample_config)

    # Access same item with different patterns
    assert cfg[("ns1", "table1")] is cfg[("ns1", "table1")]
    assert cfg[("ns3", "subns1")] is cfg[("ns3", "subns1")]

    # Set with tuple, read with string
    cfg[("new", "item")] = Dataset(repo="org/test", config="default")
    assert cfg[("new", "item")].repo == "org/test"

    # Set with string, read with tuple
    cfg[("another", "item")] = View(query="SELECT 2")
    assert cfg[("another", "item")].query == "SELECT 2"

import io
import time

from rich.console import Console
from rich.tree import Tree

from faceberg import config as cfg
from faceberg.pretty import TableState, node, progress_bars, progress_tree, tree


def test_table_state_creation():
    """Test TableState dataclass creation with defaults."""
    state = TableState()

    assert state.kind == "pending"
    assert state.progress is None
    assert state.error is None


def test_table_state_with_all_fields():
    """Test TableState with all fields specified."""
    state = TableState(kind="in_progress", progress=50, error="test error")

    assert state.kind == "in_progress"
    assert state.progress == 50
    assert state.error == "test error"


def test_table_state_icon_property():
    """Test that TableState.icon returns correct icons for different states."""
    assert TableState(kind="pending").icon == "⏳"
    assert TableState(kind="in_progress").icon == "▶️"
    assert TableState(kind="complete").icon == "✓"
    assert TableState(kind="up_to_date").icon == "✓"
    assert TableState(kind="needs_update").icon == "↻"


def test_table_state_color_property():
    """Test that TableState.color returns correct colors for different states."""
    assert TableState(kind="pending").color == "dim white"
    assert TableState(kind="in_progress").color == "yellow"
    assert TableState(kind="complete").color == "green"
    assert TableState(kind="up_to_date").color == "dim green"
    assert TableState(kind="needs_update").color == "blue"


def test_tree_empty_config():
    """Test tree building with empty config."""
    config = cfg.Config()
    result = tree(config)

    assert isinstance(result, Tree)
    assert result.hide_root is True


def test_tree_with_namespace():
    """Test tree building with namespace."""
    config = cfg.Config()
    config["default"] = cfg.Namespace()

    result = tree(config)

    assert isinstance(result, Tree)
    # Should have one child (the namespace)
    assert len(result.children) == 1
    label_str = str(result.children[0].label)
    assert "default" in label_str
    assert "cyan" in label_str
    assert "namespace" in label_str


def test_tree_with_table():
    """Test tree building with table node."""
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["my_table"] = cfg.Table()

    result = tree(config)

    assert isinstance(result, Tree)
    # Navigate to the table node
    namespace_node = result.children[0]
    assert len(namespace_node.children) == 1
    table_label = str(namespace_node.children[0].label)
    assert "my_table" in table_label
    assert "green" in table_label
    assert "table" in table_label


def test_tree_with_dataset():
    """Test tree building with dataset node."""
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["imdb"] = cfg.Dataset(repo="stanfordnlp/imdb", config="plain_text")

    result = tree(config)

    assert isinstance(result, Tree)
    namespace_node = result.children[0]
    dataset_label = str(namespace_node.children[0].label)
    assert "imdb" in dataset_label
    assert "stanfordnlp/imdb" in dataset_label
    assert "yellow" in dataset_label
    assert "dataset" in dataset_label


def test_tree_with_view():
    """Test tree building with view node."""
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["my_view"] = cfg.View(query="SELECT * FROM table WHERE condition = true")

    result = tree(config)

    assert isinstance(result, Tree)
    namespace_node = result.children[0]
    view_label = str(namespace_node.children[0].label)
    assert "my_view" in view_label
    assert "SELECT" in view_label
    assert "blue" in view_label
    assert "view" in view_label


def test_tree_with_states():
    """Test tree building with state tracking."""
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["test_table"] = cfg.Table()

    states = {("default", "test_table"): TableState(kind="in_progress", progress=50)}

    result = tree(config, states)

    assert isinstance(result, Tree)
    namespace_node = result.children[0]
    table_label = str(namespace_node.children[0].label)
    assert "▶️" in table_label  # In progress icon
    assert "yellow" in table_label  # In progress color


def test_tree_with_error_state():
    """Test tree building with error state."""
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["failed_table"] = cfg.Table()

    states = {("default", "failed_table"): TableState(kind="complete", error="Connection timeout")}

    result = tree(config, states)

    assert isinstance(result, Tree)
    namespace_node = result.children[0]
    table_node = namespace_node.children[0]
    # Error should be added as a child node
    assert len(table_node.children) == 1
    error_label = str(table_node.children[0].label)
    assert "Connection timeout" in error_label
    assert "red" in error_label


def test_node_namespace():
    """Test node function with namespace."""
    ns = cfg.Namespace()
    ns["child_table"] = cfg.Table()
    parent = Tree("root")
    states = {}

    node(ns, ("default",), parent, states)

    assert len(parent.children) == 1
    namespace_label = str(parent.children[0].label)
    assert "default" in namespace_label
    assert "cyan" in namespace_label
    # Should have one child (the table)
    assert len(parent.children[0].children) == 1


def test_node_table():
    """Test node function with table."""
    table = cfg.Table()
    parent = Tree("root")
    states = {}

    node(table, ("default", "my_table"), parent, states)

    assert len(parent.children) == 1
    table_label = str(parent.children[0].label)
    assert "my_table" in table_label
    assert "green" in table_label


def test_node_dataset():
    """Test node function with dataset."""
    dataset = cfg.Dataset(repo="org/repo", config="default")
    parent = Tree("root")
    states = {}

    node(dataset, ("default", "my_dataset"), parent, states)

    assert len(parent.children) == 1
    dataset_label = str(parent.children[0].label)
    assert "my_dataset" in dataset_label
    assert "org/repo" in dataset_label
    assert "yellow" in dataset_label


def test_node_view():
    """Test node function with view."""
    view = cfg.View(query="SELECT * FROM table")
    parent = Tree("root")
    states = {}

    node(view, ("default", "my_view"), parent, states)

    assert len(parent.children) == 1
    view_label = str(parent.children[0].label)
    assert "my_view" in view_label
    assert "SELECT" in view_label
    assert "blue" in view_label


def test_node_view_long_query():
    """Test node function with view that has long query."""
    long_query = "SELECT * FROM table WHERE condition = true AND another_condition = false"
    view = cfg.View(query=long_query)
    parent = Tree("root")
    states = {}

    node(view, ("default", "my_view"), parent, states)

    assert len(parent.children) == 1
    view_label = str(parent.children[0].label)
    # Should truncate query to 30 chars + "..."
    assert "..." in view_label
    assert len(long_query) > 30  # Verify our test query is actually long


def test_progress_tree_context_manager():
    """Test progress_tree context manager."""
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["test_table"] = cfg.Table()

    string_io = io.StringIO()
    console = Console(file=string_io, force_terminal=True, width=80)

    with progress_tree(config, console) as updater:
        # Update state
        updater(("default", "test_table"), "in_progress", percent=50)
        time.sleep(0.1)  # Give time for live display to render

        updater(("default", "test_table"), "complete")
        time.sleep(0.1)

    # Verify output was generated
    output = string_io.getvalue()
    assert len(output) > 0


def test_progress_tree_updater_with_error():
    """Test progress_tree updater with error."""
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["test_table"] = cfg.Table()

    string_io = io.StringIO()
    console = Console(file=string_io, force_terminal=True, width=80)

    with progress_tree(config, console) as updater:
        updater(("default", "test_table"), "complete", error="Test error")
        time.sleep(0.1)

    output = string_io.getvalue()
    assert len(output) > 0


def test_progress_bars_context_manager():
    """Test progress_bars context manager."""
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["dataset1"] = cfg.Dataset(repo="org/repo1", config="default")
    config["default"]["dataset2"] = cfg.Dataset(repo="org/repo2", config="default")

    identifiers = [("default", "dataset1"), ("default", "dataset2")]

    string_io = io.StringIO()
    console = Console(file=string_io, force_terminal=True, width=80)

    with progress_bars(config, console, identifiers) as updater:
        # Update progress for first dataset
        updater(("default", "dataset1"), "in_progress", percent=30, stage="Processing")
        time.sleep(0.1)

        # Update progress for second dataset
        updater(("default", "dataset2"), "in_progress", percent=60, stage="Almost done")
        time.sleep(0.1)

        # Complete both
        updater(("default", "dataset1"), "complete")
        updater(("default", "dataset2"), "complete")
        time.sleep(0.1)

    output = string_io.getvalue()
    assert len(output) > 0


def test_progress_bars_updater_stages():
    """Test progress_bars updater with different stages."""
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["test"] = cfg.Dataset(repo="org/repo", config="default")

    identifiers = [("default", "test")]

    string_io = io.StringIO()
    console = Console(file=string_io, force_terminal=True, width=80)

    with progress_bars(config, console, identifiers) as updater:
        updater(("default", "test"), "pending")
        time.sleep(0.05)

        updater(("default", "test"), "in_progress", percent=50, stage="Downloading")
        time.sleep(0.05)

        updater(("default", "test"), "complete", percent=100)
        time.sleep(0.05)

    output = string_io.getvalue()
    assert len(output) > 0


def test_nested_namespaces():
    """Test tree building with nested namespaces."""
    config = cfg.Config()
    config["level1"] = cfg.Namespace()
    config["level1"]["level2"] = cfg.Namespace()
    config["level1"]["level2"]["table"] = cfg.Table()

    result = tree(config)

    assert isinstance(result, Tree)
    # Navigate through the hierarchy
    level1_node = result.children[0]
    assert "level1" in str(level1_node.label)

    level2_node = level1_node.children[0]
    assert "level2" in str(level2_node.label)

    table_node = level2_node.children[0]
    assert "table" in str(table_node.label)


def test_multiple_tables_in_namespace():
    """Test tree building with multiple tables in a namespace."""
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["table1"] = cfg.Table()
    config["default"]["table2"] = cfg.Table()
    config["default"]["table3"] = cfg.Table()

    result = tree(config)

    assert isinstance(result, Tree)
    namespace_node = result.children[0]
    # Should have three children (the tables)
    assert len(namespace_node.children) == 3

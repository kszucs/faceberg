from faceberg.pretty import StateKind, TableState, CatalogTreeView, tree
from faceberg.catalog import Identifier, LocalCatalog
from faceberg import config as cfg
from rich.tree import Tree
from rich.console import Console
import io
import time


def test_state_kind_has_correct_values():
    """Test that StateKind enum has correct icon and color attributes."""
    assert StateKind.PENDING.icon == "⏳"
    assert StateKind.PENDING.color == "dim white"

    assert StateKind.IN_PROGRESS.icon == "▶️"
    assert StateKind.IN_PROGRESS.color == "yellow"

    assert StateKind.COMPLETE.icon == "✓"
    assert StateKind.COMPLETE.color == "green"

    assert StateKind.UP_TO_DATE.icon == "✓"
    assert StateKind.UP_TO_DATE.color == "dim green"

    assert StateKind.FAILED.icon == "✗"
    assert StateKind.FAILED.color == "red"

    assert StateKind.NEEDS_UPDATE.icon == "↻"
    assert StateKind.NEEDS_UPDATE.color == "blue"


def test_table_state_creation():
    """Test TableState dataclass creation with defaults."""
    state = TableState()

    assert state.kind == StateKind.PENDING
    assert state.progress is None
    assert state.error is None


def test_table_state_with_all_fields():
    """Test TableState with all fields specified."""
    state = TableState(
        kind=StateKind.IN_PROGRESS,
        progress=50,
        error="test error"
    )

    assert state.kind == StateKind.IN_PROGRESS
    assert state.progress == 50
    assert state.error == "test error"


def test_tree_namespace_node():
    """Test tree building for namespace node."""
    identifier = Identifier("myns")
    node = cfg.Namespace()

    result = tree(node, identifier, None, {})

    assert isinstance(result, Tree)
    label_str = str(result.label)
    assert "🗂️" in label_str
    assert "myns" in label_str
    assert "[cyan]" in label_str


def test_tree_dataset_node():
    """Test tree building for dataset node with metadata."""
    identifier = Identifier(("default", "imdb"))
    node = cfg.Dataset(repo="stanfordnlp/imdb", config="plain_text")

    result = tree(node, identifier, None, {})

    assert isinstance(result, Tree)
    label_str = str(result.label)
    assert "🤗" in label_str
    assert "imdb" in label_str
    assert "stanfordnlp/imdb" in label_str
    assert "[yellow]" in label_str


def test_tree_view_node():
    """Test tree building for view node with query snippet."""
    identifier = Identifier(("default", "my_view"))
    node = cfg.View(query="SELECT * FROM table WHERE condition = true")

    result = tree(node, identifier, None, {})

    assert isinstance(result, Tree)
    label_str = str(result.label)
    assert "👁️" in label_str
    assert "my_view" in label_str
    assert "SELECT" in label_str
    assert "[blue]" in label_str


def test_tree_table_node():
    """Test tree building for table node."""
    identifier = Identifier(("default", "my_table"))
    node = cfg.Table()

    result = tree(node, identifier, None, {})

    assert isinstance(result, Tree)
    label_str = str(result.label)
    assert "📊" in label_str
    assert "my_table" in label_str
    assert "[green]" in label_str


def test_tree_dataset_with_state():
    """Test tree building for dataset node with state tracking."""
    identifier = Identifier(("default", "test"))
    node = cfg.Dataset(repo="org/repo", config="default")
    state = TableState(
        kind=StateKind.IN_PROGRESS,
        progress=45
    )
    states = {identifier: state}

    result = tree(node, identifier, None, states)

    assert isinstance(result, Tree)
    label_str = str(result.label)
    assert "▶️" in label_str  # In progress icon
    assert "[45%]" in label_str  # Progress percentage


def test_tree_dataset_with_error():
    """Test tree building for dataset node with error state."""
    identifier = Identifier(("default", "failed"))
    node = cfg.Dataset(repo="org/repo", config="default")
    state = TableState(
        kind=StateKind.FAILED,
        error="Connection timeout"
    )
    states = {identifier: state}

    # Need a parent to see error messages
    parent = Tree("root")
    tree(node, identifier, parent, states)

    # Check that error was added
    assert len(parent.children) > 0
    # The first child should be the node
    child = parent.children[0]
    label_str = str(child.label)
    assert "✗" in label_str  # Failed icon


def test_catalog_tree_view_initialization(tmp_path):
    """Test CatalogTreeView initialization."""
    catalog_dir = tmp_path / "test_catalog"
    catalog = LocalCatalog(
        name="test",
        uri=f"file://{catalog_dir}"
    )
    catalog.init()

    tree_view = CatalogTreeView(catalog)

    assert tree_view.catalog == catalog
    assert tree_view.states == {}
    assert tree_view.config is not None


def test_build_tree_empty_catalog(tmp_path):
    """Test building tree from empty catalog."""
    catalog_dir = tmp_path / "test_catalog"
    catalog = LocalCatalog(
        name="test",
        uri=f"file://{catalog_dir}"
    )
    catalog.init()

    tree_view = CatalogTreeView(catalog)
    root_label = f"[bold cyan]📁 {catalog.name}[/bold cyan]"
    root = tree(tree_view.config, Identifier(()), None, tree_view.states, root_label)

    assert isinstance(root, Tree)
    assert "test" in str(root.label)


def test_build_tree_with_namespace_and_dataset(tmp_path):
    """Test building tree with namespace and dataset."""
    catalog_dir = tmp_path / "test_catalog"
    catalog = LocalCatalog(
        name="test",
        uri=f"file://{catalog_dir}"
    )
    catalog.init()

    # Create config with namespace and dataset
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["test_table"] = cfg.Dataset(repo="org/repo", config="default")

    # Save config manually
    config.to_yaml(catalog_dir / "faceberg.yml")

    tree_view = CatalogTreeView(catalog)
    root_label = f"[bold cyan]📁 {catalog.name}[/bold cyan]"
    root = tree(tree_view.config, Identifier(()), None, tree_view.states, root_label)

    # Verify tree structure by rendering to string
    string_io = io.StringIO()
    console = Console(file=string_io, force_terminal=True, width=80)
    console.print(root)
    tree_str = string_io.getvalue()

    assert "default" in tree_str
    assert "test_table" in tree_str


def test_update_state():
    """Test updating table state."""
    # Create a mock catalog (we'll use a simple approach)
    class MockCatalog:
        def __init__(self):
            self.name = "mock"
            self._config = cfg.Config()

        def config(self):
            return self._config

    catalog = MockCatalog()
    tree_view = CatalogTreeView(catalog)

    identifier = Identifier("default.test")
    tree_view.states[identifier] = TableState()

    # Update state
    tree_view.update_state(identifier, StateKind.IN_PROGRESS, progress=50)

    assert tree_view.states[identifier].kind == StateKind.IN_PROGRESS
    assert tree_view.states[identifier].progress == 50

    # Update with error
    tree_view.update_state(identifier, StateKind.FAILED, error="Test error")

    assert tree_view.states[identifier].kind == StateKind.FAILED
    assert tree_view.states[identifier].error == "Test error"


def test_catalog_tree_view_context_manager(tmp_path):
    """Test CatalogTreeView as context manager with live display."""
    catalog_dir = tmp_path / "test_catalog"
    catalog = LocalCatalog(
        name="test",
        uri=f"file://{catalog_dir}"
    )

    # Create config with dataset
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["test"] = cfg.Dataset(repo="org/repo", config="default")
    catalog.init()
    config.to_yaml(catalog_dir / "faceberg.yml")

    # Capture console output
    string_io = io.StringIO()
    console = Console(file=string_io, force_terminal=True, width=80)

    # Create tree view with console
    tree_view = CatalogTreeView(catalog, console=console)
    identifier = Identifier(("default", "test"))
    tree_view.states[identifier] = TableState()

    # Use CatalogTreeView as context manager
    with tree_view:
        # Update state
        tree_view.update_state(identifier, StateKind.IN_PROGRESS, progress=50)
        # Give time for live display to render
        time.sleep(0.1)

    # Verify live display was used
    output = string_io.getvalue()
    assert len(output) > 0  # Should have some output


def test_catalog_tree_view_with_console():
    """Test CatalogTreeView with custom console."""
    # Use mock catalog for simplicity
    class MockCatalog:
        def __init__(self):
            self.name = "mock"
            self._config = cfg.Config()

        def config(self):
            return self._config

    catalog = MockCatalog()

    # Create tree view with custom console
    string_io = io.StringIO()
    console = Console(file=string_io, force_terminal=True, width=80)
    tree_view = CatalogTreeView(catalog, console=console)

    identifier = Identifier("test")
    tree_view.states[identifier] = TableState()

    # Test update_state method
    tree_view.update_state(identifier, StateKind.COMPLETE, progress=100)

    assert tree_view.states[identifier].kind == StateKind.COMPLETE
    assert tree_view.states[identifier].progress == 100
    assert tree_view.console == console

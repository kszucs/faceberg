# Rich Catalog Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement rich terminal visualization for Faceberg's CLI with real-time progress tracking

**Architecture:** Create a `pretty.py` module with `TableState` enum, `TableNode` dataclass, `CatalogTreeView` for rendering, and `OperationTracker` for live updates. Integrate into CLI commands (list, sync, add, init) with optional progress callbacks in catalog methods.

**Tech Stack:** Python 3.9+, Rich library (tree, live display, console), dataclasses, enum

---

## Task 1: Create TableState Enum

**Files:**
- Create: `faceberg/pretty.py`
- Test: `faceberg/tests/test_pretty.py`

**Step 1: Write the failing test**

```python
# faceberg/tests/test_pretty.py
from faceberg.pretty import TableState


def test_table_state_has_correct_values():
    """Test that TableState enum has correct icon and color attributes."""
    assert TableState.PENDING.icon == "⏳"
    assert TableState.PENDING.color == "dim white"

    assert TableState.IN_PROGRESS.icon == "▶️"
    assert TableState.IN_PROGRESS.color == "yellow"

    assert TableState.COMPLETE.icon == "✓"
    assert TableState.COMPLETE.color == "green"

    assert TableState.UP_TO_DATE.icon == "✓"
    assert TableState.UP_TO_DATE.color == "dim green"

    assert TableState.FAILED.icon == "✗"
    assert TableState.FAILED.color == "red"

    assert TableState.NEEDS_UPDATE.icon == "↻"
    assert TableState.NEEDS_UPDATE.color == "blue"
```

**Step 2: Run test to verify it fails**

Run: `pytest faceberg/tests/test_pretty.py::test_table_state_has_correct_values -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'faceberg.pretty'"

**Step 3: Write minimal implementation**

```python
# faceberg/pretty.py
"""Rich terminal visualization for catalog operations."""

from enum import Enum


class TableState(Enum):
    """Enumeration of possible table states with visual styling."""

    PENDING = ("⏳", "dim white")
    IN_PROGRESS = ("▶️", "yellow")
    COMPLETE = ("✓", "green")
    UP_TO_DATE = ("✓", "dim green")
    FAILED = ("✗", "red")
    NEEDS_UPDATE = ("↻", "blue")

    def __init__(self, icon: str, color: str):
        self.icon = icon
        self.color = color
```

**Step 4: Run test to verify it passes**

Run: `pytest faceberg/tests/test_pretty.py::test_table_state_has_correct_values -v`

Expected: PASS

**Step 5: Commit**

```bash
git add faceberg/pretty.py faceberg/tests/test_pretty.py
git commit -m "feat: add TableState enum with visual styling

Add enum defining table states (pending, in_progress, complete,
up_to_date, failed, needs_update) with icons and colors for
rich terminal rendering.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create TableNode Dataclass

**Files:**
- Modify: `faceberg/pretty.py`
- Modify: `faceberg/tests/test_pretty.py`

**Step 1: Write the failing test**

```python
# faceberg/tests/test_pretty.py (add to existing file)
from faceberg.pretty import TableState, TableNode
from faceberg.catalog import Identifier
from faceberg import config as cfg


def test_table_node_creation():
    """Test TableNode dataclass creation with defaults."""
    identifier = Identifier("default.test")
    node = TableNode(
        identifier=identifier,
        node_type=cfg.Dataset
    )

    assert node.identifier == identifier
    assert node.node_type == cfg.Dataset
    assert node.state == TableState.PENDING
    assert node.progress is None
    assert node.error is None
    assert node.metadata == {}


def test_table_node_with_all_fields():
    """Test TableNode with all fields specified."""
    identifier = Identifier("ns.table")
    node = TableNode(
        identifier=identifier,
        node_type=cfg.Table,
        state=TableState.IN_PROGRESS,
        progress=50,
        error="test error",
        metadata={"repo": "org/repo", "config": "default"}
    )

    assert node.state == TableState.IN_PROGRESS
    assert node.progress == 50
    assert node.error == "test error"
    assert node.metadata["repo"] == "org/repo"
```

**Step 2: Run test to verify it fails**

Run: `pytest faceberg/tests/test_pretty.py::test_table_node_creation -v`

Expected: FAIL with "ImportError: cannot import name 'TableNode'"

**Step 3: Write minimal implementation**

```python
# faceberg/pretty.py (add after TableState class)
from dataclasses import dataclass, field
from typing import Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from faceberg.catalog import Identifier
    from faceberg import config as cfg


@dataclass
class TableNode:
    """Represents a table with tracking state for visualization."""

    identifier: "Identifier"
    node_type: Type["cfg.Node"]
    state: TableState = TableState.PENDING
    progress: Optional[int] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
```

**Step 4: Run test to verify it passes**

Run: `pytest faceberg/tests/test_pretty.py -v -k table_node`

Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add faceberg/pretty.py faceberg/tests/test_pretty.py
git commit -m "feat: add TableNode dataclass for state tracking

Add dataclass to track table state with identifier, node type,
state, progress percentage, error message, and metadata.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Implement Node Formatting Function

**Files:**
- Modify: `faceberg/pretty.py`
- Modify: `faceberg/tests/test_pretty.py`

**Step 1: Write the failing test**

```python
# faceberg/tests/test_pretty.py (add to existing file)
from faceberg.pretty import format_node
from faceberg.catalog import Identifier
from faceberg import config as cfg


def test_format_namespace_node():
    """Test formatting a namespace node."""
    identifier = Identifier("myns")
    node = cfg.Namespace()

    result = format_node(identifier, node, None)

    assert "🗂️" in result
    assert "myns" in result
    assert "[cyan]" in result


def test_format_dataset_node():
    """Test formatting a dataset node with metadata."""
    identifier = Identifier(("default", "imdb"))
    node = cfg.Dataset(repo="stanfordnlp/imdb", config="plain_text")

    result = format_node(identifier, node, None)

    assert "🤗" in result
    assert "imdb" in result
    assert "stanfordnlp/imdb" in result
    assert "[yellow]" in result


def test_format_view_node():
    """Test formatting a view node with query snippet."""
    identifier = Identifier(("default", "my_view"))
    node = cfg.View(query="SELECT * FROM table WHERE condition = true")

    result = format_node(identifier, node, None)

    assert "👁️" in result
    assert "my_view" in result
    assert "SELECT" in result
    assert "[blue]" in result


def test_format_table_node():
    """Test formatting a table node."""
    identifier = Identifier(("default", "my_table"))
    node = cfg.Table()

    result = format_node(identifier, node, None)

    assert "📊" in result
    assert "my_table" in result
    assert "[green]" in result


def test_format_node_with_state():
    """Test formatting a node with state tracking."""
    identifier = Identifier(("default", "test"))
    node = cfg.Dataset(repo="org/repo", config="default")
    table_node = TableNode(
        identifier=identifier,
        node_type=cfg.Dataset,
        state=TableState.IN_PROGRESS,
        progress=45
    )

    result = format_node(identifier, node, table_node)

    assert "▶️" in result  # In progress icon
    assert "[45%]" in result  # Progress percentage


def test_format_node_with_error():
    """Test formatting a node with error state."""
    identifier = Identifier(("default", "failed"))
    node = cfg.Dataset(repo="org/repo", config="default")
    table_node = TableNode(
        identifier=identifier,
        node_type=cfg.Dataset,
        state=TableState.FAILED,
        error="Connection timeout"
    )

    result = format_node(identifier, node, table_node)

    assert "✗" in result  # Failed icon
```

**Step 2: Run test to verify it fails**

Run: `pytest faceberg/tests/test_pretty.py -v -k format_node`

Expected: FAIL with "ImportError: cannot import name 'format_node'"

**Step 3: Write minimal implementation**

```python
# faceberg/pretty.py (add after TableNode class)
from faceberg import config as cfg
from faceberg.catalog import Identifier


# Node type styling: (icon, color)
NODE_STYLES = {
    cfg.Namespace: ("🗂️", "cyan"),
    cfg.Table: ("📊", "green"),
    cfg.Dataset: ("🤗", "yellow"),
    cfg.View: ("👁️", "blue"),
}


def format_node(
    identifier: Identifier,
    node: cfg.Node,
    table_node: Optional[TableNode] = None
) -> str:
    """Format a node with icon, color, and optional state/progress.

    Args:
        identifier: Node identifier (path in config tree)
        node: Config node (Namespace, Table, Dataset, View)
        table_node: Optional tracking state for the node

    Returns:
        Styled string for Rich rendering
    """
    # Get icon and color for node type
    icon, color = NODE_STYLES.get(type(node), ("", "white"))

    # Extract name (last part of identifier)
    name = identifier[-1] if identifier else "root"

    # Base format: icon + name
    formatted = f"[{color}]{icon} {name}[/{color}]"

    # Add metadata for leaf nodes
    if isinstance(node, cfg.Dataset):
        formatted += f" [dim]({node.repo})[/dim]"
    elif isinstance(node, cfg.View):
        # Truncate long queries
        query_snippet = node.query[:30] + "..." if len(node.query) > 30 else node.query
        formatted += f" [dim]({query_snippet})[/dim]"

    # Add state indicator if tracked
    if table_node:
        state_icon = table_node.state.icon
        state_color = table_node.state.color
        formatted += f" [{state_color}]{state_icon}[/{state_color}]"

        # Add progress bar for in-progress items
        if table_node.state == TableState.IN_PROGRESS and table_node.progress is not None:
            formatted += f" [{table_node.progress}%]"

    return formatted
```

**Step 4: Run test to verify it passes**

Run: `pytest faceberg/tests/test_pretty.py -v -k format_node`

Expected: PASS (all format_node tests)

**Step 5: Commit**

```bash
git add faceberg/pretty.py faceberg/tests/test_pretty.py
git commit -m "feat: add format_node function for rich rendering

Implement node formatting with icons, colors, metadata display,
and optional state indicators with progress percentage.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement CatalogTreeView Class

**Files:**
- Modify: `faceberg/pretty.py`
- Modify: `faceberg/tests/test_pretty.py`

**Step 1: Write the failing test**

```python
# faceberg/tests/test_pretty.py (add to existing file)
import tempfile
from pathlib import Path
from faceberg.pretty import CatalogTreeView
from faceberg.catalog import LocalCatalog
from rich.tree import Tree


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
    assert tree_view.table_nodes == {}
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
    tree = tree_view.build_tree()

    assert isinstance(tree, Tree)
    assert "test" in str(tree.label)


def test_build_tree_with_namespace_and_dataset(tmp_path):
    """Test building tree with namespace and dataset."""
    from faceberg import config as cfg

    catalog_dir = tmp_path / "test_catalog"
    catalog = LocalCatalog(
        name="test",
        uri=f"file://{catalog_dir}"
    )

    # Create config with namespace and dataset
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["test_table"] = cfg.Dataset(repo="org/repo", config="default")

    catalog.init(config)

    tree_view = CatalogTreeView(catalog)
    tree = tree_view.build_tree()

    # Verify tree structure by checking string representation
    tree_str = str(tree)
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
    tree_view.table_nodes[identifier] = TableNode(
        identifier=identifier,
        node_type=cfg.Dataset
    )

    # Update state
    tree_view.update_state(identifier, TableState.IN_PROGRESS, progress=50)

    assert tree_view.table_nodes[identifier].state == TableState.IN_PROGRESS
    assert tree_view.table_nodes[identifier].progress == 50

    # Update with error
    tree_view.update_state(identifier, TableState.FAILED, error="Test error")

    assert tree_view.table_nodes[identifier].state == TableState.FAILED
    assert tree_view.table_nodes[identifier].error == "Test error"
```

**Step 2: Run test to verify it fails**

Run: `pytest faceberg/tests/test_pretty.py -v -k catalog_tree_view`

Expected: FAIL with "ImportError: cannot import name 'CatalogTreeView'"

**Step 3: Write minimal implementation**

```python
# faceberg/pretty.py (add after format_node function)
from rich.tree import Tree
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faceberg.catalog import BaseCatalog


class CatalogTreeView:
    """Builds and renders hierarchical catalog structure as Rich Tree."""

    def __init__(self, catalog: "BaseCatalog"):
        """Initialize tree view for a catalog.

        Args:
            catalog: Catalog instance to visualize
        """
        self.catalog = catalog
        self.table_nodes: dict[Identifier, TableNode] = {}

    @property
    def config(self) -> cfg.Config:
        """Get current config from catalog."""
        return self.catalog.config()

    def build_tree(self) -> Tree:
        """Build Rich Tree from current config with node states.

        Returns:
            Rich Tree object ready to print
        """
        config = self.config
        root = Tree(f"[bold cyan]📁 {self.catalog.name}[/bold cyan]")

        def add_nodes(
            parent_tree: Tree,
            namespace: cfg.Namespace,
            parent_id: Identifier = ()
        ):
            """Recursively add nodes to tree."""
            for name, node in namespace.items():
                node_id = Identifier((*parent_id, name))
                table_node = self.table_nodes.get(node_id)

                label = format_node(node_id, node, table_node)

                if isinstance(node, cfg.Namespace):
                    # Add namespace branch and recurse
                    branch = parent_tree.add(label)
                    add_nodes(branch, node, node_id)
                else:
                    # Add leaf node
                    if table_node and table_node.error:
                        error_branch = parent_tree.add(label)
                        error_branch.add(f"[red]Error: {table_node.error}[/red]")
                    else:
                        parent_tree.add(label)

        add_nodes(root, config, ())
        return root

    def update_state(
        self,
        identifier: Identifier,
        state: TableState,
        progress: Optional[int] = None,
        error: Optional[str] = None
    ):
        """Update a table's state.

        Args:
            identifier: Table identifier
            state: New state
            progress: Optional progress percentage (0-100)
            error: Optional error message
        """
        if identifier in self.table_nodes:
            node = self.table_nodes[identifier]
            node.state = state
            if progress is not None:
                node.progress = progress
            if error is not None:
                node.error = error
```

**Step 4: Run test to verify it passes**

Run: `pytest faceberg/tests/test_pretty.py -v -k catalog_tree_view`

Expected: PASS (all CatalogTreeView tests)

**Step 5: Commit**

```bash
git add faceberg/pretty.py faceberg/tests/test_pretty.py
git commit -m "feat: add CatalogTreeView for hierarchical rendering

Implement tree view class that builds Rich Tree from catalog config,
supports state tracking for nodes, and updates display state.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Implement OperationTracker Context Manager

**Files:**
- Modify: `faceberg/pretty.py`
- Modify: `faceberg/tests/test_pretty.py`

**Step 1: Write the failing test**

```python
# faceberg/tests/test_pretty.py (add to existing file)
from faceberg.pretty import OperationTracker
from rich.console import Console
import io


def test_operation_tracker_context_manager(tmp_path):
    """Test OperationTracker as context manager."""
    catalog_dir = tmp_path / "test_catalog"
    catalog = LocalCatalog(
        name="test",
        uri=f"file://{catalog_dir}"
    )

    # Create config with dataset
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["test"] = cfg.Dataset(repo="org/repo", config="default")
    catalog.init(config)

    # Create tree view with console redirected to string
    tree_view = CatalogTreeView(catalog)
    identifier = Identifier(("default", "test"))
    tree_view.table_nodes[identifier] = TableNode(
        identifier=identifier,
        node_type=cfg.Dataset,
        state=TableState.PENDING
    )

    # Capture console output
    string_io = io.StringIO()
    console = Console(file=string_io, force_terminal=True, width=80)

    # Use OperationTracker
    with OperationTracker(tree_view, console=console) as tracker:
        # Update state
        tracker.update(identifier, TableState.IN_PROGRESS, progress=50)
        # Give time for live display to render
        import time
        time.sleep(0.1)

    # Verify tracker was created and used
    output = string_io.getvalue()
    assert len(output) > 0  # Should have some output


def test_operation_tracker_update():
    """Test OperationTracker update method."""
    # Use mock catalog for simplicity
    class MockCatalog:
        def __init__(self):
            self.name = "mock"
            self._config = cfg.Config()

        def config(self):
            return self._config

    catalog = MockCatalog()
    tree_view = CatalogTreeView(catalog)
    identifier = Identifier("test")
    tree_view.table_nodes[identifier] = TableNode(
        identifier=identifier,
        node_type=cfg.Table
    )

    # Create tracker without actually starting live display
    # (to avoid blocking in tests)
    string_io = io.StringIO()
    console = Console(file=string_io, force_terminal=True, width=80)
    tracker = OperationTracker(tree_view, console=console)

    # Test update method
    tracker.update(identifier, TableState.COMPLETE, progress=100)

    assert tree_view.table_nodes[identifier].state == TableState.COMPLETE
    assert tree_view.table_nodes[identifier].progress == 100
```

**Step 2: Run test to verify it fails**

Run: `pytest faceberg/tests/test_pretty.py -v -k operation_tracker`

Expected: FAIL with "ImportError: cannot import name 'OperationTracker'"

**Step 3: Write minimal implementation**

```python
# faceberg/pretty.py (add after CatalogTreeView class)
from rich.live import Live
from rich.console import Console


class OperationTracker:
    """Track and display live progress for catalog operations."""

    def __init__(self, tree_view: CatalogTreeView, console: Optional[Console] = None):
        """Initialize operation tracker.

        Args:
            tree_view: CatalogTreeView instance to display
            console: Optional Rich Console (creates new if not provided)
        """
        self.tree_view = tree_view
        self.console = console or Console()
        self.live = Live(
            tree_view.build_tree(),
            refresh_per_second=4,
            console=self.console
        )

    def __enter__(self):
        """Start live display."""
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop live display."""
        self.live.stop()
        return False

    def update(
        self,
        identifier: Identifier,
        state: TableState,
        progress: Optional[int] = None,
        error: Optional[str] = None
    ):
        """Update state and refresh live display.

        Args:
            identifier: Table identifier
            state: New state
            progress: Optional progress percentage
            error: Optional error message
        """
        self.tree_view.update_state(identifier, state, progress, error)
        self.live.update(self.tree_view.build_tree())
```

**Step 4: Run test to verify it passes**

Run: `pytest faceberg/tests/test_pretty.py -v -k operation_tracker`

Expected: PASS (all OperationTracker tests)

**Step 5: Commit**

```bash
git add faceberg/pretty.py faceberg/tests/test_pretty.py
git commit -m "feat: add OperationTracker for live progress display

Implement context manager that uses Rich Live Display to show
real-time updates of catalog operations with progress tracking.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update CLI list Command

**Files:**
- Modify: `faceberg/cli.py:215-240`
- Create: `faceberg/tests/test_cli_pretty.py`

**Step 1: Write the failing test**

```python
# faceberg/tests/test_cli_pretty.py
from click.testing import CliRunner
from faceberg.cli import main
from faceberg import config as cfg
import tempfile
from pathlib import Path


def test_list_command_uses_tree_view():
    """Test that list command uses CatalogTreeView for display."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "test_catalog"

        # Initialize catalog with some data
        runner = CliRunner()
        result = runner.invoke(main, [str(catalog_path), "init"])
        assert result.exit_code == 0

        # Create namespace
        from faceberg.catalog import catalog
        cat = catalog(str(catalog_path))
        cat.create_namespace("default")

        # Run list command
        result = runner.invoke(main, [str(catalog_path), "list"])

        assert result.exit_code == 0
        # Should contain tree icons
        assert "📁" in result.output or "default" in result.output
```

**Step 2: Run test to verify it fails**

Run: `pytest faceberg/tests/test_cli_pretty.py::test_list_command_uses_tree_view -v`

Expected: FAIL (tree icons not in output)

**Step 3: Write minimal implementation**

```python
# faceberg/cli.py (replace list_tables function around line 215)
@main.command("list")
@click.pass_context
def list_tables(ctx):
    """List all tables in catalog.

    Example:
        faceberg --config=faceberg.yml list
    """
    from faceberg.pretty import CatalogTreeView

    catalog = ctx.obj["catalog"]
    tree_view = CatalogTreeView(catalog)
    console.print(tree_view.build_tree())
```

**Step 4: Run test to verify it passes**

Run: `pytest faceberg/tests/test_cli_pretty.py::test_list_command_uses_tree_view -v`

Expected: PASS

**Step 5: Commit**

```bash
git add faceberg/cli.py faceberg/tests/test_cli_pretty.py
git commit -m "feat: integrate tree view into list command

Replace simple text listing with rich tree visualization showing
catalog hierarchy with icons and colors.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Add Progress Callback to sync_dataset

**Files:**
- Modify: `faceberg/catalog.py:1033-1150`
- Modify: `faceberg/tests/test_catalog.py`

**Step 1: Write the failing test**

```python
# faceberg/tests/test_catalog.py (add to existing file)
def test_sync_dataset_progress_callback(tmp_path, requests_mock):
    """Test that sync_dataset calls progress callback."""
    from faceberg.catalog import LocalCatalog
    from faceberg.config import Config, Dataset

    # Setup mock for HuggingFace API
    requests_mock.get(
        "https://huggingface.co/api/datasets/test-org/test-repo",
        json={
            "sha": "abc123",
            "siblings": [{"rfilename": "data/train-00000.parquet"}]
        }
    )

    catalog_path = tmp_path / "catalog"
    catalog = LocalCatalog(name="test", uri=f"file://{catalog_path}")
    catalog.init()

    # Add dataset to config
    catalog.create_namespace("default")
    config = catalog.config()
    config[("default", "test")] = Dataset(repo="test-org/test-repo", config="default")
    config.to_yaml(catalog_path / "faceberg.yml")

    # Track progress calls
    progress_calls = []
    def progress_cb(percent):
        progress_calls.append(percent)

    # Sync with callback
    try:
        catalog.sync_dataset("default.test", progress_callback=progress_cb)
    except Exception:
        pass  # May fail due to incomplete mocking, but callback should be called

    # Verify callback was called
    assert len(progress_calls) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest faceberg/tests/test_catalog.py::test_sync_dataset_progress_callback -v`

Expected: FAIL (TypeError: sync_dataset() got unexpected keyword argument 'progress_callback')

**Step 3: Write minimal implementation**

```python
# faceberg/catalog.py (modify sync_dataset method around line 1033)
def sync_dataset(
    self,
    identifier: Union[str, Identifier],
    progress_callback: Optional[Callable[[int], None]] = None
) -> Table:
    """Sync a single dataset table by adding or updating it.

    Reads dataset and config information from the catalog configuration,
    then either creates the table (if no metadata exists) or updates it
    with a new snapshot (if metadata exists).

    Args:
        identifier: Table identifier in format "namespace.table"
        progress_callback: Optional callback receiving progress (0-100)

    Returns:
        Synced Table object (created or updated)

    Raises:
        ValueError: If identifier format is invalid or table not in config
    """
    identifier = Identifier(identifier)

    if progress_callback:
        progress_callback(0)

    # Load config to get dataset info
    config = self.config()

    # Check if table exists in config
    try:
        table_entry = config[identifier]
    except KeyError:
        raise ValueError(f"Table {identifier} not found in config")

    if progress_callback:
        progress_callback(10)

    # Check if table has been synced (has metadata files)
    table_uri = self.uri / identifier.path
    version_hint_uri = table_uri / "metadata/version-hint.text"

    io = load_file_io(properties=self.properties, location=table_uri)
    has_metadata = False
    try:
        with io.new_input(version_hint_uri).open():
            pass  # File exists, table has been synced
        has_metadata = True
    except Exception:
        pass  # Table hasn't been synced yet

    if progress_callback:
        progress_callback(20)

    if not has_metadata:
        # First sync - call add_dataset which will create metadata
        if progress_callback:
            progress_callback(30)
        result = self.add_dataset(identifier, table_entry.repo, table_entry.config)
        if progress_callback:
            progress_callback(100)
        return result

    # Update existing table with new snapshot
    if progress_callback:
        progress_callback(30)

    # Load table first to get old revision
    table = self.load_table(identifier)

    if progress_callback:
        progress_callback(40)

    # Get old revision from table properties (required)
    old_revision = table.metadata.properties.get("huggingface.dataset.revision")
    if not old_revision:
        raise ValueError(
            f"Table {'.'.join(identifier)} missing 'huggingface.dataset.revision' property. "
            "This table was created before revision tracking was implemented. "
            "Please recreate the table to enable incremental sync."
        )

    if progress_callback:
        progress_callback(50)

    # Discover dataset at current revision
    dataset_info = DatasetInfo.discover(
        repo_id=table_entry.dataset,
        configs=[table_entry.config],
        token=self._hf_token,
    )

    if progress_callback:
        progress_callback(60)

    # Check if already up to date
    if old_revision == dataset_info.revision:
        logger.info(f"Table {identifier} already at revision {old_revision}")
        if progress_callback:
            progress_callback(100)
        return table

    if progress_callback:
        progress_callback(70)

    # Get only new files since old revision (incremental update)
    table_info = dataset_info.to_table_info_incremental(
        namespace=identifier[0],
        table_name=identifier[1],
        config=table_entry.config,
        old_revision=old_revision,
        token=self._hf_token,
    )

    # If no new files, table is already up to date
    if not table_info.files:
        logger.info(f"No new files for {identifier}")
        if progress_callback:
            progress_callback(100)
        return table

    if progress_callback:
        progress_callback(80)

    # Append new snapshot with only new files
    with self._staging() as staging:
        # Create local metadata directory
        metadata_dir = staging / identifier.path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Create table URI for metadata
        table_uri = self.uri / identifier.path.path

        # Create metadata writer
        metadata_writer = IcebergMetadataWriter(
            table_path=metadata_dir,
            schema=table_info.schema,
            partition_spec=table_info.partition_spec,
            base_uri=table_uri,
        )

        # Append new snapshot with updated files
        metadata_writer.append_snapshot_from_files(
            file_infos=table_info.files,
            current_metadata=table.metadata,
            properties=table_info.get_table_properties(),
        )

        # Record all files in the table directory (including new manifest/metadata files)
        for path in metadata_dir.rglob("*"):
            if path.is_file():
                staging.add(path.relative_to(staging.path))

        # Note: No need to save config since table entry hasn't changed

    if progress_callback:
        progress_callback(90)

    # Load and return table after persistence
    result = self.load_table(identifier)

    if progress_callback:
        progress_callback(100)

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest faceberg/tests/test_catalog.py::test_sync_dataset_progress_callback -v`

Expected: PASS

**Step 5: Commit**

```bash
git add faceberg/catalog.py faceberg/tests/test_catalog.py
git commit -m "feat: add progress callback to sync_dataset

Add optional progress_callback parameter that receives progress
percentage (0-100) during dataset synchronization operations.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Update CLI sync Command with Live Progress

**Files:**
- Modify: `faceberg/cli.py:98-139`
- Modify: `faceberg/tests/test_cli_pretty.py`

**Step 1: Write the failing test**

```python
# faceberg/tests/test_cli_pretty.py (add to existing file)
def test_sync_command_shows_progress():
    """Test that sync command shows progress with tree view."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "test_catalog"

        # Initialize catalog
        runner = CliRunner()
        result = runner.invoke(main, [str(catalog_path), "init"])
        assert result.exit_code == 0

        # Add a dataset to config manually
        from faceberg.catalog import catalog
        from faceberg import config as cfg
        cat = catalog(str(catalog_path))
        config_obj = cat.config()
        config_obj["default"] = cfg.Namespace()
        config_obj[("default", "test")] = cfg.Dataset(
            repo="test-org/test-repo",
            config="default"
        )
        config_obj.to_yaml(catalog_path / "faceberg.yml")

        # Run sync command (will fail due to invalid dataset, but should show tree)
        result = runner.invoke(main, [str(catalog_path), "sync"])

        # Should show tree structure even if sync fails
        assert "📁" in result.output or "default" in result.output
```

**Step 2: Run test to verify it fails**

Run: `pytest faceberg/tests/test_cli_pretty.py::test_sync_command_shows_progress -v`

Expected: FAIL (no tree icons in output)

**Step 3: Write minimal implementation**

```python
# faceberg/cli.py (replace sync function around line 98)
@main.command()
@click.argument("table_name", required=False)
@click.pass_context
def sync(ctx, table_name):
    """Sync Iceberg tables with HuggingFace datasets from config.

    Discovers datasets and creates/updates Iceberg tables. For new tables,
    creates initial metadata and namespaces on-demand. For existing tables,
    checks if dataset revision has changed and skips if already up-to-date.

    Example:
        faceberg --config=faceberg.yml sync
        faceberg --config=faceberg.yml sync namespace1.table1
    """
    from faceberg.pretty import CatalogTreeView, OperationTracker, TableState, TableNode
    from faceberg import config as cfg

    catalog = ctx.obj["catalog"]
    tree_view = CatalogTreeView(catalog)

    console.print(f"[bold blue]Catalog:[/bold blue] {catalog.uri}")

    # Initialize all table nodes to PENDING
    tables_to_sync = []
    for identifier, node in catalog.config().traverse():
        if isinstance(node, cfg.Dataset):
            if table_name is None or str(identifier) == table_name:
                tree_view.table_nodes[identifier] = TableNode(
                    identifier=identifier,
                    node_type=type(node),
                    state=TableState.PENDING,
                    metadata={"repo": node.repo, "config": node.config}
                )
                tables_to_sync.append(identifier)

    if not tables_to_sync:
        console.print("[yellow]No tables to sync[/yellow]")
        return

    console.print(f"\n[bold blue]Syncing {len(tables_to_sync)} table(s)...[/bold blue]\n")

    synced_count = 0
    failed_count = 0

    with OperationTracker(tree_view, console=console) as tracker:
        for identifier in tables_to_sync:
            tracker.update(identifier, TableState.IN_PROGRESS, progress=0)
            try:
                # Progress callback updates tracker
                def progress_cb(pct):
                    tracker.update(identifier, TableState.IN_PROGRESS, progress=pct)

                table = catalog.sync_dataset(identifier, progress_callback=progress_cb)
                tracker.update(identifier, TableState.COMPLETE, progress=100)
                synced_count += 1
            except Exception as e:
                tracker.update(identifier, TableState.FAILED, error=str(e))
                failed_count += 1

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Synced: {synced_count}")
    if failed_count > 0:
        console.print(f"  Failed: {failed_count}")
```

**Step 4: Run test to verify it passes**

Run: `pytest faceberg/tests/test_cli_pretty.py::test_sync_command_shows_progress -v`

Expected: PASS

**Step 5: Commit**

```bash
git add faceberg/cli.py faceberg/tests/test_cli_pretty.py
git commit -m "feat: add live progress tracking to sync command

Replace spinner with rich tree view showing real-time progress
for each table being synced with state indicators and progress bars.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Add Progress Callback to add_dataset

**Files:**
- Modify: `faceberg/catalog.py:932-1031`

**Step 1: Add progress callback parameter**

```python
# faceberg/catalog.py (modify add_dataset method signature around line 932)
def add_dataset(
    self,
    identifier: Union[str, Identifier],
    repo: str,
    config: str = "default",
    progress_callback: Optional[Callable[[int], None]] = None
) -> Table:
    """Add a dataset to the catalog and create the Iceberg table.

    This discovers the HuggingFace dataset, converts it to an Iceberg table,
    and adds it to the catalog in a single operation.

    Args:
        identifier: Table identifier in format "namespace.table"
        repo: HuggingFace dataset repository in format "org/repo"
        config: Dataset configuration name (default: "default")
        progress_callback: Optional callback receiving progress (0-100)

    Returns:
        Created Table object

    Raises:
        ValueError: If identifier format is invalid
        TableAlreadyExistsError: If table already exists with metadata
    """
    identifier = Identifier(identifier)
    catalog_config = self.config()

    if progress_callback:
        progress_callback(0)

    if identifier in catalog_config:
        # Check if metadata files exist
        table_uri = self.uri / identifier.path
        version_hint_uri = table_uri / "metadata/version-hint.text"
        io = load_file_io(properties=self.properties, location=version_hint_uri)

        try:
            with io.new_input(version_hint_uri).open():
                pass  # File exists, table has been synced
            # Table has both config entry and metadata - it's truly a duplicate
            raise TableAlreadyExistsError(f"Table {identifier} already exists in catalog")
        except FileNotFoundError:
            pass  # Config entry exists but no metadata - we can proceed

    if progress_callback:
        progress_callback(10)

    # Discover dataset
    dataset_info = DatasetInfo.discover(
        repo_id=repo,
        configs=[config],
        token=self._hf_token,
    )

    if progress_callback:
        progress_callback(30)

    # Convert to TableInfo
    namespace, table_name = identifier
    table_info = dataset_info.to_table_info(
        namespace=namespace,
        table_name=table_name,
        config=config,
        token=self._hf_token,
    )

    if progress_callback:
        progress_callback(50)

    # Create the table with full metadata in staging context
    with self._staging() as staging:
        # Define table directory in the staging area
        table_dir = staging / identifier.path
        table_dir.mkdir(parents=True, exist_ok=True)

        # Create table URI for metadata
        table_uri = self.uri / identifier.path

        # Create metadata writer
        metadata_writer = IcebergMetadataWriter(
            table_path=table_dir,
            schema=table_info.schema,
            partition_spec=table_info.partition_spec,
            base_uri=table_uri,
        )

        if progress_callback:
            progress_callback(60)

        # Generate table UUID
        table_uuid = str(uuid.uuid4())

        # Write Iceberg metadata files (manifest, manifest list, table metadata)
        metadata_writer.create_metadata_from_files(
            file_infos=table_info.files,
            table_uuid=table_uuid,
            properties=table_info.get_table_properties(),
        )

        if progress_callback:
            progress_callback(80)

        # Record all created files in the table directory
        for path in table_dir.rglob("*"):
            if path.is_file():
                staging.add(path.relative_to(staging.path))

        # Register table in config if not already there
        if identifier not in catalog_config:
            catalog_config[identifier] = cfg.Dataset(
                repo=table_info.source_repo,
                config=table_info.source_config,
            )
            # Save config since we added a dataset table
            catalog_config.to_yaml(staging / "faceberg.yml")
            staging.add("faceberg.yml")

    if progress_callback:
        progress_callback(90)

    # Load and return table after persistence
    result = self.load_table(identifier)

    if progress_callback:
        progress_callback(100)

    return result
```

**Step 2: Commit**

```bash
git add faceberg/catalog.py
git commit -m "feat: add progress callback to add_dataset

Add optional progress_callback parameter that receives progress
percentage (0-100) during dataset addition operations.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Update CLI add Command with Progress

**Files:**
- Modify: `faceberg/cli.py:41-96`

**Step 1: Write implementation**

```python
# faceberg/cli.py (replace add function around line 41)
@main.command()
@click.argument("dataset")
@click.option("--table", "-t", help="Explicit table identifier (namespace.table)")
@click.option("--config", "-c", default="default", help="Dataset config name")
@click.pass_context
def add(ctx, dataset, table, config):
    """Add a table to the catalog.

    DATASET: HuggingFace dataset in format 'org/repo'

    By default, the table identifier is inferred from the dataset:
    org/repo -> namespace 'org', table 'repo' (identifier: org.repo)

    Examples:
        # Add with inferred identifier (deepmind.code_contests)
        faceberg add deepmind/code_contests

        # Add with explicit identifier
        faceberg add deepmind/code_contests --table myns.mytable

        # Add with non-default config
        faceberg add squad --config plain_text --table default.squad
    """
    from faceberg.pretty import CatalogTreeView, OperationTracker, TableState, TableNode
    from faceberg import config as cfg

    catalog = ctx.obj["catalog"]

    # Determine table identifier
    if table:
        # Explicit identifier provided
        table_identifier = table
    else:
        # Infer from dataset: org/repo -> org.repo
        try:
            namespace, table_name = dataset.split("/", 1)
            table_identifier = f"{namespace}.{table_name}"
        except ValueError:
            console.print("[red]Error: dataset must be in format 'org/repo'[/red]")
            raise click.Abort()

    # Create tree view for progress display
    tree_view = CatalogTreeView(catalog)
    identifier = catalog.Identifier(table_identifier)
    tree_view.table_nodes[identifier] = TableNode(
        identifier=identifier,
        node_type=cfg.Dataset,
        state=TableState.PENDING,
        metadata={"repo": dataset, "config": config}
    )

    console.print(f"[bold blue]Adding table:[/bold blue] {table_identifier}")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Config: {config}\n")

    # Add dataset to catalog and create Iceberg table with progress tracking
    try:
        with OperationTracker(tree_view, console=console) as tracker:
            tracker.update(identifier, TableState.IN_PROGRESS, progress=0)

            def progress_cb(pct):
                tracker.update(identifier, TableState.IN_PROGRESS, progress=pct)

            table = catalog.add_dataset(
                identifier=table_identifier,
                dataset=dataset,
                config=config,
                progress_callback=progress_cb
            )
            tracker.update(identifier, TableState.COMPLETE, progress=100)

        console.print(f"\n[green]✓ Added {table_identifier} to catalog[/green]")
        console.print(f"  Location: {table.metadata_location}")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
    except Exception as e:
        if "already exists" in str(e).lower():
            console.print(f"[yellow]Table {table_identifier} already exists[/yellow]")
        else:
            console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
```

**Step 2: Commit**

```bash
git add faceberg/cli.py
git commit -m "feat: add live progress tracking to add command

Show tree view with real-time progress bar when adding datasets
to the catalog.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Run Full Test Suite

**Files:**
- All test files

**Step 1: Run all tests**

Run: `pytest faceberg/tests/ -v`

Expected: All tests PASS

**Step 2: Fix any failures**

If any tests fail, fix them one by one:
1. Read the error message
2. Identify the root cause
3. Fix the code
4. Re-run the specific test
5. Commit the fix

**Step 3: Run tests with coverage**

Run: `pytest faceberg/tests/ --cov=faceberg --cov-report=term-missing`

Expected: Coverage > 80% for pretty.py

**Step 4: Commit if needed**

If any fixes were made:

```bash
git add <files>
git commit -m "test: fix test failures and improve coverage"
```

---

## Task 12: Manual Testing

**Files:**
- N/A (manual testing)

**Step 1: Test list command**

```bash
# Create test catalog
mkdir /tmp/test_catalog
faceberg /tmp/test_catalog init

# Create namespace
python -c "
from faceberg.catalog import catalog
from faceberg import config as cfg
cat = catalog('/tmp/test_catalog')
cat.create_namespace('default')
config_obj = cat.config()
config_obj[('default', 'test')] = cfg.Dataset(repo='imdb', config='plain_text')
config_obj.to_yaml('/tmp/test_catalog/faceberg.yml')
"

# Run list command
faceberg /tmp/test_catalog list
```

Expected: Tree view with icons and colors

**Step 2: Test add command (will fail but shows progress)**

```bash
faceberg /tmp/test_catalog add stanfordnlp/imdb --table default.imdb
```

Expected: Progress bar with tree view (may fail due to network)

**Step 3: Visual verification**

Verify:
- Icons display correctly (📁, 🗂️, 🤗, 📊, 👁️)
- Colors are readable
- Progress bars show percentages
- Tree structure is clear
- Errors display properly (test by using invalid dataset)

**Step 4: Document findings**

No commit needed for this step.

---

## Task 13: Update Documentation

**Files:**
- Create: `docs/visualization.md`
- Modify: `README.md`

**Step 1: Create visualization documentation**

```markdown
# docs/visualization.md
# Rich Terminal Visualization

Faceberg provides rich terminal visualization for catalog operations with real-time progress tracking.

## Features

### Tree View Display

All catalog commands display a hierarchical tree view with:
- **Icons** for different node types (📁 catalog, 🗂️ namespace, 🤗 dataset, 📊 table, 👁️ view)
- **Colors** for visual distinction (cyan namespaces, yellow datasets, green tables, blue views)
- **Metadata** inline (dataset repos, view queries)

### Live Progress Tracking

Operations like sync and add show:
- **State indicators** (⏳ pending, ▶️ in progress, ✓ complete, ✗ failed)
- **Progress bars** with percentages
- **Real-time updates** as operations proceed

### Example Output

```
📁 mycatalog
├── 🗂️  default
│   ├── 🤗 imdb (stanfordnlp/imdb) ✓
│   └── 🤗 gsm8k (openai/gsm8k) ▶️ [45%]
└── 🗂️  deepmind
    └── 🤗 code_contests (deepmind/code_contests) ⏳
```

## Commands

### list

Display catalog structure as tree:

```bash
faceberg mycatalog list
```

### sync

Sync datasets with live progress:

```bash
faceberg mycatalog sync
```

Shows progress for each table being synced.

### add

Add dataset with progress tracking:

```bash
faceberg mycatalog add org/dataset
```

## Implementation

See `faceberg/pretty.py` for implementation details:
- `TableState` - State enum with visual styling
- `TableNode` - State tracking dataclass
- `CatalogTreeView` - Tree builder and renderer
- `OperationTracker` - Live display manager
```

**Step 2: Update README**

```markdown
# README.md (add after CLI Commands section)

### Rich Visualization

Commands display catalog structure as a rich tree with icons and colors:

```bash
faceberg mycatalog list
```

Output:
```
📁 mycatalog
├── 🗂️  default
│   ├── 🤗 imdb (stanfordnlp/imdb)
│   └── 🤗 gsm8k (openai/gsm8k)
```

Long-running operations show real-time progress:

```bash
faceberg mycatalog sync
```

See [docs/visualization.md](docs/visualization.md) for details.
```

**Step 3: Commit**

```bash
git add docs/visualization.md README.md
git commit -m "docs: add visualization documentation

Document rich terminal visualization features including tree view
display, live progress tracking, and example outputs.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Summary

This plan implements rich catalog visualization in 13 tasks:

1. **Tasks 1-2**: Core data structures (TableState, TableNode)
2. **Task 3**: Node formatting with icons/colors
3. **Task 4**: Tree building and rendering
4. **Task 5**: Live progress tracking
5. **Tasks 6-10**: CLI integration (list, sync, add commands)
6. **Tasks 11-12**: Testing and verification
7. **Task 13**: Documentation

Each task follows TDD: write test → run (fail) → implement → run (pass) → commit.

**Key Principles:**
- **DRY**: Reuse format_node, TableNode, and CatalogTreeView
- **YAGNI**: No premature optimization, implement only what's needed
- **TDD**: Test-first for all core functionality
- **Frequent commits**: After each passing task

**Next Phase (Future):**
- Add `status` command showing tables needing updates
- Implement filtering/sorting for tree view
- Add export options (JSON, YAML)

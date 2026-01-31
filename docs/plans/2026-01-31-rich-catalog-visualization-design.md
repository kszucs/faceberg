# Rich Catalog Visualization Design

## Overview

Add rich terminal visualization to Faceberg's CLI for displaying catalog structure and tracking operation progress in real-time. This provides users with clear visual feedback during catalog operations like initialization, syncing, and adding datasets.

## Motivation

Currently, CLI commands provide minimal feedback during operations. Users need:
- Visual representation of catalog hierarchy (namespaces, tables, datasets, views)
- Real-time progress tracking for long-running operations
- Clear status indicators for table states (pending, syncing, complete, up-to-date, failed)
- Better feedback when operations run in parallel

## Architecture

### Component Structure

The visualization system consists of three main components in a new `faceberg/pretty.py` module:

1. **`TableState`** - Enum defining possible table states with visual styling
2. **`CatalogTreeView`** - Builds and renders hierarchical catalog structure
3. **`OperationTracker`** - Context manager for live progress updates

```
┌─────────────────────────────────────────────┐
│           CLI Commands (cli.py)             │
│  init, sync, add, list, status (future)     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│      Visualization Layer (pretty.py)        │
│  ┌─────────────────────────────────────┐   │
│  │     CatalogTreeView                 │   │
│  │  - build_tree()                     │   │
│  │  - update_state()                   │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │     OperationTracker                │   │
│  │  - Live display context manager     │   │
│  │  - Real-time refresh                │   │
│  └─────────────────────────────────────┘   │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│      Business Logic (catalog.py)            │
│  BaseCatalog, LocalCatalog, RemoteCatalog   │
│  - sync_dataset() with progress callback    │
│  - add_dataset() with progress callback     │
└─────────────────────────────────────────────┘
```

## Data Structures

### TableState Enum

```python
from enum import Enum

class TableState(Enum):
    PENDING = ("⏳", "dim white")        # Not started
    IN_PROGRESS = ("▶️", "yellow")       # Currently processing
    COMPLETE = ("✓", "green")            # Successfully finished
    UP_TO_DATE = ("✓", "dim green")      # Already synced, skipped
    FAILED = ("✗", "red")                # Error occurred
    NEEDS_UPDATE = ("↻", "blue")         # New version available (future)

    def __init__(self, icon: str, color: str):
        self.icon = icon
        self.color = color
```

### TableNode

```python
from dataclasses import dataclass, field
from typing import Optional, Type

@dataclass
class TableNode:
    """Represents a table with tracking state."""
    identifier: Identifier
    node_type: Type[cfg.Node]  # Table, Dataset, View
    state: TableState = TableState.PENDING
    progress: Optional[int] = None  # 0-100 percentage
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)  # repo, config, etc.
```

### Node Type Styling

```python
NODE_STYLES = {
    cfg.Namespace: ("🗂️", "cyan"),
    cfg.Table: ("📊", "green"),
    cfg.Dataset: ("🤗", "yellow"),
    cfg.View: ("👁️", "blue"),
}
```

## Core Classes

### CatalogTreeView

Main class for building and rendering catalog structure as a Rich Tree.

```python
class CatalogTreeView:
    def __init__(self, catalog: BaseCatalog):
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

    def update_state(self, identifier: Identifier, state: TableState,
                     progress: Optional[int] = None, error: Optional[str] = None):
        """Update a table's state.

        Args:
            identifier: Table identifier
            state: New state
            progress: Optional progress percentage (0-100)
            error: Optional error message
        """

    def format_node(self, identifier: Identifier, node: cfg.Node,
                    table_node: Optional[TableNode] = None) -> str:
        """Format a node with icon, color, and optional state/progress.

        Returns styled string for Rich rendering.
        """
```

### OperationTracker

Context manager for live progress tracking during operations.

```python
class OperationTracker:
    """Track and display live progress for catalog operations."""

    def __init__(self, tree_view: CatalogTreeView, console: Console = None):
        self.tree_view = tree_view
        self.console = console or Console()
        self.live = Live(
            tree_view.build_tree(),
            refresh_per_second=4,
            console=self.console
        )

    def __enter__(self):
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.live.stop()

    def update(self, identifier: Identifier, state: TableState,
               progress: Optional[int] = None, error: Optional[str] = None):
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

## Tree Rendering Algorithm

The tree is built recursively from the config structure:

1. Create root node with catalog name
2. Traverse config starting from root namespace
3. For each node:
   - Format with icon, color, and name
   - Add metadata (repo for Dataset, query snippet for View)
   - Add state indicator if node is tracked
   - Add progress bar if in progress
4. For namespaces, recurse into children
5. For errors, add error message as child node

```python
def build_tree(self) -> Tree:
    """Build Rich Tree recursively from config."""
    config = self.config
    root = Tree(f"[bold cyan]📁 {self.catalog.name}[/bold cyan]")

    def add_nodes(parent_tree: Tree, namespace: cfg.Namespace,
                  parent_id: Identifier = ()):
        """Recursively add nodes to tree."""
        for name, node in namespace.items():
            node_id = Identifier((*parent_id, name))
            table_node = self.table_nodes.get(node_id)

            label = self.format_node(node_id, node, table_node)

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
```

## CLI Integration

### Static Display (list command)

For commands that just display the catalog without operations:

```python
@main.command("list")
@click.pass_context
def list_tables(ctx):
    catalog = ctx.obj["catalog"]
    tree_view = CatalogTreeView(catalog)
    console.print(tree_view.build_tree())
```

### Live Progress (sync command)

For commands with operations:

```python
@main.command()
@click.pass_context
def sync(ctx, table_name):
    catalog = ctx.obj["catalog"]
    tree_view = CatalogTreeView(catalog)

    # Initialize all table nodes to PENDING
    for identifier, node in catalog.config().traverse():
        if isinstance(node, cfg.Dataset):
            tree_view.table_nodes[identifier] = TableNode(
                identifier=identifier,
                node_type=type(node),
                state=TableState.PENDING,
                metadata={"repo": node.repo, "config": node.config}
            )

    with OperationTracker(tree_view) as tracker:
        for identifier, node in catalog.config().traverse():
            if isinstance(node, cfg.Dataset):
                tracker.update(identifier, TableState.IN_PROGRESS, progress=0)
                try:
                    # Progress callback updates tracker
                    def progress_cb(pct):
                        tracker.update(identifier, TableState.IN_PROGRESS, progress=pct)

                    table = catalog.sync_dataset(identifier, progress_callback=progress_cb)
                    tracker.update(identifier, TableState.COMPLETE, progress=100)
                except Exception as e:
                    tracker.update(identifier, TableState.FAILED, error=str(e))
```

### Commands to Update

1. **`list`** - Replace current output with tree view
2. **`sync`** - Add live progress tracking with OperationTracker
3. **`init`** - Show progress as tables are created from config
4. **`add`** - Show single table addition with completion indicator

## Catalog Progress Callbacks

To enable progress reporting, catalog methods need optional callbacks:

```python
# In catalog.py
def sync_dataset(
    self,
    identifier: Union[str, Identifier],
    progress_callback: Optional[Callable[[int], None]] = None
) -> Table:
    """Sync dataset with optional progress reporting.

    Args:
        identifier: Table identifier
        progress_callback: Optional callback receiving progress (0-100)
    """
    # ... existing code ...

    if progress_callback:
        progress_callback(25)  # Dataset discovered

    # ... conversion work ...

    if progress_callback:
        progress_callback(75)  # Metadata written

    # ... finalization ...

    if progress_callback:
        progress_callback(100)  # Complete

    return table
```

## Implementation Phases

### Phase 1 - Core Infrastructure
- Create `faceberg/pretty.py` module
- Implement `TableState` enum
- Implement `TableNode` dataclass
- Implement `CatalogTreeView.build_tree()` with static rendering
- Unit tests for tree building

### Phase 2 - Live Progress Tracking
- Implement `OperationTracker` context manager
- Add `CatalogTreeView.update_state()` method
- Test live updates with mock operations

### Phase 3 - Catalog Integration
- Add optional `progress_callback` parameter to `sync_dataset()`
- Add progress reporting to `add_dataset()`
- Implement parallel operation tracking for multiple tables

### Phase 4 - CLI Integration
- Update `list` command to use `CatalogTreeView`
- Update `sync` command to use `OperationTracker`
- Update `init` command to show progress
- Update `add` command to show completion

### Phase 5 - Future Enhancements
- Add `status` command to show which tables need updates
- Implement needs-update detection logic
- Add filtering/sorting options for tree view

## Testing Strategy

### Unit Tests
- Tree building with various config structures (empty, nested, mixed node types)
- State transitions for TableNode
- Node formatting with different states and metadata
- Error display formatting

### Integration Tests
- Static tree rendering with real catalog
- Live updates with mock catalog operations
- Progress callback integration
- Parallel operation tracking

### Visual Tests
- Capture rendered output for regression testing
- Test with different terminal widths
- Test color output vs no-color mode

## Example Output

### Static Tree (list command)
```
📁 mycatalog
├── 🗂️  default
│   ├── 🤗 imdb (stanfordnlp/imdb)
│   └── 🤗 gsm8k (openai/gsm8k)
└── 🗂️  deepmind
    └── 🤗 code_contests (deepmind/code_contests)
```

### Live Progress (sync command)
```
📁 mycatalog
├── 🗂️  default
│   ├── 🤗 imdb (stanfordnlp/imdb) ✓
│   └── 🤗 gsm8k (openai/gsm8k) ▶️ [45%]
└── 🗂️  deepmind
    └── 🤗 code_contests (deepmind/code_contests) ⏳
```

### With Errors
```
📁 mycatalog
├── 🗂️  default
│   ├── 🤗 imdb (stanfordnlp/imdb) ✓
│   └── 🤗 gsm8k (openai/gsm8k) ✗
│       └── Error: Dataset not found: openai/gsm8k
└── 🗂️  deepmind
    └── 🤗 code_contests (deepmind/code_contests) ✓
```

## Dependencies

- `rich` - Already a dependency, provides Tree, Live, Console

## Benefits

1. **Better UX** - Clear visual feedback during operations
2. **Progress Visibility** - Users see what's happening in real-time
3. **Error Context** - Errors shown in hierarchy context
4. **Parallel Operations** - Track multiple operations simultaneously
5. **Extensibility** - Easy to add new states and commands

## Future Enhancements

1. **`faceberg status` command** - Show which tables need updates by comparing revisions
2. **Filtering** - Filter tree by namespace, state, or node type
3. **Statistics** - Show counts at namespace level (e.g., "default (3 tables)")
4. **Export** - Export tree to JSON or YAML format
5. **Interactive Mode** - Use Rich's live keyboard input for interactive exploration

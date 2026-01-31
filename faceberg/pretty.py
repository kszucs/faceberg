"""Rich terminal visualization for catalog operations."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import singledispatch
from typing import Literal, Optional

from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn
from rich.tree import Tree

from . import config as cfg

StateKind = Literal["pending", "in_progress", "complete", "up_to_date", "needs_update"]

_state_icons = {
    "pending": "⏳",
    "in_progress": "▶️",
    "complete": "✓",
    "up_to_date": "✓",
    "needs_update": "↻",
}

_state_colors = {
    "pending": "dim white",
    "in_progress": "yellow",
    "complete": "green",
    "up_to_date": "dim green",
    "needs_update": "blue",
}

# Stage descriptions for progress display
_stage_messages = {
    "pending": "Preparing to add dataset",
    "in_progress": "Processing dataset",
    "complete": "Completed successfully",
}


@dataclass
class TableState:
    """Represents the state of a table operation with visual styling."""

    kind: StateKind = "pending"
    progress: Optional[int] = None
    error: Optional[str] = None

    @property
    def icon(self) -> str:
        """Get the icon for the current state kind."""
        return _state_icons[self.kind]

    @property
    def color(self) -> str:
        """Get the color for the current state kind."""
        return _state_colors[self.kind]


def tree(config: cfg.Config, states: dict[tuple, TableState] = None):
    """Build catalog tree with state icons.

    Args:
        config: Config object with catalog structure
        states: Dictionary mapping paths to their TableState

    Returns:
        Tree with catalog structure and state indicators
    """
    states = states or {}
    tree = Tree("Catalog", hide_root=True)

    for name, namespace in config.items():
        node(namespace, (name,), tree, states)

    return tree


@singledispatch
def node(ns: cfg.Namespace, path: tuple, parent: Tree, states: dict[tuple, TableState]) -> Tree:
    """Format and build tree for Namespace nodes."""
    name = path[-1]
    metadata = " [dim](namespace)[/dim]"
    label = f"[cyan]{name}[/cyan]{metadata}"
    tree = parent.add(label)

    # Recursively add children
    for name, child in ns.items():
        node(child, path + (name,), tree, states)


@node.register(cfg.Table)
@node.register(cfg.Dataset)
@node.register(cfg.View)
def node_leaf(
    node: cfg.Table | cfg.Dataset | cfg.View,
    path: tuple,
    parent: Tree,
    states: dict[tuple, TableState],
) -> Tree:
    """Format and build tree for Table, Dataset, and View nodes."""
    name = path[-1]

    # Build label based on node type
    if isinstance(node, cfg.Table):
        metadata = " [dim](table)[/dim]"
        label = f"[green]{name}[/green]{metadata}"
    elif isinstance(node, cfg.Dataset):
        metadata = f" [dim](dataset: {node.repo})[/dim]"
        label = f"[yellow]{name}[/yellow]{metadata}"
    elif isinstance(node, cfg.View):
        query_snippet = node.query[:30] + "..." if len(node.query) > 30 else node.query
        metadata = f" [dim](view: {query_snippet})[/dim]"
        label = f"[blue]{name}[/blue]{metadata}"
    else:
        raise TypeError(f"Unsupported node type: {type(node)}")

    # Add state tracking if state exists
    state = states.get(path)
    if state:
        label += f" [{state.color}]{state.icon}[/{state.color}]"

    # Add to parent tree
    current = parent.add(label)

    # Add error message if state has error
    if state and state.error:
        current.add(f"[red]Error: {state.error}[/red]")


@contextmanager
def progress_tree(config, console):
    """Context manager for displaying catalog tree with state tracking.

    Shows a tree view with state icons for items. Updates in-place without clearing console.

    Args:
        config: Config object with catalog structure
        console: Rich Console instance for output
    """
    states = {}
    rendered = tree(config, states)
    live = Live(rendered, console=console)
    live.start()

    def updater(path, state, percent=None, stage=None, error=None):
        """Update state for a specific item and refresh tree.

        Args:
            path: Table identifier (tuple)
            state: Current state ('pending', 'in_progress', 'complete', 'up_to_date')
            percent: Progress percentage (unused, for API compatibility)
            stage: Optional stage description (unused, for API compatibility)
            error: Error message if any
        """
        states[path] = TableState(state, progress=percent, error=error)
        rendered = tree(config, states)
        live.update(rendered)

    try:
        yield updater
    finally:
        live.stop()


@contextmanager
def progress_bars(config, console, identifiers):
    """Context manager for displaying progress for multiple datasets.

    Shows a table with progress bars for all datasets being processed.

    Args:
        config: Config object with catalog structure
        console: Rich Console instance for output
        identifiers: Optional list of specific dataset identifiers to track
    """
    # Create progress display with columns
    prog = Progress(
        TextColumn("[bold cyan]{task.fields[identifier]}[/bold cyan]", table_column=None),
        BarColumn(complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.fields[stage]}"),
        console=console,
        transient=False,
    )

    tasks = {}
    for identifier in identifiers:
        tasks[identifier] = prog.add_task(
            "processing",
            identifier=".".join(identifier),
            stage="Pending",
            total=100,
        )

    def updater(path, state, percent=None, stage=None, error=None):
        """Update progress for a specific dataset.

        Args:
            path: Table identifier (tuple or string)
            state: Current state ('pending', 'in_progress', 'complete')
            percent: Progress percentage (0-100)
            stage: Optional stage description (e.g., "Discovering dataset", "Writing metadata")
            error: Error message if any
        """
        task_id = tasks[path]
        # Update progress bar
        prog.update(
            task_id,
            completed=percent or 0,
            stage=stage or _stage_messages.get(state, state),
        )
        # Mark as complete if finished
        if state in ("complete", "up_to_date"):
            prog.update(task_id, completed=100)
        # Show error if present
        if error:
            prog.console.print(f"[red]✗ {identifier}: {error}[/red]")

    prog.start()
    try:
        yield updater
    finally:
        prog.stop()

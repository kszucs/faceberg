"""Rich terminal visualization for catalog operations."""

from dataclasses import dataclass
from functools import singledispatch
from typing import Literal, Optional

from . import config as cfg

from rich.tree import Tree
from rich.live import Live

from contextlib import contextmanager


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
    states = states or {}
    tree = Tree("Catalog", hide_root=True)

    for name, namespace in config.items():
        node(namespace, (name,), tree, states)

    return tree


@singledispatch
def node(
    ns: cfg.Namespace,
    path: tuple,
    parent: Tree,
    states: dict[tuple, TableState]
) -> Tree:
    """Format and build tree for Namespace nodes."""
    name = path[-1]
    metadata = f" [dim](namespace)[/dim]"
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
    states: dict[tuple, TableState]
) -> Tree:
    """Format and build tree for Table, Dataset, and View nodes."""
    name = path[-1]

    # Build label based on node type
    if isinstance(node, cfg.Table):
        metadata = f" [dim](table)[/dim]"
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
        if state.kind == "in_progress" and state.progress is not None:
            label += f" [{state.progress}%]"

    # Add to parent tree
    current = parent.add(label)

    # Add error message if state has error
    if state and state.error:
        current.add(f"[red]Error: {state.error}[/red]")


@contextmanager
def tree_progress(config, console):
    states = {}

    rendered = tree(config, states)
    live = Live(rendered, refresh_per_second=4, console=console)
    live.start()

    def updater(path, state, percent, error=None):
        states[path] = TableState(state, progress=percent, error=error)
        rendered = tree(config, states)
        live.update(rendered)

    try:
        yield updater
    finally:
        live.stop()
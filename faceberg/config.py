"""Config module for Faceberg catalog configuration management."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union
from pyiceberg.exceptions import (
    NamespaceAlreadyExistsError,
    NamespaceNotEmptyError,
    NoSuchNamespaceError,
    NoSuchTableError,
    TableAlreadyExistsError,
)
from collections.abc import Mapping
import yaml

class Identifier(tuple[str, ...]):

    def __new__(cls, value):
        """Create a new Identifier instance.

        Args:
            value: String (dot-separated) or list/tuple of strings

        Returns:
            Identifier instance
        """
        if isinstance(value, str):
            parts = tuple(value.split("."))
        elif isinstance(value, (list, tuple)):
            parts = tuple(value)
        else:
            raise TypeError("Identifier must be created from str, list, or tuple")
        return super().__new__(cls, parts)

    @property
    def path(self) -> Path:
        """Get the path representation of the identifier."""
        return Path(*self)


# =============================================================================
# Base Node Class
# =============================================================================


class Node:
    """Base class for all leaf node types.

    Provides common serialization/deserialization interface.
    Subclasses should implement from_dict() as a classmethod.
    """

    def to_dict(self) -> dict:
        """Convert node to dictionary using dataclasses.asdict().

        Subclasses can override to add additional fields (e.g., type discriminator).
        """
        if isinstance(self, Namespace):
            return {k: v.to_dict() for k, v in self.items()}
        elif isinstance(self, Table):
            return {"type": "table", **asdict(self)}
        elif isinstance(self, Dataset):
            return {"type": "dataset", **asdict(self)}
        elif isinstance(self, View):
            return {"type": "view", **asdict(self)}
        else:
            raise TypeError(f"Unsupported node type for serialization: {type(self)}")

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        """Create node from dictionary.

        Subclasses must implement this to construct instances from dicts.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict to deserialize {cls.__name__}, got {type(data)}")

        if "type" in data:
            typ = data.pop("type")
        else:
            typ = "namespace"

        if typ == "table":
            return Table(**data)
        elif typ == "dataset":
            return Dataset(**data)
        elif typ == "view":
            return View(**data)
        elif typ == "namespace":
            return Namespace({k: cls.from_dict(v) for k, v in data.items()})
        else:
            raise ValueError(f"Unknown node type: {typ}")


# =============================================================================
# Leaf Node Classes
# =============================================================================


@dataclass
class Table(Node):
    """Physical Iceberg table."""


@dataclass
class Dataset(Node):
    """External HuggingFace dataset reference."""
    repo: str
    config: str = "default"


@dataclass
class View(Node):
    """Logical view with SQL query."""
    query: str



# =============================================================================
# Namespace Class
# =============================================================================


class Namespace(Node, dict):

    def __repr__(self):
        return f"Namespace({super().__repr__()})"


# =============================================================================
# Config Class
# =============================================================================


class Config:
    """Root catalog configuration with tuple identifier support.

    Config extends dict to support both:
    - String keys for top-level namespaces: cfg["analytics"]
    - Tuple identifiers for nested paths: cfg[("analytics", "sales", "orders")]

    Tuple identifiers automatically create intermediate Namespace objects
    as needed, making it easy to build hierarchical structures.
    """
    data: dict[str, Namespace]

    def __init__(self, data: dict[str, Namespace] = None):
        self.data = data or {}

    def __getitem__(self, key: Union[str, tuple]) -> Any:
        """Get item by string key or tuple identifier.

        Args:
            key: String key or tuple identifier

        Returns:
            Value at the specified location

        Raises:
            KeyError: If key doesn't exist
        """
        path = Identifier(key)
        data = self.data
        for part in path:
            data = data[part]
        return data

    def __setitem__(self, key: Union[str, tuple], value: Any):
        """Set item by string key or tuple identifier.

        Args:
            key: String key or tuple identifier
            value: Value to set (Namespace or leaf)
        """
        path = Identifier(key)
        path, last = path[:-1], path[-1]
        data = self.data
        for part in path:
            if part not in data:
                data[part] = Namespace()
            data = data[part]
        data[last] = value

    def __delitem__(self, key: Union[str, tuple]):
        """Delete item by string key or tuple identifier.

        Args:
            key: String key or tuple identifier

        Raises:
            KeyError: If key doesn't exist
        """
        path = Identifier(key)
        path, last = path[:-1], path[-1]
        data = self.data
        for part in path:
            data = data[part]
        del data[last]

    def __contains__(self, key: Union[str, tuple]) -> bool:
        """Check if key exists.

        Args:
            key: String key or tuple identifier

        Returns:
            True if key exists
        """
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __repr__(self):
        return f"Config(data={self.data!r})"

    def traverse(self):
        """Generator to traverse all nodes in the config."""
        def traverse(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield from traverse(v, path + (k,))
            else:
                yield path, node

        yield from traverse(self.data, ())

    # =========================================================================
    # YAML Serialization
    # =========================================================================

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        _uri = data.pop("uri", None)
        node = Node.from_dict(data)
        return cls(node)

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.data.items()}

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Write config to YAML file.

        Args:
            path: Path to write YAML file to
        """
        data = self.to_dict()
        yaml_str = yaml.safe_dump(data, sort_keys=False)
        Path(path).write_text(yaml_str)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load config from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Config instance
        """
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        return cls.from_dict(data or {})



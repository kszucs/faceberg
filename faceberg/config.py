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
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        elif isinstance(key, tuple):
            path, last = key[:-1], key[-1]
            node = self
            for part in path:
                node = node[part]
            return node[last]
        else:
            raise TypeError("Key must be a tuple or string")

    def __setitem__(self, key, value):
        if isinstance(key, str):
            super().__setitem__(key, value)
        elif isinstance(key, tuple):
            path, last = key[:-1], key[-1]
            node = self
            for part in path:
                if part not in node:
                    node[part] = Namespace()
                node = node[part]
            node[last] = value
        else:
            raise TypeError("Key must be a tuple or string")

    def __delitem__(self, key):
        if isinstance(key, tuple):
            path, last = key[:-1], key[-1]
            self[path].__delitem__(last)
        elif isinstance(key, str):
            super().__delitem__(key)
        else:
            raise TypeError("Key must be a tuple or string")

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

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


class Config(Namespace):

    @classmethod
    def from_dict(cls, data: dict):
        return cls(Namespace.from_dict(data))

"""Config module for Faceberg catalog configuration management."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import yaml
from pyiceberg.typedef import Identifier


@dataclass
class Entry:
    """Table configuration entry."""

    dataset: str
    config: str = "default"


# Register Entry with YAML for automatic serialization/deserialization
def _entry_representer(dumper: yaml.SafeDumper, entry: Entry) -> yaml.Node:
    """Represent Entry as a YAML mapping."""
    return dumper.represent_dict(asdict(entry))


yaml.add_representer(Entry, _entry_representer, Dumper=yaml.SafeDumper)


class Config(Mapping):
    """Config store - nested dictionary with Mapping protocol for Identifier keys.

    Stores table configurations indexed by pyiceberg Identifiers.
    Supports arbitrary-depth identifiers.

    YAML format:
    uri: file:///path/to/catalog
    namespace:
      table:
        dataset: org/repo
        config: default

    Internal structure (nested dict):
    {
        'uri': 'file:///path/to/catalog',
        'data': {
            'namespace': {
                'table': Entry(dataset='org/repo', config='default')
            }
        }
    }

    Leaf nodes are Entry dataclass instances.
    """

    def __init__(self, uri: str, data: dict | None = None):
        """Initialize Config with URI and optional nested data.

        Args:
            uri: Catalog URI
            data: Optional nested dict with arbitrary depth
        """
        self.uri = uri
        self._data: dict = data or {}

    def __getitem__(self, key: Identifier) -> Entry:
        """Get table configuration by Identifier.

        Args:
            key: Identifier (tuple of strings)

        Returns:
            Table configuration Entry

        Raises:
            KeyError: If table doesn't exist
        """
        current = self._data
        for part in key:
            current = current[part]
        return current

    def __setitem__(self, key: Identifier, value: Entry) -> None:
        """Set table configuration by Identifier.

        Args:
            key: Identifier (tuple of strings)
            value: Table configuration Entry

        Raises:
            ValueError: If value is not an Entry
        """
        # Validate value type
        if not isinstance(value, Entry):
            raise ValueError(f"Value must be an Entry, got {type(value).__name__}")

        # Navigate to parent, creating intermediate dicts as needed
        current = self._data
        for part in key[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the leaf value
        current[key[-1]] = value

    def __delitem__(self, key: Identifier) -> None:
        """Delete table configuration by Identifier.

        Args:
            key: Identifier (tuple of strings)

        Raises:
            KeyError: If table doesn't exist
        """
        # Navigate to parent
        current = self._data
        for part in key[:-1]:
            current = current[part]

        # Delete the leaf
        del current[key[-1]]

    def __len__(self) -> int:
        """Return number of tables in config."""
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over Identifier tuples."""

        def traverse(d: dict, path: tuple) -> Iterator[tuple]:
            for key, value in d.items():
                current_path = path + (key,)
                if isinstance(value, Entry):
                    # This is a leaf Entry
                    yield current_path
                elif isinstance(value, dict):
                    # This is an intermediate navigation dict
                    yield from traverse(value, current_path)

        yield from traverse(self._data, ())

    def __contains__(self, key: object) -> bool:
        """Check if Identifier exists in config."""
        if not isinstance(key, (tuple, list)):
            return False

        try:
            current = self._data
            for part in key:
                current = current[part]
            return isinstance(current, Entry)
        except (KeyError, TypeError):
            return False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from YAML file.

        Args:
            path: Path to faceberg.yml file

        Returns:
            Parsed Config

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("Config file is empty")

        # Extract URI
        uri = data.pop("uri", None)
        if not uri:
            raise ValueError("Missing required 'uri' field in config")

        # Convert leaf dicts to Entry objects
        def convert_to_entries(d: dict) -> dict:
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    if "dataset" in value:
                        # This is a leaf config dict, convert to Entry
                        result[key] = Entry(
                            dataset=value["dataset"],
                            config=value.get("config", "default"),
                        )
                    else:
                        # This is an intermediate dict, recurse
                        result[key] = convert_to_entries(value)
                else:
                    result[key] = value
            return result

        return cls(uri=uri, data=convert_to_entries(data))

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file.

        Args:
            path: Path to save faceberg.yml file
        """
        path = Path(path)

        with open(path, "w") as f:
            yaml.safe_dump(
                {"uri": self.uri, **self._data},
                f,
                default_flow_style=False,
                sort_keys=False,
            )

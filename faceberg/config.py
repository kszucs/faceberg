"""Config module for Faceberg catalog configuration management."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml


class Identifier(tuple):
    """A tuple of exactly 2 strings: (namespace, table)."""

    def __new__(cls, value):
        if isinstance(value, str):
            parts = tuple(value.split("."))
        else:
            parts = tuple(value)
        return super().__new__(cls, parts)

    @property
    def path(self) -> Path:
        return Path(*self)

    def is_namespace(self) -> bool:
        return len(self) == 1

    def is_table(self) -> bool:
        return len(self) == 2


class Namespace(dict):
    """Namespace containing tables."""


@dataclass
class Table:
    """Table configuration entry."""

    dataset: Optional[str] = None
    config: Optional[str] = None


@dataclass
class Config:
    """Config store for table configurations."""

    uri: str
    data: dict[str, Namespace] = field(default_factory=dict)

    def __getitem__(self, key: Identifier) -> Table | Namespace:
        key = Identifier(key)
        if key.is_namespace():
            (namespace,) = key
            return self.data[namespace]
        elif key.is_table():
            namespace, table = key
            return self.data[namespace][table]
        else:
            raise KeyError("Only 2-level identifiers (namespace, table) are supported")

    def __setitem__(self, key: Identifier, value) -> None:
        key = Identifier(key)
        if key.is_namespace():
            (namespace,) = key
            self.data[namespace] = Namespace(value)
        elif key.is_table():
            namespace, table = key
            if namespace not in self.data:
                self.data[namespace] = Namespace()
            self.data[namespace][table] = value
        else:
            raise KeyError("Only 2-level identifiers (namespace, table) are supported")

    def __delitem__(self, key: Identifier) -> None:
        key = Identifier(key)
        if key.is_namespace():
            (namespace,) = key
            del self.data[namespace]
        elif key.is_table():
            namespace, table = key
            del self.data[namespace][table]
        else:
            raise KeyError("Only 2-level identifiers (namespace, table) are supported")

    def __contains__(self, key: Identifier) -> bool:
        key = Identifier(key)
        try:
            self[key]
            return True
        except KeyError:
            return False

    @property
    def namespaces(self) -> list[Identifier]:
        return [Identifier((ns,)) for ns in self.data.keys()]

    @property
    def tables(self) -> list[Identifier]:
        result = []
        for namespace, tables in self.data.items():
            for table in tables.keys():
                result.append(Identifier((namespace, table)))
        return result

    def to_yaml(self, path) -> None:
        """Write config to YAML file.

        Args:
            path: Path to write YAML file to
        """
        # Convert Table objects to dicts
        data_dict = {
            ns: {name: asdict(table) for name, table in tables.items()}
            for ns, tables in self.data.items()
        }
        output = {"uri": self.uri, **data_dict}
        yaml_str = yaml.safe_dump(output, sort_keys=False)
        Path(path).write_text(yaml_str)

    @classmethod
    def from_yaml(cls, path) -> "Config":
        """Load config from YAML file."""
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        if not data:
            raise ValueError("Config is empty")

        uri = data.pop("uri", None)
        if not uri:
            raise ValueError("Missing required 'uri' field in config")

        # Convert dicts to Namespace/Table (2-level hierarchy only)
        result = {}
        for namespace, tables in data.items():
            if not isinstance(tables, dict):
                raise ValueError(f"Namespace '{namespace}' must contain a dict of tables")

            namespace_obj = Namespace()
            for table_name, table_data in tables.items():
                if not isinstance(table_data, dict):
                    raise ValueError(f"Table '{namespace}.{table_name}' must be a dict")
                if "dataset" not in table_data:
                    raise ValueError(f"Table '{namespace}.{table_name}' must have 'dataset' field")

                namespace_obj[table_name] = Table(**table_data)

            result[namespace] = namespace_obj

        return cls(uri=uri, data=result)

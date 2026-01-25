"""Store module for Faceberg catalog state management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class Table:
    """Table entry in catalog store."""

    dataset: str
    uri: str
    config: str = "default"


@dataclass
class Namespace:
    """Namespace containing tables."""

    tables: Dict[str, Table] = field(default_factory=dict)


@dataclass
class Catalog:
    """Catalog store - single source of truth for catalog state."""

    uri: str
    namespaces: Dict[str, Namespace] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Catalog":
        """Load catalog from YAML file.

        Args:
            path: Path to faceberg.yml file

        Returns:
            Parsed Catalog

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

        # Extract URI if present
        uri = data.pop("uri", None)
        if not uri:
            raise ValueError("Missing required 'uri' field in config")

        # Parse namespaces config
        namespaces = {}
        for namespace_name, namespace_tables in data.items():
            # Validate namespace name
            if not namespace_name:
                raise ValueError("Namespace name cannot be empty")

            # Check for reserved names
            if namespace_name == "catalog":
                raise ValueError("Cannot use 'catalog' as namespace name (reserved)")

            # Validate namespace name format (alphanumeric, underscore, hyphen)
            import re

            if not re.match(r"^[a-zA-Z0-9_-]+$", namespace_name):
                raise ValueError(
                    f"Invalid namespace name '{namespace_name}'. "
                    "Must contain only alphanumeric characters, underscores, or hyphens"
                )

            if not isinstance(namespace_tables, dict):
                raise ValueError(f"Namespace '{namespace_name}' must be a dict of tables")

            # Parse tables in this namespace
            tables = {}
            for table_name, table_data in namespace_tables.items():
                if not isinstance(table_data, dict):
                    raise ValueError(
                        f"Table '{namespace_name}.{table_name}' must be a dict with 'dataset' field"
                    )

                if "dataset" not in table_data:
                    raise ValueError(f"Missing 'dataset' in {namespace_name}.{table_name}")

                tables[table_name] = Table(
                    dataset=table_data["dataset"],
                    uri=table_data.get("uri", ""),  # Empty string if not synced yet
                    config=table_data.get("config", "default"),
                )

            # Allow empty namespaces for newly created namespaces
            namespaces[namespace_name] = Namespace(tables=tables)

        # Allow empty namespaces for new catalogs
        return cls(uri=uri, namespaces=namespaces)

    def to_yaml(self, path: str | Path) -> None:
        """Save catalog to YAML file.

        Args:
            path: Path to save faceberg.yml file
        """
        path = Path(path)

        data = {"uri": self.uri}

        # Add each namespace as a top-level key
        for namespace_name, namespace in self.namespaces.items():
            namespace_data = {}
            for table_name, table in namespace.tables.items():
                table_data = {
                    "dataset": table.dataset,
                    "config": table.config,
                }
                # Include uri only if set
                if table.uri:
                    table_data["uri"] = table.uri
                namespace_data[table_name] = table_data

            data[namespace_name] = namespace_data

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

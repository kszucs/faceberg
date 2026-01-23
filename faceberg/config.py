"""Configuration file parsing for Faceberg."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class TableConfig:
    """Table configuration within a namespace."""
    name: str
    dataset: str
    config: str = "default"


@dataclass
class NamespaceConfig:
    """Namespace configuration."""
    name: str
    tables: List[TableConfig]


@dataclass
class CatalogConfig:
    """Catalog configuration."""
    name: str
    location: str
    namespaces: List[NamespaceConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CatalogConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to faceberg.yml file

        Returns:
            Parsed CatalogConfig

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

        # Parse catalog config
        if "catalog" not in data:
            raise ValueError("Missing 'catalog' section in config")

        catalog_data = data["catalog"]
        if "name" not in catalog_data:
            raise ValueError("Missing 'name' in catalog config")
        if "location" not in catalog_data:
            raise ValueError("Missing 'location' in catalog config")

        catalog_name = catalog_data["name"]
        catalog_location = catalog_data["location"]

        # Parse namespaces config
        namespaces = []
        for key, value in data.items():
            if key == "catalog":
                continue  # Skip catalog section

            # Each remaining top-level key is a namespace
            namespace_name = key

            # Validate namespace name
            if not namespace_name:
                raise ValueError("Namespace name cannot be empty")

            # Check for reserved names
            if namespace_name == "catalog":
                raise ValueError("Cannot use 'catalog' as namespace name (reserved)")

            # Validate namespace name format (alphanumeric, underscore, hyphen)
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', namespace_name):
                raise ValueError(
                    f"Invalid namespace name '{namespace_name}'. "
                    "Must contain only alphanumeric characters, underscores, or hyphens"
                )

            if not isinstance(value, dict):
                raise ValueError(f"Namespace '{namespace_name}' must be a dict of tables")

            # Parse tables in this namespace
            tables = []
            for table_name, table_data in value.items():
                if not isinstance(table_data, dict):
                    raise ValueError(
                        f"Table '{namespace_name}.{table_name}' must be a dict with 'dataset' field"
                    )

                if "dataset" not in table_data:
                    raise ValueError(f"Missing 'dataset' in {namespace_name}.{table_name}")

                tables.append(
                    TableConfig(
                        name=table_name,
                        dataset=table_data["dataset"],
                        config=table_data.get("config", "default"),
                    )
                )

            if not tables:
                raise ValueError(f"Namespace '{namespace_name}' has no tables defined")

            namespaces.append(
                NamespaceConfig(
                    name=namespace_name,
                    tables=tables,
                )
            )

        if not namespaces:
            raise ValueError("No namespaces defined in config")

        return cls(name=catalog_name, location=catalog_location, namespaces=namespaces)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save faceberg.yml file
        """
        path = Path(path)

        data = {
            "catalog": {
                "name": self.name,
                "location": self.location,
            }
        }

        # Add each namespace as a top-level key
        for namespace in self.namespaces:
            data[namespace.name] = {
                table.name: {
                    "dataset": table.dataset,
                    "config": table.config,
                }
                for table in namespace.tables
            }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

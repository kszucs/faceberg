"""Configuration file parsing for Faceberg."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

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
    """Catalog configuration - defines which datasets to sync as tables."""

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

        # Parse namespaces config
        namespaces = []
        for key, value in data.items():
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

            if not re.match(r"^[a-zA-Z0-9_-]+$", namespace_name):
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

        return cls(namespaces=namespaces)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save faceberg.yml file
        """
        path = Path(path)

        data = {}

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

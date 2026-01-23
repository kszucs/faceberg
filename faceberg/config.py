"""Configuration file parsing for Faceberg."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class CatalogConfig:
    """Catalog configuration."""
    name: str
    location: str


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    repo: str
    configs: Optional[List[str]] = None  # If None, use all configs from dataset


@dataclass
class FacebergConfig:
    """Top-level Faceberg configuration."""
    catalog: CatalogConfig
    datasets: List[DatasetConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FacebergConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to faceberg.yml file

        Returns:
            Parsed FacebergConfig

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

        catalog = CatalogConfig(
            name=catalog_data["name"],
            location=catalog_data["location"],
        )

        # Parse datasets config
        if "datasets" not in data:
            raise ValueError("Missing 'datasets' section in config")

        if not isinstance(data["datasets"], list):
            raise ValueError("'datasets' must be a list")

        if not data["datasets"]:
            raise ValueError("'datasets' list is empty")

        datasets = []
        for i, ds_data in enumerate(data["datasets"]):
            if "name" not in ds_data:
                raise ValueError(f"Missing 'name' in dataset {i}")
            if "repo" not in ds_data:
                raise ValueError(f"Missing 'repo' in dataset {i}")

            datasets.append(
                DatasetConfig(
                    name=ds_data["name"],
                    repo=ds_data["repo"],
                    configs=ds_data.get("configs"),
                )
            )

        return cls(catalog=catalog, datasets=datasets)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save faceberg.yml file
        """
        path = Path(path)

        data = {
            "catalog": {
                "name": self.catalog.name,
                "location": self.catalog.location,
            },
            "datasets": [
                {
                    "name": ds.name,
                    "repo": ds.repo,
                    **({"configs": ds.configs} if ds.configs else {}),
                }
                for ds in self.datasets
            ],
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

"""Dataset discovery using HuggingFace datasets library."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset_builder
from huggingface_hub import HfFileSystem


@dataclass
class DatasetInfo:
    """Information about a HuggingFace dataset."""
    repo_id: str
    configs: List[str]
    splits: Dict[str, List[str]]  # config -> list of splits
    parquet_files: Dict[str, Dict[str, List[str]]]  # config -> split -> list of files


class DatasetDiscovery:
    """Discover Parquet files and metadata in HuggingFace datasets using the datasets library."""

    def __init__(self, token: Optional[str] = None):
        """Initialize dataset discovery.

        Args:
            token: HuggingFace API token (uses HF_TOKEN env var if not provided)
        """
        self.token = token
        self.fs = HfFileSystem(token=token)

    def discover_dataset(
        self,
        repo_id: str,
        configs: Optional[List[str]] = None
    ) -> DatasetInfo:
        """Discover Parquet files and structure in a HuggingFace dataset.

        Uses the datasets library to get proper metadata about configs and splits.

        Args:
            repo_id: HuggingFace dataset repository ID (e.g., "kszucs/dataset1")
            configs: List of configs to discover (if None, discovers all)

        Returns:
            DatasetInfo with discovered structure

        Raises:
            ValueError: If dataset not found or has errors
        """
        # Get all available configs
        try:
            all_configs = get_dataset_config_names(repo_id, token=self.token)
        except Exception as e:
            # If get_dataset_config_names fails, try with just the default config
            try:
                # Try loading a builder to see if dataset exists
                load_dataset_builder(repo_id, token=self.token)
                all_configs = ["default"]
            except Exception:
                raise ValueError(f"Dataset {repo_id} not found or not accessible: {e}")

        # Filter to requested configs
        if configs:
            missing_configs = set(configs) - set(all_configs)
            if missing_configs:
                raise ValueError(
                    f"Configs not found in dataset: {missing_configs}. "
                    f"Available: {all_configs}"
                )
            discovered_configs = [c for c in all_configs if c in configs]
        else:
            discovered_configs = all_configs

        # Discover splits and files for each config
        splits_dict = {}
        parquet_files = {}

        for config_name in discovered_configs:
            # Get splits for this config
            try:
                config_splits = get_dataset_split_names(
                    repo_id,
                    config_name=config_name,
                    token=self.token
                )
            except Exception:
                # If no splits metadata, try to discover files manually
                config_splits = self._discover_splits_from_files(repo_id, config_name)

            splits_dict[config_name] = config_splits

            # Get Parquet files for each split
            parquet_files[config_name] = {}
            for split in config_splits:
                split_files = self._get_parquet_files_for_split(
                    repo_id, config_name, split
                )
                parquet_files[config_name][split] = split_files

        if not parquet_files or all(not config_files for config_files in parquet_files.values()):
            raise ValueError(f"No Parquet files found in dataset {repo_id}")

        return DatasetInfo(
            repo_id=repo_id,
            configs=discovered_configs,
            splits=splits_dict,
            parquet_files=parquet_files,
        )

    def _discover_splits_from_files(self, repo_id: str, config_name: str) -> List[str]:
        """Discover splits by looking at file structure when metadata is unavailable.

        Args:
            repo_id: Dataset repository ID
            config_name: Configuration name

        Returns:
            List of split names
        """
        splits = set()
        split_names = {"train", "test", "validation", "val", "dev"}

        # Look for common split patterns in file paths
        try:
            all_files = self.fs.ls(f"datasets/{repo_id}", detail=False, recursive=True)
            parquet_files = [f for f in all_files if f.endswith(".parquet")]

            for file_path in parquet_files:
                # Check if any split name appears in the path
                for split_name in split_names:
                    if f"/{split_name}/" in file_path or f"-{split_name}" in file_path:
                        splits.add(split_name)
                        break
                else:
                    # If no split found, assume "default"
                    splits.add("default")

        except Exception:
            splits.add("default")

        return list(splits) if splits else ["default"]

    def _get_parquet_files_for_split(
        self, repo_id: str, config_name: str, split: str
    ) -> List[str]:
        """Get Parquet file paths for a specific config and split.

        Args:
            repo_id: Dataset repository ID
            config_name: Configuration name
            split: Split name

        Returns:
            List of relative file paths (not hf:// URIs)
        """
        parquet_files = []

        # Common patterns for Parquet files
        patterns = [
            f"data/{split}/*.parquet",
            f"data/{config_name}/{split}/*.parquet",
            f"{config_name}/{split}/*.parquet",
            f"{split}/*.parquet",
            f"data/{split}-*.parquet",
            f"{split}-*.parquet",
            f"*-{split}-*.parquet",
        ]

        # If config is default, also try without config prefix
        if config_name == "default":
            patterns.extend([
                f"data/*.parquet",
                f"*.parquet",
            ])

        for pattern in patterns:
            try:
                files = self.fs.glob(f"datasets/{repo_id}/{pattern}")
                if files:
                    # Convert to relative paths
                    rel_files = [
                        f.replace(f"datasets/{repo_id}/", "")
                        for f in files
                    ]
                    parquet_files.extend(rel_files)
            except Exception:
                continue

        # Filter to ensure we only get files matching the split
        if parquet_files and split != "default":
            # Only keep files that have the split name in them
            parquet_files = [
                f for f in parquet_files
                if split in f.lower()
            ]

        # Remove duplicates
        parquet_files = list(set(parquet_files))

        return sorted(parquet_files)

    def get_parquet_files_for_table(
        self,
        dataset_info: DatasetInfo,
        config: str,
    ) -> List[str]:
        """Get all Parquet files for a specific config across all splits.

        Args:
            dataset_info: Dataset information
            config: Configuration name

        Returns:
            List of hf:// URIs for all Parquet files in this config
        """
        if config not in dataset_info.parquet_files:
            raise ValueError(f"Config {config} not found in dataset")

        files = []
        for split, split_files in dataset_info.parquet_files[config].items():
            for file_path in split_files:
                # Convert to hf:// URI
                hf_uri = f"hf://datasets/{dataset_info.repo_id}/{file_path}"
                files.append(hf_uri)

        return files

    def get_sample_parquet_file(
        self,
        dataset_info: DatasetInfo,
        config: str,
    ) -> str:
        """Get a sample Parquet file for schema inference.

        Args:
            dataset_info: Dataset information
            config: Configuration name

        Returns:
            hf:// URI to a sample Parquet file
        """
        files = self.get_parquet_files_for_table(dataset_info, config)
        if not files:
            raise ValueError(f"No Parquet files found for config {config}")
        return files[0]

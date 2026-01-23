"""Dataset discovery using HuggingFace datasets library."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset_builder


@dataclass
class DatasetInfo:
    """Information about a HuggingFace dataset."""

    repo_id: str
    configs: List[str]
    splits: Dict[str, List[str]]  # config -> list of splits
    parquet_files: Dict[str, Dict[str, List[str]]]  # config -> split -> list of files

    @classmethod
    def discover(
        cls,
        repo_id: str,
        configs: Optional[List[str]] = None,
        token: Optional[str] = None,
    ) -> "DatasetInfo":
        """Discover Parquet files and structure in a HuggingFace dataset.

        Uses the datasets library to get proper metadata about configs and splits.

        Args:
            repo_id: HuggingFace dataset repository ID (e.g., "kszucs/dataset1")
            configs: List of configs to discover (if None, discovers all)
            token: HuggingFace API token (uses HF_TOKEN env var if not provided)

        Returns:
            DatasetInfo with discovered structure

        Raises:
            ValueError: If dataset not found or has errors
        """
        # Get all available configs
        try:
            all_configs = get_dataset_config_names(repo_id, token=token)
        except Exception as e:
            # If get_dataset_config_names fails, try with just the default config
            try:
                load_dataset_builder(repo_id, token=token)
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
            config_splits, config_files = cls._discover_config(repo_id, config_name, token)
            splits_dict[config_name] = config_splits
            parquet_files[config_name] = config_files

        if not parquet_files or all(not files for files in parquet_files.values()):
            raise ValueError(f"No Parquet files found in dataset {repo_id}")

        return cls(
            repo_id=repo_id,
            configs=discovered_configs,
            splits=splits_dict,
            parquet_files=parquet_files,
        )

    @staticmethod
    def _discover_config(
        repo_id: str, config_name: str, token: Optional[str]
    ) -> tuple[List[str], Dict[str, List[str]]]:
        """Discover splits and files for a specific config.

        Args:
            repo_id: Dataset repository ID
            config_name: Configuration name
            token: HuggingFace API token

        Returns:
            Tuple of (splits list, files dict mapping split -> file paths)
        """
        try:
            builder = load_dataset_builder(repo_id, name=config_name, token=token)
            data_files = (
                builder.config.data_files if hasattr(builder.config, "data_files") else None
            )

            if data_files and isinstance(data_files, dict):
                # data_files is a dict: {split: [file_paths]}
                splits = list(data_files.keys())
                files = {
                    split: [DatasetInfo._extract_relative_path(repo_id, path) for path in paths]
                    for split, paths in data_files.items()
                }
                return splits, files

            # Fallback: use get_dataset_split_names
            splits = get_dataset_split_names(repo_id, config_name=config_name, token=token)
            return splits, {split: [] for split in splits}

        except Exception:
            # If builder fails, try get_dataset_split_names as fallback
            try:
                splits = get_dataset_split_names(repo_id, config_name=config_name, token=token)
            except Exception:
                splits = ["train"]  # Default fallback

            return splits, {split: [] for split in splits}

    @staticmethod
    def _extract_relative_path(repo_id: str, file_path: str) -> str:
        """Extract relative path from various file path formats.

        Args:
            repo_id: Dataset repository ID
            file_path: File path in various formats

        Returns:
            Relative path suitable for hf:// URIs
        """
        if not isinstance(file_path, str):
            return str(file_path)

        # Format from datasets library: {repo_id}@{revision}/{relative_path}
        if "@" in file_path and "/" in file_path:
            parts = file_path.split("@", 1)
            if len(parts) > 1:
                revision_and_path = parts[1]
                if "/" in revision_and_path:
                    return revision_and_path.split("/", 1)[1]

        # Remove hf://datasets/{repo_id}/ prefix
        if file_path.startswith(f"hf://datasets/{repo_id}/"):
            return file_path.replace(f"hf://datasets/{repo_id}/", "")

        # Extract path after repo_id from hf:// URI
        if file_path.startswith("hf://"):
            path_parts = file_path.split("/")
            if len(path_parts) > 3:  # hf://datasets/{repo_id}/...
                return "/".join(path_parts[3:])

        # Already a relative path
        return file_path

    def get_parquet_files_for_table(self, config: str) -> List[str]:
        """Get all Parquet files for a specific config across all splits.

        Args:
            config: Configuration name

        Returns:
            List of hf:// URIs for all Parquet files in this config
        """
        if config not in self.parquet_files:
            raise ValueError(f"Config {config} not found in dataset")

        files = []
        for split_files in self.parquet_files[config].values():
            for file_path in split_files:
                hf_uri = f"hf://datasets/{self.repo_id}/{file_path}"
                files.append(hf_uri)

        return files

    def get_sample_parquet_file(self, config: str) -> str:
        """Get a sample Parquet file for schema inference.

        Args:
            config: Configuration name

        Returns:
            hf:// URI to a sample Parquet file
        """
        files = self.get_parquet_files_for_table(config)
        if not files:
            raise ValueError(f"No Parquet files found for config {config}")
        return files[0]

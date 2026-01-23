"""Bridge between HuggingFace datasets and Apache Iceberg tables.

This module discovers HuggingFace datasets and converts them to TableInfo objects
that contain all the Iceberg metadata needed for table creation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import Features, get_dataset_config_names, get_dataset_split_names, load_dataset_builder
from pyiceberg.io.pyarrow import _pyarrow_to_schema_without_ids
from pyiceberg.schema import Schema, assign_fresh_schema_ids
from pyiceberg.types import NestedField, StringType

from faceberg.convert import FileInfo, TableInfo, build_split_partition_spec


# =============================================================================
# Schema Conversion Functions
# =============================================================================


def build_iceberg_schema_from_features(
    features, include_split_column: bool = True
) -> Schema:
    """
    Build an Iceberg Schema from HuggingFace dataset features using Arrow as an intermediate format.

    This approach ensures globally unique field IDs across nested structures by leveraging
    PyIceberg's built-in conversion and ID assignment logic.

    Args:
        features: HuggingFace Features object or dict of features
        include_split_column: If True, adds a 'split' column to the schema (default: True)

    Returns:
        PyIceberg Schema object with globally unique field IDs
    """
    # Convert to Features if dict
    if isinstance(features, dict):
        features = Features(features)

    # Convert: Features → Arrow Schema → Iceberg Schema (without IDs) → Assign fresh IDs
    # This ensures globally unique field IDs across all nested structures
    arrow_schema = features.arrow_schema
    iceberg_schema_no_ids = _pyarrow_to_schema_without_ids(arrow_schema)
    schema = assign_fresh_schema_ids(iceberg_schema_no_ids)

    # Add split column as the first field if requested
    if include_split_column:
        # Create split field (will get ID 1 after reassignment)
        # Note: Although the schema uses StringType, the actual Parquet data
        # will use dictionary encoding (int8 indices) for compression efficiency
        split_field = NestedField(
            field_id=-1,  # Temporary ID, will be reassigned
            name="split",
            field_type=StringType(),
            required=True,
        )
        # Prepend split field to existing fields
        new_fields = [split_field] + list(schema.fields)

        # Create new schema and reassign all field IDs globally
        # This ensures field IDs are globally unique across nested structures
        schema_with_split = Schema(*new_fields)
        schema = assign_fresh_schema_ids(schema_with_split)

    return schema


def infer_schema_from_dataset(
    repo_id: str,
    config_name: Optional[str] = None,
    token: Optional[str] = None,
    include_split_column: bool = True,
) -> Schema:
    """Infer Iceberg schema from a HuggingFace dataset.

    Args:
        repo_id: HuggingFace dataset repository ID
        config_name: Dataset configuration name (optional)
        token: HuggingFace API token (optional)
        include_split_column: If True, adds a 'split' column to the schema

    Returns:
        Iceberg Schema with globally unique field IDs
    """
    # Load dataset builder to access features
    builder = load_dataset_builder(repo_id, name=config_name, token=token)

    # Get features from builder info
    features = builder.info.features

    # Convert to Iceberg schema
    return build_iceberg_schema_from_features(features, include_split_column)


# =============================================================================
# Dataset Discovery and Bridging
# =============================================================================


@dataclass
class DatasetInfo:
    """Information about a HuggingFace dataset.

    This class discovers and represents the structure of a HuggingFace dataset,
    including its configurations, splits, and Parquet files. It serves as the
    discovery layer that gathers all necessary information before conversion
    to Iceberg format.
    """

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

    def to_table_infos(
        self,
        namespace: str = "default",
        table_name_prefix: Optional[str] = None,
        token: Optional[str] = None,
    ) -> List[TableInfo]:
        """Convert DatasetInfo to a list of TableInfo objects.

        This is the key bridge method that transforms HuggingFace dataset
        discovery information into Iceberg-ready table metadata.

        Each config in the dataset becomes a separate TableInfo object,
        which contains all the Iceberg metadata needed for table creation.

        Args:
            namespace: Iceberg namespace for the tables (default: "default")
            table_name_prefix: Prefix for table names (if None, uses last part of repo_id)
            token: HuggingFace API token (optional)

        Returns:
            List of TableInfo objects, one per config
        """
        # Determine table name prefix
        if table_name_prefix is None:
            # Use last part of repo_id (e.g., "squad" from "huggingface/squad")
            table_name_prefix = self.repo_id.split("/")[-1].replace("-", "_")

        table_infos = []

        for config_name in self.configs:
            # Load dataset builder to get features
            builder = load_dataset_builder(self.repo_id, name=config_name, token=token)
            features = builder.info.features

            if not features:
                raise ValueError(f"Dataset {self.repo_id} config {config_name} has no features")

            # Build Iceberg schema with split column
            schema = build_iceberg_schema_from_features(features, include_split_column=True)

            # Build partition spec (partitioned by split)
            partition_spec = build_split_partition_spec(schema)

            # Collect file information
            files = []
            for split_name, file_paths in self.parquet_files[config_name].items():
                for file_path in file_paths:
                    hf_uri = f"hf://datasets/{self.repo_id}/{file_path}"
                    # File size and row count will be read during metadata creation
                    files.append(
                        FileInfo(
                            path=hf_uri,
                            size_bytes=0,  # Will be enriched later
                            row_count=0,  # Will be enriched later
                            split=split_name,
                        )
                    )

            # Create table name
            table_name = f"{table_name_prefix}_{config_name}"

            # Create TableInfo
            table_info = TableInfo(
                namespace=namespace,
                table_name=table_name,
                schema=schema,
                partition_spec=partition_spec,
                files=files,
                source_repo=self.repo_id,
                source_config=config_name,
            )

            table_infos.append(table_info)

        return table_infos

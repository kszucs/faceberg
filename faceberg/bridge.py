"""Bridge between HuggingFace datasets and Apache Iceberg tables.

This module discovers HuggingFace datasets and converts them to TableInfo objects
that contain all the Iceberg metadata needed for table creation.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import (
    Features,
    get_dataset_config_names,
    load_dataset_builder,
)
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import HfApi, HfFileSystem
from pyiceberg.io.pyarrow import _pyarrow_to_schema_without_ids
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema, assign_fresh_schema_ids
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import ListType, MapType, NestedField, StringType, StructType

# =============================================================================
# Bridge Output Classes
# =============================================================================


@dataclass
class FileInfo:
    """Information about a data file in Iceberg table."""

    path: str  # Full hf:// URI to the file
    size_bytes: int  # File size in bytes
    row_count: int  # Number of rows in the file
    split: Optional[str] = None  # Split name (train, test, validation, etc.)


@dataclass
class TableInfo:
    """Complete information needed to create an Iceberg table.

    This class serves as the output of the bridge layer, containing all the
    metadata needed to convert a HuggingFace dataset into an Iceberg table.
    """

    # Table identity
    namespace: str  # Iceberg namespace (e.g., "default")
    table_name: str  # Table name (e.g., "squad_plain_text")

    # Iceberg schema and partitioning
    schema: Schema  # Iceberg schema with field IDs
    partition_spec: PartitionSpec  # Partition specification

    # Data files
    files: List[FileInfo]  # List of data files with metadata

    # Source metadata (for traceability)
    source_repo: str  # HuggingFace repo ID
    source_config: str  # Dataset configuration name
    source_revision: Optional[str] = None  # Git revision/SHA of the dataset

    @property
    def identifier(self) -> str:
        """Get table identifier in 'namespace.table_name' format."""
        return f"{self.namespace}.{self.table_name}"

    @property
    def total_rows(self) -> int:
        """Get total row count across all files."""
        return sum(f.row_count for f in self.files)

    @property
    def total_size(self) -> int:
        """Get total size in bytes across all files."""
        return sum(f.size_bytes for f in self.files)

    def get_table_properties(self) -> Dict[str, str]:
        """Get table properties for Iceberg metadata.

        Returns:
            Dictionary of table properties including source metadata and name mapping
        """
        # Create schema name mapping for Parquet files without embedded field IDs
        name_mapping = build_name_mapping(self.schema)

        properties = {
            "format-version": "2",
            "write.parquet.compression-codec": "snappy",
            "huggingface.dataset.repo": self.source_repo,
            "huggingface.dataset.config": self.source_config,
            "schema.name-mapping.default": json.dumps(name_mapping),
        }

        # Add revision if available
        if self.source_revision:
            properties["huggingface.dataset.revision"] = self.source_revision

        return properties


# =============================================================================
# Schema Name Mapping Helpers
# =============================================================================


def build_field_mapping(field: NestedField) -> Dict[str, any]:
    """Build name mapping for a single field, recursively handling nested types.

    Args:
        field: Iceberg NestedField to create mapping for

    Returns:
        Dictionary containing field-id, names, and optionally nested fields
    """
    mapping = {
        "field-id": field.field_id,
        "names": [field.name],
    }

    # Handle nested types
    if isinstance(field.field_type, StructType):
        # Recursively map nested struct fields
        nested_fields = []
        for nested_field in field.field_type.fields:
            nested_fields.append(build_field_mapping(nested_field))
        if nested_fields:
            mapping["fields"] = nested_fields
    elif isinstance(field.field_type, ListType):
        # Create mapping for the list element
        element_mapping = {
            "field-id": field.field_type.element_id,
            "names": ["element"],
        }
        # If element is a struct, recursively map its fields
        if isinstance(field.field_type.element_type, StructType):
            element_fields = []
            for nested_field in field.field_type.element_type.fields:
                element_fields.append(build_field_mapping(nested_field))
            if element_fields:
                element_mapping["fields"] = element_fields
        mapping["fields"] = [element_mapping]
    elif isinstance(field.field_type, MapType):
        # Create mappings for key and value
        map_fields = []

        # Map the key
        key_mapping = {
            "field-id": field.field_type.key_id,
            "names": ["key"],
        }
        if isinstance(field.field_type.key_type, StructType):
            key_fields = []
            for nested_field in field.field_type.key_type.fields:
                key_fields.append(build_field_mapping(nested_field))
            if key_fields:
                key_mapping["fields"] = key_fields
        map_fields.append(key_mapping)

        # Map the value
        value_mapping = {
            "field-id": field.field_type.value_id,
            "names": ["value"],
        }
        if isinstance(field.field_type.value_type, StructType):
            value_fields = []
            for nested_field in field.field_type.value_type.fields:
                value_fields.append(build_field_mapping(nested_field))
            if value_fields:
                value_mapping["fields"] = value_fields
        map_fields.append(value_mapping)

        mapping["fields"] = map_fields

    return mapping


def build_name_mapping(schema: Schema) -> List[Dict[str, any]]:
    """Build Iceberg name mapping from schema, recursively handling nested fields.

    Name mapping is used to map Parquet column names to Iceberg field IDs for
    files that don't have embedded field IDs.

    Args:
        schema: Iceberg schema to create mapping for

    Returns:
        List of field mappings with field-id, names, and nested fields
    """
    fields = []
    for field in schema.fields:
        fields.append(build_field_mapping(field))
    return fields


# ============================================================================
# Schema and Partition Spec Builders
# ============================================================================


def build_split_partition_spec(schema: Schema) -> PartitionSpec:
    """Build a partition spec that uses 'split' as a partition key.

    This creates an identity partition on the split column, which means the split
    value will be stored in metadata and used for partition pruning.

    Args:
        schema: Iceberg schema containing a 'split' field

    Returns:
        PartitionSpec with split as partition key

    Raises:
        ValueError: If schema doesn't contain a 'split' field
    """
    split_field = schema.find_field("split")
    if split_field is None:
        raise ValueError("Schema must contain a 'split' field to create split partition spec")

    return PartitionSpec(
        PartitionField(
            source_id=split_field.field_id,
            field_id=1000,  # Partition field IDs start at 1000
            transform=IdentityTransform(),
            name="split",
        ),
        spec_id=0,
    )


def build_iceberg_schema_from_features(features, include_split_column: bool = True) -> Schema:
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
        # The split column is optional since it doesn't exist in the source Parquet files,
        # it's derived from partition metadata
        split_field = NestedField(
            field_id=-1,  # Temporary ID, will be reassigned
            name="split",
            field_type=StringType(),
            required=False,
        )
        # Prepend split field to existing fields
        new_fields = [split_field] + list(schema.fields)

        # Create new schema and reassign all field IDs globally
        # This ensures field IDs are globally unique across nested structures
        schema_with_split = Schema(*new_fields)
        schema = assign_fresh_schema_ids(schema_with_split)

    return schema


# =============================================================================
# Datasets Helper Functions
# =============================================================================


def resolve_hf_path(fs: HfFileSystem, file_path: str) -> str:
    """Resolve HuggingFace file path to relative path using official API.

    Uses HfFileSystem.resolve_path to properly parse file paths from the datasets
    library, which can come in various formats.

    Args:
        fs: HfFileSystem instance for path resolution
        file_path: File path in various formats (hf://, repo@revision/path, etc.)

    Returns:
        Relative path suitable for hf:// URIs

    Raises:
        Exception: If path cannot be resolved
    """
    # Handle hf:// URIs
    if file_path.startswith("hf://"):
        resolved = fs.resolve_path(file_path)
        return resolved.path_in_repo

    # Handle format from datasets library: {repo_id}@{revision}/{relative_path}
    if "@" in file_path and "/" in file_path:
        # Convert to datasets/repo_id@revision/path format for resolve_path
        datasets_path = f"datasets/{file_path}"
        resolved = fs.resolve_path(datasets_path)
        return resolved.path_in_repo

    # Already a relative path
    return file_path


def load_dataset_builder_safe(
    repo_id: str,
    config_name: Optional[str] = None,
    token: Optional[str] = None,
):
    """Load dataset builder while avoiding picking up local files.

    Changes to a temporary directory before loading to ensure the datasets
    library doesn't pick up local files in the current directory.

    Args:
        repo_id: HuggingFace dataset repository ID
        config_name: Optional configuration name
        token: Optional HuggingFace API token

    Returns:
        Dataset builder object

    Raises:
        Exception: If loading fails
    """
    original_cwd = os.getcwd()

    try:
        # Change to a temporary directory to avoid dataset library picking up local files
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            return load_dataset_builder(repo_id, name=config_name, token=token)
    finally:
        # Always restore the original directory
        os.chdir(original_cwd)


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
    revision: Optional[str] = None  # Git revision/SHA of the dataset

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
        except DatasetNotFoundError as e:
            raise ValueError(f"Dataset {repo_id} not found or not accessible") from e

        # Filter to requested configs
        if configs:
            missing_configs = set(configs) - set(all_configs)
            if missing_configs:
                raise ValueError(
                    f"Configs not found in dataset: {missing_configs}. Available: {all_configs}"
                )
            discovered_configs = [c for c in all_configs if c in configs]
        else:
            discovered_configs = all_configs

        # Discover splits and files for each config
        splits_dict = {}
        parquet_files = {}
        revision = None

        for config_name in discovered_configs:
            config_splits, config_files, config_revision = cls._discover_config(
                repo_id, config_name, token
            )
            splits_dict[config_name] = config_splits
            parquet_files[config_name] = config_files
            if config_revision and not revision:
                revision = config_revision

        if not parquet_files or all(not files for files in parquet_files.values()):
            raise ValueError(f"No Parquet files found in dataset {repo_id}")

        return cls(
            repo_id=repo_id,
            configs=discovered_configs,
            splits=splits_dict,
            parquet_files=parquet_files,
            revision=revision,
        )

    @staticmethod
    def _discover_config(
        repo_id: str, config_name: str, token: Optional[str]
    ) -> tuple[List[str], Dict[str, List[str]], Optional[str]]:
        """Discover splits and files for a specific config.

        Uses the datasets library's official inspection utilities for robust discovery.

        Args:
            repo_id: Dataset repository ID
            config_name: Configuration name
            token: HuggingFace API token

        Returns:
            Tuple of (splits list, files dict mapping split -> file paths, revision)

        Raises:
            ValueError: If no data files found or builder cannot be loaded
        """
        builder = load_dataset_builder_safe(repo_id, config_name=config_name, token=token)

        # Get dataset revision using official HuggingFace Hub API
        api = HfApi()
        dataset_info = api.dataset_info(repo_id, token=token)
        revision = dataset_info.sha

        # Get splits from builder.info.splits if available, otherwise from data_files
        splits = []
        if builder.info.splits:
            splits = list(builder.info.splits.keys())

        # Get data files for parquet file paths
        data_files = builder.config.data_files if hasattr(builder.config, "data_files") else None

        if not data_files or not isinstance(data_files, dict):
            raise ValueError(
                f"No data files found for dataset {repo_id} config {config_name}. "
                "Cannot create Iceberg table without source data files."
            )

        # Use splits from data_files if not found in builder.info
        if not splits:
            splits = list(data_files.keys())

        # Resolve all file paths using HfFileSystem
        fs = HfFileSystem(token=token)
        files = {
            split: [resolve_hf_path(fs, path) for path in paths]
            for split, paths in data_files.items()
        }

        return splits, files, revision

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

    def to_table_info(
        self,
        namespace: str,
        table_name: str,
        config: str,
        token: Optional[str] = None,
    ) -> TableInfo:
        """Convert DatasetInfo to a single TableInfo for a specific config.

        This method creates table metadata for a single HuggingFace dataset config
        with an explicit table name, supporting the namespace-based configuration.

        Args:
            namespace: Iceberg namespace for the table
            table_name: Explicit table name (no auto-generation)
            config: Specific config to create table for
            token: HuggingFace API token (optional)

        Returns:
            TableInfo object

        Raises:
            ValueError: If config not found in dataset
        """
        if config not in self.configs:
            raise ValueError(
                f"Config '{config}' not found in dataset {self.repo_id}. "
                f"Available configs: {', '.join(self.configs)}"
            )

        # Get features from dataset builder
        builder = load_dataset_builder_safe(self.repo_id, config_name=config, token=token)
        features = builder.info.features

        # Features must be available from builder
        if not features:
            raise ValueError(
                f"Dataset {self.repo_id} config {config} has no features available. "
                "Features must be provided by the dataset builder."
            )

        # Build Iceberg schema with split column
        schema = build_iceberg_schema_from_features(features, include_split_column=True)

        # Build partition spec (partitioned by split)
        partition_spec = build_split_partition_spec(schema)

        # Collect file information
        files = []
        for split_name, file_paths in self.parquet_files[config].items():
            for file_path in file_paths:
                hf_uri = f"hf://datasets/{self.repo_id}/{file_path}"
                files.append(
                    FileInfo(
                        path=hf_uri,
                        size_bytes=0,  # Will be enriched later
                        row_count=0,  # Will be enriched later
                        split=split_name,
                    )
                )

        # Create TableInfo with explicit naming
        return TableInfo(
            namespace=namespace,
            table_name=table_name,  # Direct from config, no auto-generation
            schema=schema,
            partition_spec=partition_spec,
            files=files,
            source_repo=self.repo_id,
            source_config=config,
            source_revision=self.revision,
        )

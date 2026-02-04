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
    load_dataset_builder,
)
from huggingface_hub import HfApi
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

    uri: str  # Full hf:// URI to the file
    split: Optional[str] = None  # Split name (train, test, validation, etc.)
    size_bytes: Optional[int] = None  # File size in bytes (enriched later)
    row_count: Optional[int] = None  # Number of rows in the file (enriched later)


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
    data_files: List[FileInfo]  # List of data files with metadata
    data_dir: str  # Data directory path relative to repo root

    # Source metadata (for traceability)
    dataset_repo: str  # HuggingFace repo ID
    dataset_config: str  # Dataset configuration name
    dataset_revision: str  # Git revision/SHA of the dataset

    @property
    def identifier(self) -> str:
        """Get table identifier in 'namespace.table_name' format."""
        return f"{self.namespace}.{self.table_name}"

    @property
    def total_rows(self) -> int:
        """Get total row count across all files."""
        return sum(f.row_count for f in self.data_files if f.row_count is not None)

    @property
    def total_size(self) -> int:
        """Get total size in bytes across all files."""
        return sum(f.size_bytes for f in self.data_files if f.size_bytes is not None)

    def get_table_properties(self) -> Dict[str, str]:
        """Get table properties for Iceberg metadata.

        Returns:
            Dictionary of table properties including source metadata and name mapping
        """
        # Create schema name mapping for Parquet files without embedded field IDs
        name_mapping = iceberg_name_mapping(self.schema)

        # Use data directory from discovery
        data_path = (
            f"hf://datasets/{self.dataset_repo}/{self.data_dir}"
            if self.data_dir
            else f"hf://datasets/{self.dataset_repo}"
        )

        # TODO(kszucs): split should be configurable
        properties = {
            "format-version": "3",
            "write.parquet.compression-codec": "snappy",
            "write.py-location-provider.impl": "faceberg.catalog.HfLocationProvider",
            "write.data.path": data_path,
            # HuggingFace source metadata
            "hf.dataset.repo": self.dataset_repo,
            "hf.dataset.config": self.dataset_config,
            "hf.dataset.revision": self.dataset_revision,
            # Write configuration
            "hf.write.pattern": "{split}-{uuid}-iceberg.parquet",
            "hf.write.split": "train",
            # Schema mapping
            "schema.name-mapping.default": json.dumps(name_mapping),
        }

        return properties


# =============================================================================
# Iceberg Helpers (Schema and Metadata)
# =============================================================================


def iceberg_field_mapping(field: NestedField) -> Dict[str, any]:
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
            nested_fields.append(iceberg_field_mapping(nested_field))
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
                element_fields.append(iceberg_field_mapping(nested_field))
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
                key_fields.append(iceberg_field_mapping(nested_field))
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
                value_fields.append(iceberg_field_mapping(nested_field))
            if value_fields:
                value_mapping["fields"] = value_fields
        map_fields.append(value_mapping)

        mapping["fields"] = map_fields

    return mapping


def iceberg_name_mapping(schema: Schema) -> List[Dict[str, any]]:
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
        fields.append(iceberg_field_mapping(field))
    return fields


def iceberg_partition_spec(schema: Schema) -> PartitionSpec:
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


def iceberg_schema_from_features(features, include_split_column: bool = True) -> Schema:
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
# Dataset Helpers (HuggingFace)
# =============================================================================


def dataset_builder_safe(
    repo_id: str,
    config: str,
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
            return load_dataset_builder(repo_id, config, token=token)
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
    including its configuration, splits, and Parquet files. It serves as the
    discovery layer that gathers all necessary information before conversion
    to Iceberg format.
    """

    repo_id: str
    config: str
    splits: List[str]
    revision: str  # Git revision/SHA of the dataset
    features: Features  # HuggingFace dataset features
    data_dir: str
    data_files: Dict[str, List[FileInfo]]  # split -> list of FileInfo objects with metadata

    @classmethod
    def discover(
        cls,
        repo_id: str,
        config: str,
        token: Optional[str] = None,
        since_revision: Optional[str] = None,
    ) -> "DatasetInfo":
        """Discover Parquet files and structure in a HuggingFace dataset.

        Discovery process:
        1. Validate config exists in dataset
        2. Load dataset builder to get metadata
        3. Extract splits from builder
        4. Get data files (fully qualified URIs with revision)
           - If since_revision is provided, only get files added since that revision
           - Otherwise, get all files from builder
        5. Get dataset revision (SHA) from Hub
        6. Extract data directory from config or URIs
        7. Return DatasetInfo with all metadata

        Args:
            repo_id: HuggingFace dataset repository ID (e.g., "kszucs/dataset1")
            config: Configuration name to discover
            token: HuggingFace API token (uses HF_TOKEN env var if not provided)
            since_revision: Optional revision SHA to get only files added since that revision

        Returns:
            DatasetInfo with discovered structure

        Raises:
            ValueError: If dataset not found or config doesn't exist
        """
        try:
            builder = dataset_builder_safe(repo_id, config=config, token=token)
        except Exception as e:
            raise ValueError(
                f"Dataset {repo_id} config {config} not found or not accessible: {e}"
            ) from e

        revision = builder.hash
        features = builder.info.features

        # Fetch dataset info with file metadata once for current revision
        api = HfApi(token=token)
        # get the current revision's dataset info
        new_info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
        new_files = {s.rfilename: s for s in new_info.siblings}

        # if since_revision is provided, get the old dataset info for diffing
        old_files = {}
        if since_revision:
            old_info = api.dataset_info(repo_id, revision=since_revision, files_metadata=True)
            old_files = {s.rfilename: s for s in old_info.siblings}

        added_paths = new_files.keys() - old_files.keys()
        added_uris = {
            f"hf://datasets/{repo_id}@{revision}/{path}": new_files[path] for path in added_paths
        }

        # Process data files: filter, extract paths, and create FileInfo objects
        data_files = {}
        for split, file_uris in builder.config.data_files.items():
            file_infos = []
            for uri in file_uris:
                if uri in added_uris:
                    size_bytes = added_uris[uri].size
                    file_infos.append(FileInfo(uri=uri, split=split, size_bytes=size_bytes))
            data_files[split] = file_infos

        # Extract common data directory from all file paths
        try:
            file_dirs = [os.path.dirname(path) for path in new_files.keys()]
            data_dir = os.path.commonpath(file_dirs)
        except ValueError as e:
            raise ValueError(
                f"Unable to determine common data directory from files: {new_files.keys()}"
            ) from e

        # Get list of splits
        splits = list(data_files.keys())

        return cls(
            repo_id=repo_id,
            config=config,
            splits=splits,
            data_files=data_files,
            data_dir=data_dir,
            features=features,
            revision=revision,
        )

    def to_table_info(
        self,
        namespace: str,
        table_name: str,
    ) -> TableInfo:
        """Convert DatasetInfo to TableInfo.

        This method creates table metadata for the HuggingFace dataset config
        with an explicit table name, supporting the namespace-based configuration.

        Args:
            namespace: Iceberg namespace for the table
            table_name: Explicit table name (no auto-generation)

        Returns:
            TableInfo object
        """
        # Build Iceberg schema with split column
        schema = iceberg_schema_from_features(self.features, include_split_column=True)

        # Build partition spec (partitioned by split)
        partition_spec = iceberg_partition_spec(schema)

        # Collect file information (already FileInfo objects with sizes)
        files = []
        for file_infos in self.data_files.values():
            files.extend(file_infos)

        # Create TableInfo with explicit naming
        return TableInfo(
            namespace=namespace,
            table_name=table_name,  # Direct from config, no auto-generation
            schema=schema,
            partition_spec=partition_spec,
            data_files=files,
            data_dir=self.data_dir,
            dataset_repo=self.repo_id,
            dataset_config=self.config,
            dataset_revision=self.revision,
        )

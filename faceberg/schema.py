"""Schema conversion from HuggingFace datasets to Iceberg."""

from typing import Optional

from datasets import Features, load_dataset_builder
from pyiceberg.io.pyarrow import _pyarrow_to_schema_without_ids
from pyiceberg.schema import Schema, assign_fresh_schema_ids
from pyiceberg.types import NestedField, StringType


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

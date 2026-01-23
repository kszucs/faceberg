"""Tests for the bridge layer (dataset discovery, schema conversion, and TableInfo creation)."""

import os

import pytest
from datasets import Features
from datasets.features import ClassLabel, Sequence, Value
from pyiceberg.schema import Schema
from pyiceberg.types import (
    IntegerType,
    ListType,
    LongType,
    StringType,
    StructType,
)

from faceberg.bridge import (
    DatasetInfo,
    build_iceberg_schema_from_features,
    infer_schema_from_dataset,
)


def test_discover_public_dataset():
    """Test discovering a public HuggingFace dataset."""
    # Test with a known public dataset
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb")

    assert dataset_info.repo_id == "stanfordnlp/imdb"
    assert len(dataset_info.configs) > 0
    assert "plain_text" in dataset_info.configs

    # Check splits
    assert "plain_text" in dataset_info.splits
    splits = dataset_info.splits["plain_text"]
    assert "train" in splits
    assert "test" in splits
    assert "unsupervised" in splits

    # Check Parquet files
    assert "plain_text" in dataset_info.parquet_files
    assert "train" in dataset_info.parquet_files["plain_text"]
    train_files = dataset_info.parquet_files["plain_text"]["train"]
    assert len(train_files) > 0
    assert all(isinstance(f, str) for f in train_files)


def test_discover_with_specific_config():
    """Test discovering a dataset with a specific config."""
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb", configs=["plain_text"])

    assert dataset_info.configs == ["plain_text"]
    assert "plain_text" in dataset_info.splits


def test_discover_nonexistent_dataset():
    """Test discovering a non-existent dataset raises ValueError."""
    with pytest.raises(ValueError, match="not found or not accessible"):
        DatasetInfo.discover("nonexistent/fake-dataset-12345")


def test_discover_nonexistent_config():
    """Test discovering a non-existent config raises ValueError."""
    with pytest.raises(ValueError, match="Configs not found"):
        DatasetInfo.discover("stanfordnlp/imdb", configs=["fake_config"])


def test_get_parquet_files_for_table():
    """Test getting Parquet files for a specific config."""
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb")

    files = dataset_info.get_parquet_files_for_table("plain_text")

    assert len(files) > 0
    assert all(f.startswith("hf://datasets/stanfordnlp/imdb/") for f in files)
    assert all(f.endswith(".parquet") for f in files)


def test_get_sample_parquet_file():
    """Test getting a sample Parquet file."""
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb")

    sample = dataset_info.get_sample_parquet_file("plain_text")

    assert sample.startswith("hf://datasets/stanfordnlp/imdb/")
    assert sample.endswith(".parquet")


def test_get_parquet_files_nonexistent_config():
    """Test getting Parquet files for non-existent config raises ValueError."""
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb")

    with pytest.raises(ValueError, match="Config .* not found"):
        dataset_info.get_parquet_files_for_table("fake_config")


def test_extract_relative_path():
    """Test path extraction from various formats."""
    # Test with revision format
    path1 = "repo@abc123/plain_text/train-00000.parquet"
    result1 = DatasetInfo._extract_relative_path("repo", path1)
    assert result1 == "plain_text/train-00000.parquet"

    # Test with hf:// URI
    path2 = "hf://datasets/repo/plain_text/train-00000.parquet"
    result2 = DatasetInfo._extract_relative_path("repo", path2)
    assert result2 == "plain_text/train-00000.parquet"

    # Test with relative path
    path3 = "plain_text/train-00000.parquet"
    result3 = DatasetInfo._extract_relative_path("repo", path3)
    assert result3 == "plain_text/train-00000.parquet"


def test_to_table_infos():
    """Test converting DatasetInfo to TableInfo objects."""
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb", configs=["plain_text"])

    # Convert to TableInfo
    table_infos = dataset_info.to_table_infos(
        namespace="default",
        table_name_prefix="imdb",
    )

    # Should have one TableInfo per config
    assert len(table_infos) == 1

    table_info = table_infos[0]
    assert table_info.namespace == "default"
    assert table_info.table_name == "imdb_plain_text"
    assert table_info.identifier == "default.imdb_plain_text"
    assert table_info.source_repo == "stanfordnlp/imdb"
    assert table_info.source_config == "plain_text"

    # Check schema
    assert table_info.schema is not None
    assert len(table_info.schema.fields) > 0
    # Should have split column as first field
    assert table_info.schema.fields[0].name == "split"

    # Check partition spec (should be partitioned by split)
    assert table_info.partition_spec is not None
    assert len(table_info.partition_spec.fields) == 1
    assert table_info.partition_spec.fields[0].name == "split"

    # Check files
    assert len(table_info.files) > 0
    for file_info in table_info.files:
        assert file_info.path.startswith("hf://datasets/stanfordnlp/imdb/")
        assert file_info.split in ["train", "test", "unsupervised"]

    # Check properties
    props = table_info.get_table_properties()
    assert props["faceberg.source.repo"] == "stanfordnlp/imdb"
    assert props["faceberg.source.config"] == "plain_text"


# =============================================================================
# Schema Conversion Tests
# =============================================================================


def test_build_schema_from_simple_features():
    """Test building schema from simple features."""
    features = Features(
        {
            "text": Value("string"),
            "label": Value("int64"),
        }
    )

    schema = build_iceberg_schema_from_features(features, include_split_column=True)

    # Check split column is first
    assert schema.fields[0].name == "split"
    assert schema.fields[0].field_id == 1
    assert isinstance(schema.fields[0].field_type, StringType)

    # Check original fields
    assert len(schema.fields) == 3  # split + text + label
    field_names = [f.name for f in schema.fields]
    assert "text" in field_names
    assert "label" in field_names


def test_build_schema_without_split_column():
    """Test building schema without split column."""
    features = Features(
        {
            "id": Value("int64"),
            "text": Value("string"),
        }
    )

    schema = build_iceberg_schema_from_features(features, include_split_column=False)

    # No split column
    field_names = [f.name for f in schema.fields]
    assert "split" not in field_names
    assert len(schema.fields) == 2


def test_build_schema_with_nested_features():
    """Test building schema with nested structures."""
    features = Features(
        {
            "id": Value("int64"),
            "metadata": {
                "title": Value("string"),
                "author": Value("string"),
                "year": Value("int32"),
            },
            "tags": Sequence(Value("string")),
        }
    )

    schema = build_iceberg_schema_from_features(features, include_split_column=False)

    # Verify structure
    field_names = [f.name for f in schema.fields]
    assert "id" in field_names
    assert "metadata" in field_names
    assert "tags" in field_names

    # Find metadata field
    metadata_field = next(f for f in schema.fields if f.name == "metadata")
    assert isinstance(metadata_field.field_type, StructType)

    # Find tags field
    tags_field = next(f for f in schema.fields if f.name == "tags")
    assert isinstance(tags_field.field_type, ListType)


def test_build_schema_with_class_label():
    """Test building schema with ClassLabel feature."""
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=["negative", "positive"]),
        }
    )

    schema = build_iceberg_schema_from_features(features, include_split_column=False)

    # ClassLabel should be converted to an integer type
    label_field = next(f for f in schema.fields if f.name == "label")
    # ClassLabel is typically represented as int64 in Arrow
    assert isinstance(label_field.field_type, (IntegerType, LongType))


def test_unique_field_ids():
    """Test that all field IDs are unique across nested structures."""
    features = Features(
        {
            "id": Value("int64"),
            "nested": {
                "field1": Value("string"),
                "field2": Value("int32"),
                "deeper": {
                    "field3": Value("string"),
                },
            },
            "list_field": Sequence(Value("string")),
        }
    )

    schema = build_iceberg_schema_from_features(features, include_split_column=True)

    # Collect all field IDs recursively
    def collect_field_ids(field_type, ids=None):
        if ids is None:
            ids = []

        if isinstance(field_type, StructType):
            for field in field_type.fields:
                ids.append(field.field_id)
                collect_field_ids(field.field_type, ids)
        elif isinstance(field_type, ListType):
            ids.append(field_type.element_id)
            collect_field_ids(field_type.element_type, ids)

        return ids

    # Get all field IDs
    all_ids = [f.field_id for f in schema.fields]
    for field in schema.fields:
        all_ids.extend(collect_field_ids(field.field_type))

    # Check all IDs are unique
    assert len(all_ids) == len(set(all_ids)), f"Duplicate field IDs found: {all_ids}"


def test_infer_schema_from_dataset():
    """Test inferring schema from actual HuggingFace dataset."""
    # Skip if no HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")

    # Test with a known dataset
    schema = infer_schema_from_dataset(
        repo_id="stanfordnlp/imdb",
        config_name="plain_text",
        token=os.getenv("HF_TOKEN"),
        include_split_column=True,
    )

    # Verify schema structure
    assert isinstance(schema, Schema)
    assert len(schema.fields) > 0

    # Check split column is first
    assert schema.fields[0].name == "split"
    assert schema.fields[0].field_id == 1

    # Should have text and label fields from IMDB
    field_names = [f.name for f in schema.fields]
    assert "text" in field_names
    assert "label" in field_names


def test_features_dict_to_features_object():
    """Test that dict features are properly converted to Features object."""
    features_dict = {
        "id": Value("int64"),
        "text": Value("string"),
    }

    schema = build_iceberg_schema_from_features(features_dict, include_split_column=False)

    # Should work the same as passing Features object
    assert isinstance(schema, Schema)
    field_names = [f.name for f in schema.fields]
    assert "id" in field_names
    assert "text" in field_names


if __name__ == "__main__":
    # Run basic smoke test
    print("Running basic discovery test...")
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb")
    print(f"✓ Discovered {len(dataset_info.configs)} config(s)")
    print(f"✓ Found splits: {list(dataset_info.splits.keys())}")

    files = dataset_info.get_parquet_files_for_table("plain_text")
    print(f"✓ Found {len(files)} Parquet files")

    sample = dataset_info.get_sample_parquet_file("plain_text")
    print(f"✓ Sample file: {sample}")

    print("\nRunning schema conversion tests...")
    test_build_schema_from_simple_features()
    print("✓ Simple features test passed")

    test_build_schema_without_split_column()
    print("✓ No split column test passed")

    test_build_schema_with_nested_features()
    print("✓ Nested features test passed")

    test_unique_field_ids()
    print("✓ Unique field IDs test passed")

    print("\n✓ All tests passed!")

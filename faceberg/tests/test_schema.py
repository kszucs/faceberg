"""Tests for schema conversion."""

import os

import pytest
from datasets import Features
from datasets.features import ClassLabel, Sequence, Value
from pyiceberg.schema import Schema
from pyiceberg.types import (
    IntegerType,
    ListType,
    LongType,
    NestedField,
    StringType,
    StructType,
)

from faceberg.schema import build_iceberg_schema_from_features, infer_schema_from_dataset


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
    # Run basic tests
    print("Testing simple features...")
    test_build_schema_from_simple_features()
    print("✓ Simple features test passed")

    print("Testing without split column...")
    test_build_schema_without_split_column()
    print("✓ No split column test passed")

    print("Testing nested features...")
    test_build_schema_with_nested_features()
    print("✓ Nested features test passed")

    print("Testing unique field IDs...")
    test_unique_field_ids()
    print("✓ Unique field IDs test passed")

    print("\n✓ All basic tests passed!")

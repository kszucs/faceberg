"""Tests for the bridge layer (dataset discovery, schema conversion, and TableInfo creation)."""

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
    FileInfo,
    dataset_builder_safe,
    iceberg_schema_from_features,
)


def test_discover_public_dataset():
    """Test discovering a public HuggingFace dataset."""
    # Test with a known public dataset
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb", config="plain_text")

    assert dataset_info.repo_id == "stanfordnlp/imdb"
    assert dataset_info.config == "plain_text"

    # Check splits
    splits = dataset_info.splits
    assert "train" in splits
    assert "test" in splits
    assert "unsupervised" in splits

    # Check Parquet files
    assert "train" in dataset_info.data_files
    train_files = dataset_info.data_files["train"]
    assert len(train_files) > 0
    assert all(isinstance(f, FileInfo) for f in train_files)
    # Check that file sizes are populated
    assert all(f.size_bytes is not None and f.size_bytes > 0 for f in train_files)


def test_discover_with_specific_config():
    """Test discovering a dataset with a specific config."""
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb", config="plain_text")

    assert dataset_info.config == "plain_text"
    assert len(dataset_info.splits) > 0


def test_discover_nonexistent_dataset():
    """Test discovering a non-existent dataset raises ValueError."""
    with pytest.raises(ValueError, match="not found or not accessible"):
        DatasetInfo.discover("nonexistent/fake-dataset-12345", config="default")


def test_discover_nonexistent_config():
    """Test discovering a non-existent config raises ValueError."""
    with pytest.raises(ValueError, match="Config .* not found"):
        DatasetInfo.discover("stanfordnlp/imdb", config="fake_config")


def test_to_table_infos():
    """Test converting DatasetInfo to TableInfo objects."""
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb", config="plain_text")

    # Convert to TableInfo
    table_info = dataset_info.to_table_info(
        namespace="default",
        table_name="imdb_plain_text",
    )

    assert table_info.namespace == "default"
    assert table_info.table_name == "imdb_plain_text"
    assert table_info.identifier == "default.imdb_plain_text"
    assert table_info.dataset_repo == "stanfordnlp/imdb"
    assert table_info.dataset_config == "plain_text"

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
    assert len(table_info.data_files) > 0
    for file_info in table_info.data_files:
        # URIs now include revision: hf://datasets/stanfordnlp/imdb@<revision>/...
        assert file_info.uri.startswith("hf://datasets/stanfordnlp/imdb")
        assert "@" in file_info.uri or "/" in file_info.uri
        assert file_info.split in ["train", "test", "unsupervised"]

    # Check properties
    props = table_info.get_table_properties()
    assert props["hf.dataset.repo"] == "stanfordnlp/imdb"
    assert props["hf.dataset.config"] == "plain_text"


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

    schema = iceberg_schema_from_features(features, include_split_column=True)

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

    schema = iceberg_schema_from_features(features, include_split_column=False)

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

    schema = iceberg_schema_from_features(features, include_split_column=False)

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

    schema = iceberg_schema_from_features(features, include_split_column=False)

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

    schema = iceberg_schema_from_features(features, include_split_column=True)

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


def test_features_dict_to_features_object():
    """Test that dict features are properly converted to Features object."""
    features_dict = {
        "id": Value("int64"),
        "text": Value("string"),
    }

    schema = iceberg_schema_from_features(features_dict, include_split_column=False)

    # Should work the same as passing Features object
    assert isinstance(schema, Schema)
    field_names = [f.name for f in schema.fields]
    assert "id" in field_names
    assert "text" in field_names


def test_dataset_builder_safe():
    """Test that the safe builder loader works and avoids local files."""
    # Test with a known public dataset
    builder = dataset_builder_safe("stanfordnlp/imdb", config="plain_text")

    assert builder is not None
    assert builder.info is not None
    assert builder.info.features is not None


def test_dataset_builder_safe_nonexistent():
    """Test that safe builder loader raises error for non-existent dataset."""
    with pytest.raises(Exception):
        dataset_builder_safe("nonexistent/fake-dataset-12345")


def test_table_properties_use_hf_prefix():
    """Test that table properties use hf.dataset.* prefix."""
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb", config="plain_text")
    table_info = dataset_info.to_table_info(
        namespace="default",
        table_name="imdb_plain_text",
    )

    props = table_info.get_table_properties()

    # Check that properties use hf.dataset prefix
    assert "hf.dataset.repo" in props
    assert "hf.dataset.config" in props
    assert props["hf.dataset.repo"] == "stanfordnlp/imdb"
    assert props["hf.dataset.config"] == "plain_text"

    # Check that revision is always included (now mandatory)
    assert "hf.dataset.revision" in props
    assert props["hf.dataset.revision"] == table_info.dataset_revision

    # Verify old prefix is not used
    assert "faceberg.source.repo" not in props
    assert "faceberg.source.config" not in props
    assert "faceberg.source.revision" not in props


def test_table_info_name_mapping_with_nested_structs():
    """Test that name mapping includes nested struct fields."""
    import json

    from pyiceberg.types import IntegerType, NestedField, StringType, StructType

    from faceberg.bridge import TableInfo

    # Create a schema with nested structs
    schema = Schema(
        NestedField(field_id=1, name="id", field_type=StringType(), required=False),
        NestedField(
            field_id=2,
            name="metadata",
            field_type=StructType(
                NestedField(field_id=3, name="author", field_type=StringType(), required=False),
                NestedField(field_id=4, name="year", field_type=IntegerType(), required=False),
            ),
            required=False,
        ),
    )

    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table_info = TableInfo(
        namespace="test",
        table_name="table",
        schema=schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
        data_files=[],
        data_dir="data",
        dataset_repo="test/repo",
        dataset_config="default",
        dataset_revision="abc123",
    )

    properties = table_info.get_table_properties()
    name_mapping = json.loads(properties["schema.name-mapping.default"])

    # Check top-level fields
    assert len(name_mapping) == 2
    assert name_mapping[0]["field-id"] == 1
    assert name_mapping[0]["names"] == ["id"]

    # Check nested struct field
    metadata_mapping = name_mapping[1]
    assert metadata_mapping["field-id"] == 2
    assert metadata_mapping["names"] == ["metadata"]
    assert "fields" in metadata_mapping
    assert len(metadata_mapping["fields"]) == 2

    # Check nested struct's child fields
    assert metadata_mapping["fields"][0]["field-id"] == 3
    assert metadata_mapping["fields"][0]["names"] == ["author"]
    assert metadata_mapping["fields"][1]["field-id"] == 4
    assert metadata_mapping["fields"][1]["names"] == ["year"]


def test_table_info_name_mapping_with_lists():
    """Test that name mapping includes list element mappings."""
    import json

    from pyiceberg.types import ListType, NestedField, StringType, StructType

    from faceberg.bridge import TableInfo

    # Create a schema with list of strings and list of structs
    schema = Schema(
        NestedField(field_id=1, name="id", field_type=StringType(), required=False),
        NestedField(
            field_id=2,
            name="tags",
            field_type=ListType(element_id=3, element_type=StringType(), element_required=False),
            required=False,
        ),
        NestedField(
            field_id=4,
            name="items",
            field_type=ListType(
                element_id=5,
                element_type=StructType(
                    NestedField(field_id=6, name="name", field_type=StringType(), required=False),
                    NestedField(field_id=7, name="value", field_type=StringType(), required=False),
                ),
                element_required=False,
            ),
            required=False,
        ),
    )

    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table_info = TableInfo(
        namespace="test",
        table_name="table",
        schema=schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
        data_files=[],
        data_dir="data",
        dataset_repo="test/repo",
        dataset_config="default",
        dataset_revision="abc123",
    )

    properties = table_info.get_table_properties()
    name_mapping = json.loads(properties["schema.name-mapping.default"])

    # Check list of strings (tags)
    tags_mapping = name_mapping[1]
    assert tags_mapping["field-id"] == 2
    assert tags_mapping["names"] == ["tags"]
    assert "fields" in tags_mapping
    assert len(tags_mapping["fields"]) == 1

    # Check element mapping for simple list
    element_mapping = tags_mapping["fields"][0]
    assert element_mapping["field-id"] == 3
    assert element_mapping["names"] == ["element"]

    # Check list of structs (items)
    items_mapping = name_mapping[2]
    assert items_mapping["field-id"] == 4
    assert items_mapping["names"] == ["items"]
    assert "fields" in items_mapping

    # Check element mapping for list of structs
    items_element = items_mapping["fields"][0]
    assert items_element["field-id"] == 5
    assert items_element["names"] == ["element"]
    assert "fields" in items_element

    # Check struct fields within list element
    assert len(items_element["fields"]) == 2
    assert items_element["fields"][0]["field-id"] == 6
    assert items_element["fields"][0]["names"] == ["name"]
    assert items_element["fields"][1]["field-id"] == 7
    assert items_element["fields"][1]["names"] == ["value"]


def test_table_info_name_mapping_with_maps():
    """Test that name mapping includes map key and value mappings."""
    import json

    from pyiceberg.types import IntegerType, MapType, NestedField, StringType, StructType

    from faceberg.bridge import TableInfo

    # Create a schema with a map
    schema = Schema(
        NestedField(field_id=1, name="id", field_type=StringType(), required=False),
        NestedField(
            field_id=2,
            name="metadata",
            field_type=MapType(
                key_id=3,
                key_type=StringType(),
                value_id=4,
                value_type=StructType(
                    NestedField(field_id=5, name="count", field_type=IntegerType(), required=False),
                    NestedField(field_id=6, name="name", field_type=StringType(), required=False),
                ),
                value_required=False,
            ),
            required=False,
        ),
    )

    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table_info = TableInfo(
        namespace="test",
        table_name="table",
        schema=schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
        data_files=[],
        data_dir="data",
        dataset_repo="test/repo",
        dataset_config="default",
        dataset_revision="abc123",
    )

    properties = table_info.get_table_properties()
    name_mapping = json.loads(properties["schema.name-mapping.default"])

    # Check map field
    metadata_mapping = name_mapping[1]
    assert metadata_mapping["field-id"] == 2
    assert metadata_mapping["names"] == ["metadata"]
    assert "fields" in metadata_mapping
    assert len(metadata_mapping["fields"]) == 2

    # Check key mapping
    key_mapping = metadata_mapping["fields"][0]
    assert key_mapping["field-id"] == 3
    assert key_mapping["names"] == ["key"]

    # Check value mapping
    value_mapping = metadata_mapping["fields"][1]
    assert value_mapping["field-id"] == 4
    assert value_mapping["names"] == ["value"]
    assert "fields" in value_mapping

    # Check struct fields within map value
    assert len(value_mapping["fields"]) == 2
    assert value_mapping["fields"][0]["field-id"] == 5
    assert value_mapping["fields"][0]["names"] == ["count"]
    assert value_mapping["fields"][1]["field-id"] == 6
    assert value_mapping["fields"][1]["names"] == ["name"]


# =============================================================================
# Revision Diff Tests
# =============================================================================


def test_discover_with_since_revision():
    """Test that passing since_revision to discover filters to new files only."""
    from unittest.mock import Mock, patch

    from datasets.features import Value

    # Mock dataset_builder_safe to return a mock builder
    mock_builder = Mock()
    mock_builder.hash = "def456"
    mock_builder.info.features = Features(
        {
            "text": Value("string"),
            "label": Value("int64"),
        }
    )
    mock_builder.config.data_dir = None
    mock_builder.config.data_files = {
        "train": [
            "hf://datasets/test/dataset@def456/plain_text/train-00000.parquet",
            "hf://datasets/test/dataset@def456/plain_text/train-00001.parquet",
        ],
        "test": ["hf://datasets/test/dataset@def456/plain_text/test-00000.parquet"],
    }

    # Mock HfApi.dataset_info to return dataset_info with siblings
    def create_sibling(rfilename, size):
        sibling = Mock()
        sibling.rfilename = rfilename
        sibling.size = size
        return sibling

    mock_dataset_info_current = Mock()
    mock_dataset_info_current.siblings = [
        create_sibling("plain_text/train-00000.parquet", 1000),
        create_sibling("plain_text/train-00001.parquet", 2000),
        create_sibling("plain_text/test-00000.parquet", 3000),
    ]

    mock_dataset_info_old = Mock()
    mock_dataset_info_old.siblings = [
        create_sibling("plain_text/train-00000.parquet", 1000),
    ]

    mock_api = Mock()

    def dataset_info_side_effect(repo_id, files_metadata=None, revision=None):
        if revision == "abc123":
            return mock_dataset_info_old
        else:
            return mock_dataset_info_current

    mock_api.dataset_info.side_effect = dataset_info_side_effect

    # Mock HfFileSystem to resolve file URIs
    mock_fs = Mock()

    def mock_resolve_path(uri):
        parts = uri.split("/")
        path = "/".join(parts[5:])
        mock_result = Mock()
        mock_result.path_in_repo = path
        return mock_result

    mock_fs.resolve_path.side_effect = mock_resolve_path

    with (
        patch("faceberg.bridge.dataset_builder_safe", return_value=mock_builder),
        patch("faceberg.bridge.HfApi", return_value=mock_api),
        patch("faceberg.bridge.HfFileSystem", return_value=mock_fs),
    ):
        # Discover with since_revision (should return only new files)
        dataset_info = DatasetInfo.discover(
            repo_id="test/dataset",
            config="plain_text",
            since_revision="abc123",
        )

    # Should have only 2 files (the new ones)
    assert len(dataset_info.splits) == 2
    assert "train" in dataset_info.splits
    assert "test" in dataset_info.splits

    # Verify data files are populated with new files (now FileInfo objects)
    assert "train" in dataset_info.data_files
    assert "test" in dataset_info.data_files
    train_files = dataset_info.data_files["train"]
    test_files = dataset_info.data_files["test"]
    assert len(train_files) == 1
    assert len(test_files) == 1
    assert all(isinstance(f, FileInfo) for f in train_files + test_files)
    assert train_files[0].uri == "hf://datasets/test/dataset@def456/plain_text/train-00001.parquet"
    assert test_files[0].uri == "hf://datasets/test/dataset@def456/plain_text/test-00000.parquet"
    # Check file sizes are populated
    assert train_files[0].size_bytes == 2000
    assert test_files[0].size_bytes == 3000

    # Now convert to TableInfo and verify
    table_info = dataset_info.to_table_info(
        namespace="default",
        table_name="test_table",
    )

    # Should have only 2 files (the new ones)
    assert len(table_info.data_files) == 2
    file_paths = [f.uri for f in table_info.data_files]
    assert "hf://datasets/test/dataset@def456/plain_text/train-00001.parquet" in file_paths
    assert "hf://datasets/test/dataset@def456/plain_text/test-00000.parquet" in file_paths

    # Verify files are properly organized by split
    splits = {f.split for f in table_info.data_files}
    assert "train" in splits
    assert "test" in splits


def test_features_stored_in_dataset_info():
    """Test that features are stored in DatasetInfo during discover()."""
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb", config="plain_text")

    # Features should be stored in DatasetInfo
    assert hasattr(dataset_info, "features")
    assert dataset_info.features is not None
    assert isinstance(dataset_info.features, Features)

    # Features should have expected fields for this dataset
    assert "text" in dataset_info.features
    assert "label" in dataset_info.features


def test_to_table_info_uses_stored_features():
    """Test that to_table_info uses stored features instead of calling dataset_builder_safe."""
    from unittest.mock import patch

    dataset_info = DatasetInfo.discover("stanfordnlp/imdb", config="plain_text")

    # Mock dataset_builder_safe to ensure it's NOT called
    with patch("faceberg.bridge.dataset_builder_safe") as mock_builder:
        table_info = dataset_info.to_table_info(
            namespace="default",
            table_name="imdb_plain_text",
        )

        # dataset_builder_safe should NOT have been called since features are stored
        mock_builder.assert_not_called()

    # TableInfo should still be created successfully
    assert table_info.schema is not None
    assert len(table_info.schema.fields) > 0


if __name__ == "__main__":
    # Run basic smoke test
    print("Running basic discovery test...")
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb", config="plain_text")
    print(f"✓ Discovered config: {dataset_info.config}")
    print(f"✓ Found splits: {dataset_info.splits}")

    # Count total parquet files
    total_files = sum(len(files) for files in dataset_info.data_files.values())
    print(f"✓ Found {total_files} Parquet files across {len(dataset_info.data_files)} splits")

    # Get a sample file
    first_split_files = next(iter(dataset_info.data_files.values()))
    if first_split_files:
        # Files are already fully qualified URIs
        sample = first_split_files[0]
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

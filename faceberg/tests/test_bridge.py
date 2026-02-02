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
    build_iceberg_schema_from_features,
    load_dataset_builder_safe,
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


def test_resolve_hf_path():
    """Test path resolution using HfFileSystem API."""
    from huggingface_hub import HfFileSystem

    from faceberg.bridge import resolve_hf_path

    fs = HfFileSystem()

    # Test with hf:// URI (using stanfordnlp/imdb which is stable and used in other tests)
    path1 = "hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet"
    result1 = resolve_hf_path(fs, path1)
    assert result1 == "plain_text/train-00000-of-00001.parquet"

    # Test with relative path (should return as-is)
    path2 = "plain_text/train-00000-of-00001.parquet"
    result2 = resolve_hf_path(fs, path2)
    assert result2 == "plain_text/train-00000-of-00001.parquet"


def test_to_table_infos():
    """Test converting DatasetInfo to TableInfo objects."""
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb", configs=["plain_text"])

    # Convert to TableInfo
    table_info = dataset_info.to_table_info(
        namespace="default",
        table_name="imdb_plain_text",
        config="plain_text",
    )

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
    assert props["huggingface.dataset.repo"] == "stanfordnlp/imdb"
    assert props["huggingface.dataset.config"] == "plain_text"


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


def test_load_dataset_builder_safe():
    """Test that the safe builder loader works and avoids local files."""
    # Test with a known public dataset
    builder = load_dataset_builder_safe("stanfordnlp/imdb", config_name="plain_text")

    assert builder is not None
    assert builder.info is not None
    assert builder.info.features is not None


def test_load_dataset_builder_safe_nonexistent():
    """Test that safe builder loader raises error for non-existent dataset."""
    with pytest.raises(Exception):
        load_dataset_builder_safe("nonexistent/fake-dataset-12345")


def test_to_table_info_without_features():
    """Test that to_table_info raises error if features are not available."""
    # Create a mock DatasetInfo with empty parquet_files
    dataset_info = DatasetInfo(
        repo_id="fake/dataset",
        configs=["default"],
        splits={"default": ["train"]},
        parquet_files={"default": {"train": []}},
        data_dirs={"default": "data"},
        revision=None,
    )

    # Mock scenario: dataset doesn't exist or has no features
    # load_dataset_builder_safe will raise an exception
    with pytest.raises(Exception):
        dataset_info.to_table_info(
            namespace="default",
            table_name="test_table",
            config="default",
        )


def test_table_properties_use_huggingface_prefix():
    """Test that table properties use huggingface.dataset.* prefix."""
    dataset_info = DatasetInfo.discover("stanfordnlp/imdb", configs=["plain_text"])
    table_info = dataset_info.to_table_info(
        namespace="default",
        table_name="imdb_plain_text",
        config="plain_text",
    )

    props = table_info.get_table_properties()

    # Check that properties use huggingface.dataset prefix
    assert "huggingface.dataset.repo" in props
    assert "huggingface.dataset.config" in props
    assert props["huggingface.dataset.repo"] == "stanfordnlp/imdb"
    assert props["huggingface.dataset.config"] == "plain_text"

    # Check that revision is included if available
    if table_info.source_revision:
        assert "huggingface.dataset.revision" in props
        assert props["huggingface.dataset.revision"] == table_info.source_revision

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
        files=[],
        data_dir="data",
        source_repo="test/repo",
        source_config="default",
        source_revision="abc123",
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
        files=[],
        data_dir="data",
        source_repo="test/repo",
        source_config="default",
        source_revision="abc123",
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
        files=[],
        data_dir="data",
        source_repo="test/repo",
        source_config="default",
        source_revision="abc123",
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


def test_extract_index_from_hf_pattern():
    """Test extracting index from HF-style filenames."""
    # Test data-NNNNN-of-NNNNN pattern
    assert DatasetInfo._extract_index_from_filename("data-00005-of-00010.parquet") == 5
    assert DatasetInfo._extract_index_from_filename("train-00000-of-00001.parquet") == 0

    # Test split-NNNNN-iceberg pattern
    assert DatasetInfo._extract_index_from_filename("train-00005-iceberg.parquet") == 5

    # Test simple split-NNNNN pattern
    assert DatasetInfo._extract_index_from_filename("train-00003.parquet") == 3

    # Test no index pattern
    assert DatasetInfo._extract_index_from_filename("random-file.parquet") is None


# =============================================================================
# Revision Diff Tests
# =============================================================================


def test_get_new_parquet_files_no_new_files():
    """Test when no files were added between revisions."""
    from unittest.mock import Mock, patch

    from faceberg.bridge import get_new_parquet_files

    # Mock HfApi
    mock_api = Mock()
    mock_api.list_repo_files.return_value = [
        "plain_text/train-00000.parquet",
        "plain_text/test-00000.parquet",
        "README.md",
    ]

    with patch("faceberg.bridge.HfApi", return_value=mock_api):
        result = get_new_parquet_files(
            repo_id="test/dataset",
            config="plain_text",
            old_revision="abc123",
            new_revision="def456",
        )

    # Should return empty list when files are the same
    assert result == []

    # Verify API was called with both revisions
    assert mock_api.list_repo_files.call_count == 2
    calls = mock_api.list_repo_files.call_args_list
    assert calls[0].kwargs["revision"] == "abc123"
    assert calls[1].kwargs["revision"] == "def456"


def test_get_new_parquet_files_with_new_files():
    """Test when new parquet files were added."""
    from unittest.mock import Mock, patch

    from faceberg.bridge import get_new_parquet_files

    # Mock HfApi
    mock_api = Mock()

    def list_files_side_effect(**kwargs):
        if kwargs["revision"] == "abc123":
            # Old revision has 2 files
            return [
                "plain_text/train-00000.parquet",
                "plain_text/test-00000.parquet",
                "README.md",
            ]
        else:
            # New revision has 4 files (2 new)
            return [
                "plain_text/train-00000.parquet",
                "plain_text/train-00001.parquet",  # NEW
                "plain_text/test-00000.parquet",
                "plain_text/validation-00000.parquet",  # NEW
                "README.md",
            ]

    mock_api.list_repo_files.side_effect = list_files_side_effect

    with patch("faceberg.bridge.HfApi", return_value=mock_api):
        result = get_new_parquet_files(
            repo_id="test/dataset",
            config="plain_text",
            old_revision="abc123",
            new_revision="def456",
        )

    # Should return only the 2 new parquet files in sorted order
    assert result == [
        "plain_text/train-00001.parquet",
        "plain_text/validation-00000.parquet",
    ]


def test_get_new_parquet_files_filters_by_config():
    """Test that only files for specified config are returned."""
    from unittest.mock import Mock, patch

    from faceberg.bridge import get_new_parquet_files

    # Mock HfApi
    mock_api = Mock()

    def list_files_side_effect(**kwargs):
        if kwargs["revision"] == "abc123":
            return ["README.md"]
        else:
            # New files in multiple configs
            return [
                "plain_text/train-00000.parquet",  # Should be included
                "other_config/train-00000.parquet",  # Should be excluded
                "README.md",
            ]

    mock_api.list_repo_files.side_effect = list_files_side_effect

    with patch("faceberg.bridge.HfApi", return_value=mock_api):
        result = get_new_parquet_files(
            repo_id="test/dataset",
            config="plain_text",
            old_revision="abc123",
            new_revision="def456",
        )

    # Should return only plain_text config files
    assert result == ["plain_text/train-00000.parquet"]


def test_get_new_parquet_files_ignores_non_parquet():
    """Test that non-parquet files are filtered out."""
    from unittest.mock import Mock, patch

    from faceberg.bridge import get_new_parquet_files

    # Mock HfApi
    mock_api = Mock()

    def list_files_side_effect(**kwargs):
        if kwargs["revision"] == "abc123":
            return []
        else:
            # Mix of file types
            return [
                "plain_text/train-00000.parquet",  # Should be included
                "plain_text/metadata.json",  # Should be excluded
                "plain_text/dataset_info.txt",  # Should be excluded
                "README.md",  # Should be excluded
            ]

    mock_api.list_repo_files.side_effect = list_files_side_effect

    with patch("faceberg.bridge.HfApi", return_value=mock_api):
        result = get_new_parquet_files(
            repo_id="test/dataset",
            config="plain_text",
            old_revision="abc123",
            new_revision="def456",
        )

    # Should return only parquet files
    assert result == ["plain_text/train-00000.parquet"]


def test_to_table_info_incremental_with_old_revision():
    """Test that passing old_revision filters to new files only."""
    from unittest.mock import Mock, patch

    from datasets.features import Value

    # Create a mock DatasetInfo
    dataset_info = DatasetInfo(
        repo_id="test/dataset",
        configs=["plain_text"],
        splits={"plain_text": ["train", "test"]},
        parquet_files={
            "plain_text": {
                "train": ["plain_text/train-00000.parquet", "plain_text/train-00001.parquet"],
                "test": ["plain_text/test-00000.parquet"],
            }
        },
        data_dirs={"plain_text": "plain_text"},
        revision="def456",
    )

    # Mock load_dataset_builder_safe with proper Features
    mock_builder = Mock()
    mock_builder.info.features = Features(
        {
            "text": Value("string"),
            "label": Value("int64"),
        }
    )

    # Mock get_new_parquet_files to return only 2 new files
    mock_get_new_files = Mock(
        return_value=[
            "plain_text/train-00001.parquet",  # New train file
            "plain_text/test-00000.parquet",  # New test file
        ]
    )

    with (
        patch("faceberg.bridge.load_dataset_builder_safe", return_value=mock_builder),
        patch("faceberg.bridge.get_new_parquet_files", mock_get_new_files),
    ):
        # Call with old_revision (should return only new files)
        table_info = dataset_info.to_table_info_incremental(
            namespace="default",
            table_name="test_table",
            config="plain_text",
            old_revision="abc123",
        )

    # Should have only 2 files (the new ones)
    assert len(table_info.files) == 2
    file_paths = [f.path for f in table_info.files]
    assert "hf://datasets/test/dataset/plain_text/train-00001.parquet" in file_paths
    assert "hf://datasets/test/dataset/plain_text/test-00000.parquet" in file_paths

    # Verify get_new_parquet_files was called with correct args
    mock_get_new_files.assert_called_once_with(
        repo_id="test/dataset",
        config="plain_text",
        old_revision="abc123",
        new_revision="def456",
        token=None,
    )

    # Verify files are properly organized by split
    splits = {f.split for f in table_info.files}
    assert "train" in splits
    assert "test" in splits


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

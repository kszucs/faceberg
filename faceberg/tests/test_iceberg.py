"""Tests for the iceberg module (Iceberg metadata generation)."""

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.manifest import ManifestEntryStatus
from pyiceberg.table import StaticTable
from pyiceberg.types import ListType, StructType

from faceberg.iceberg import ParquetFile, create_schema, diff_snapshot, write_snapshot


@pytest.fixture
def arrow_schema():
    """Create a simple PyArrow schema for testing."""
    return pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
            pa.field("value", pa.float64()),
        ]
    )


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of file contents."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


@pytest.fixture
def parquet_files(tmp_path, arrow_schema):
    """Create 5 parquet files with 20 rows each (100 total), each row unique."""
    files = []
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        path = data_dir / f"part-{i:05d}.parquet"

        # Each file has 20 unique rows
        start_id = i * 20
        table = pa.table(
            {
                "id": pa.array(list(range(start_id, start_id + 20)), type=pa.int64()),
                "name": [f"name_{j}" for j in range(start_id, start_id + 20)],
                "value": [float(j) * 1.5 for j in range(start_id, start_id + 20)],
            },
            schema=arrow_schema,
        )
        pq.write_table(table, path)
        files.append(
            ParquetFile(
                uri=str(path),
                path=str(path),
                size=path.stat().st_size,
                blob_id=compute_file_hash(path),
            )
        )

    return files


def make_extra_files(tmp_path, arrow_schema, count=2, start_index=5):
    """Create additional parquet files for append tests."""
    files = []
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        idx = start_index + i
        path = data_dir / f"part-{idx:05d}.parquet"

        start_id = idx * 20
        table = pa.table(
            {
                "id": pa.array(list(range(start_id, start_id + 20)), type=pa.int64()),
                "name": [f"name_{j}" for j in range(start_id, start_id + 20)],
                "value": [float(j) * 1.5 for j in range(start_id, start_id + 20)],
            },
            schema=arrow_schema,
        )
        pq.write_table(table, path)
        files.append(
            ParquetFile(
                uri=str(path),
                path=str(path),
                size=path.stat().st_size,
                blob_id=compute_file_hash(path),
            )
        )

    return files


class TestInitialSnapshot:
    """Tests for creating initial table snapshots."""

    def test_initial_snapshot_creates_valid_metadata(self, tmp_path, parquet_files, arrow_schema):
        """Test that initial snapshot creates valid Iceberg metadata."""
        _metadata = write_snapshot(
            files=parquet_files,
            schema=arrow_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        # Validate using StaticTable - pass the metadata file path
        metadata_file = tmp_path / "metadata" / "v1.metadata.json"
        table = StaticTable.from_metadata(str(metadata_file))

        # Check data files
        data_files = table.inspect.data_files()
        assert len(data_files) == 5

        # Check file paths match
        file_paths = set(data_files["file_path"].to_pylist())
        expected_paths = {f.uri for f in parquet_files}
        assert file_paths == expected_paths

        # Check snapshot summary
        snapshot = table.current_snapshot()
        assert snapshot is not None
        assert snapshot.summary.operation.value == "append"
        assert int(snapshot.summary["added-data-files"]) == 5
        assert int(snapshot.summary["total-records"]) == 100

    def test_initial_snapshot_scan_returns_data(self, tmp_path, parquet_files, arrow_schema):
        """Test that initial snapshot can be scanned correctly."""

        write_snapshot(
            files=parquet_files,
            schema=arrow_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        metadata_file = tmp_path / "metadata" / "v1.metadata.json"
        table = StaticTable.from_metadata(str(metadata_file))
        result = table.scan().to_arrow()

        # Should have 100 rows (5 files * 20 rows)
        assert len(result) == 100

        # Check all IDs are present (0-99)
        ids = sorted(result["id"].to_pylist())
        assert ids == list(range(100))


class TestAppendSnapshot:
    """Tests for appending files to existing snapshots."""

    def test_append_files_creates_new_snapshot(self, tmp_path, parquet_files, arrow_schema):
        """Test that appending files creates a new snapshot with all files."""

        # Create initial snapshot
        metadata = write_snapshot(
            files=parquet_files,
            schema=arrow_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        # Create additional files
        extra_files = make_extra_files(tmp_path, arrow_schema, count=2, start_index=5)

        # Append files
        updated_metadata = write_snapshot(
            files=parquet_files + extra_files,
            schema=arrow_schema,
            current_metadata=metadata,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        # Validate
        metadata_file = tmp_path / "metadata" / f"v{len(updated_metadata.snapshots)}.metadata.json"
        table = StaticTable.from_metadata(str(metadata_file))

        # Should have 7 files now
        data_files = table.inspect.data_files()
        assert len(data_files) == 7

        # Should have 2 snapshots
        snapshots = table.inspect.snapshots()
        assert len(snapshots) == 2

        # Scan should return 140 rows
        result = table.scan().to_arrow()
        assert len(result) == 140

        # Check IDs 0-139 are present
        ids = sorted(result["id"].to_pylist())
        assert ids == list(range(140))


class TestDeleteSnapshot:
    """Tests for deleting files from snapshots."""

    def test_delete_files_removes_from_snapshot(self, tmp_path, parquet_files, arrow_schema):
        """Test that deleting files removes them from the current snapshot."""

        # Create initial snapshot
        metadata = write_snapshot(
            files=parquet_files,
            schema=arrow_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        # Delete first 2 files (IDs 0-39) by passing only the remaining files
        remaining_files = parquet_files[2:]

        updated_metadata = write_snapshot(
            files=remaining_files,
            schema=arrow_schema,
            current_metadata=metadata,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        # Validate
        metadata_file = tmp_path / "metadata" / f"v{len(updated_metadata.snapshots)}.metadata.json"
        table = StaticTable.from_metadata(str(metadata_file))

        # Should have 3 files now
        data_files = table.inspect.data_files()
        assert len(data_files) == 3

        # Deleted files should not be present
        file_paths = set(data_files["file_path"].to_pylist())
        for deleted in parquet_files[:2]:
            assert deleted.uri not in file_paths

        # Scan should return 60 rows (IDs 40-99)
        result = table.scan().to_arrow()
        assert len(result) == 60

        ids = sorted(result["id"].to_pylist())
        assert ids == list(range(40, 100))


class TestOverwriteSnapshot:
    """Tests for overwrite operations (delete + add)."""

    def test_overwrite_replaces_files(self, tmp_path, parquet_files, arrow_schema):
        """Test that overwrite removes old files and adds new ones."""

        # Create initial snapshot
        metadata = write_snapshot(
            files=parquet_files,
            schema=arrow_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        # Create a replacement file
        replacement_path = tmp_path / "data" / "replacement.parquet"
        replacement_table = pa.table(
            {
                "id": pa.array(list(range(1000, 1020)), type=pa.int64()),
                "name": [f"replaced_{j}" for j in range(20)],
                "value": [float(j) * 2.0 for j in range(20)],
            },
            schema=arrow_schema,
        )
        pq.write_table(replacement_table, replacement_path)
        replacement_file = ParquetFile(
            uri=str(replacement_path),
            path=str(replacement_path),
            size=replacement_path.stat().st_size,
            blob_id=compute_file_hash(replacement_path),
        )

        # Overwrite: replace first file with replacement
        updated_metadata = write_snapshot(
            files=[replacement_file] + parquet_files[1:],
            schema=arrow_schema,
            current_metadata=metadata,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        # Validate
        metadata_file = tmp_path / "metadata" / f"v{len(updated_metadata.snapshots)}.metadata.json"
        table = StaticTable.from_metadata(str(metadata_file))

        # Still 5 files
        data_files = table.inspect.data_files()
        assert len(data_files) == 5

        # Old file should be gone, replacement should be present
        file_paths = set(data_files["file_path"].to_pylist())
        assert parquet_files[0].uri not in file_paths
        assert str(replacement_path) in file_paths

        # Snapshot should be OVERWRITE
        snapshot = table.current_snapshot()
        assert snapshot.summary.operation.value == "overwrite"

        # Scan should return 100 rows (20-99 from original + 1000-1019 from replacement)
        result = table.scan().to_arrow()
        assert len(result) == 100


class TestRenameFile:
    """Tests for file rename operations (delete old URI + add new URI)."""

    def test_rename_file_updates_uri(self, tmp_path, parquet_files, arrow_schema):
        """Test renaming a file (delete old URI + add new URI with same content)."""

        # Create initial snapshot
        metadata = write_snapshot(
            files=parquet_files,
            schema=arrow_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        # "Rename" first file: copy to new location
        old_file = parquet_files[0]
        new_path = tmp_path / "data" / "renamed-file.parquet"
        shutil.copy(old_file.uri, new_path)
        new_file = ParquetFile(
            uri=str(new_path),
            path=str(new_path),
            size=new_path.stat().st_size,
            blob_id=compute_file_hash(new_path),
        )

        # Create overwrite snapshot with renamed file
        updated_metadata = write_snapshot(
            files=[new_file] + parquet_files[1:],
            schema=arrow_schema,
            current_metadata=metadata,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        # Validate
        metadata_file = tmp_path / "metadata" / f"v{len(updated_metadata.snapshots)}.metadata.json"
        table = StaticTable.from_metadata(str(metadata_file))
        data_files = table.inspect.data_files()
        file_paths = set(data_files["file_path"].to_pylist())

        # Old file should not be present
        assert old_file.uri not in file_paths
        # New file should be present
        assert str(new_path) in file_paths
        # Total files still 5
        assert len(data_files) == 5

        # Data should be unchanged (100 rows with same IDs)
        result = table.scan().to_arrow()
        assert len(result) == 100

        ids = sorted(result["id"].to_pylist())
        assert ids == list(range(100))


class TestManifestEntries:
    """Tests for manifest entry correctness."""

    def test_initial_entries_are_added(self, tmp_path, parquet_files, arrow_schema):
        """Test that initial snapshot entries have ADDED status."""

        write_snapshot(
            files=parquet_files,
            schema=arrow_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        metadata_file = tmp_path / "metadata" / "v1.metadata.json"
        table = StaticTable.from_metadata(str(metadata_file))
        entries = table.inspect.entries()

        # All entries should be ADDED (status=1)
        statuses = entries["status"].to_pylist()
        assert all(s == 1 for s in statuses)
        assert len(statuses) == 5

    def test_append_entries_are_added(self, tmp_path, parquet_files, arrow_schema):
        """Test that appended files have ADDED status in new manifest."""

        metadata = write_snapshot(
            files=parquet_files,
            schema=arrow_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        extra_files = make_extra_files(tmp_path, arrow_schema, count=1, start_index=5)

        updated_metadata = write_snapshot(
            files=parquet_files + extra_files,
            schema=arrow_schema,
            current_metadata=metadata,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        metadata_file = tmp_path / "metadata" / f"v{len(updated_metadata.snapshots)}.metadata.json"
        table = StaticTable.from_metadata(str(metadata_file))
        entries = table.inspect.entries()

        # Should have 6 entries total
        assert len(entries) == 6

        # All entries visible should be ADDED (1) or EXISTING (0)
        statuses = entries["status"].to_pylist()
        assert all(s in (0, 1) for s in statuses)


class TestDiffSnapshotFiles:
    """Tests for diff_snapshot function."""

    def test_initial_snapshot_all_added(self, tmp_path, parquet_files, arrow_schema):
        """Test that with no previous metadata, all files are ADDED."""

        io = PyArrowFileIO()
        result = diff_snapshot(parquet_files, None, io)

        # All files should be ADDED
        assert len(result) == 5
        for status, pf in result:
            assert status == ManifestEntryStatus.ADDED
            assert pf in parquet_files

    def test_existing_files_unchanged(self, tmp_path, parquet_files, arrow_schema):
        """Test that files unchanged from previous snapshot are EXISTING."""

        # Create initial snapshot
        metadata = write_snapshot(
            files=parquet_files,
            schema=arrow_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        # Diff with same files
        io = PyArrowFileIO()
        result = diff_snapshot(parquet_files, metadata, io)

        # All files should be EXISTING
        assert len(result) == 5
        for status, pf in result:
            assert status == ManifestEntryStatus.EXISTING
            assert pf in parquet_files

    def test_removed_files(self, tmp_path, parquet_files, arrow_schema):
        """Test that files in previous snapshot but not in current are REMOVED."""

        # Create initial snapshot
        metadata = write_snapshot(
            files=parquet_files,
            schema=arrow_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        # Diff with subset of files (remove first 2)
        io = PyArrowFileIO()
        current_files = parquet_files[2:]  # Keep only last 3 files
        result = diff_snapshot(current_files, metadata, io)

        # Should have 3 EXISTING + 2 REMOVED = 5 total
        assert len(result) == 5

        existing_count = sum(1 for status, _ in result if status == ManifestEntryStatus.EXISTING)
        removed_count = sum(1 for status, _ in result if status == ManifestEntryStatus.DELETED)

        assert existing_count == 3
        assert removed_count == 2

        # Check that removed files are the first 2
        removed_files = [pf for status, pf in result if status == ManifestEntryStatus.DELETED]
        assert len(removed_files) == 2
        for pf in removed_files:
            assert pf.uri in [parquet_files[0].uri, parquet_files[1].uri]

    def test_changed_files_removed_and_added(self, tmp_path, parquet_files, arrow_schema):
        """Test that files with same URI but different hash/size are REMOVED + ADDED."""

        # Create initial snapshot
        metadata = write_snapshot(
            files=parquet_files,
            schema=arrow_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
        )

        # Modify first file (same URI, different content)
        first_file_path = Path(parquet_files[0].uri)
        modified_table = pa.table(
            {
                "id": pa.array([999], type=pa.int64()),
                "name": ["modified"],
                "value": [999.9],
            },
            schema=arrow_schema,
        )
        pq.write_table(modified_table, first_file_path)

        # Create new ParquetFile with same URI but new hash
        modified_file = ParquetFile(
            uri=str(first_file_path),
            path=str(first_file_path),
            size=first_file_path.stat().st_size,
            blob_id=compute_file_hash(first_file_path),
        )

        current_files = [modified_file] + parquet_files[1:]

        # Diff
        io = PyArrowFileIO()
        result = diff_snapshot(current_files, metadata, io)

        # Should have: 1 REMOVED (old version) + 1 ADDED (new version) + 4 EXISTING = 6 total
        assert len(result) == 6

        added_count = sum(1 for status, _ in result if status == ManifestEntryStatus.ADDED)
        removed_count = sum(1 for status, _ in result if status == ManifestEntryStatus.DELETED)
        existing_count = sum(1 for status, _ in result if status == ManifestEntryStatus.EXISTING)

        assert added_count == 1
        assert removed_count == 1
        assert existing_count == 4


class TestSchemaConversion:
    """Tests for create_schema with complex nested structures."""

    def test_schema_with_nested_struct(self):
        """Test schema conversion with nested struct fields."""
        # Create PyArrow schema with nested struct
        arrow_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field(
                    "metadata",
                    pa.struct(
                        [
                            pa.field("title", pa.string()),
                            pa.field("author", pa.string()),
                            pa.field("year", pa.int32()),
                        ]
                    ),
                ),
            ]
        )

        schema = create_schema(arrow_schema, include_split_column=False)

        # Verify structure
        field_names = [f.name for f in schema.fields]
        assert "id" in field_names
        assert "metadata" in field_names

        # Find metadata field
        metadata_field = next(f for f in schema.fields if f.name == "metadata")
        assert isinstance(metadata_field.field_type, StructType)

        # Verify nested fields
        nested_field_names = [f.name for f in metadata_field.field_type.fields]
        assert "title" in nested_field_names
        assert "author" in nested_field_names
        assert "year" in nested_field_names

    def test_schema_with_list_field(self):
        """Test schema conversion with list fields."""
        arrow_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("tags", pa.list_(pa.string())),
            ]
        )

        schema = create_schema(arrow_schema, include_split_column=False)

        # Find tags field
        tags_field = next(f for f in schema.fields if f.name == "tags")
        assert isinstance(tags_field.field_type, ListType)

    def test_schema_with_deeply_nested_structures(self):
        """Test schema conversion with deeply nested structures."""
        arrow_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field(
                    "nested",
                    pa.struct(
                        [
                            pa.field("field1", pa.string()),
                            pa.field("field2", pa.int32()),
                            pa.field(
                                "deeper",
                                pa.struct([pa.field("field3", pa.string())]),
                            ),
                        ]
                    ),
                ),
                pa.field("list_field", pa.list_(pa.string())),
            ]
        )

        schema = create_schema(arrow_schema, include_split_column=True)

        # Should include split column
        field_names = [f.name for f in schema.fields]
        assert "split" in field_names
        assert schema.fields[0].name == "split"

        # Verify nested field exists
        nested_field = next(f for f in schema.fields if f.name == "nested")
        assert isinstance(nested_field.field_type, StructType)

    def test_unique_field_ids_across_nested_structures(self):
        """Test that all field IDs are unique across nested structures."""
        arrow_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field(
                    "nested",
                    pa.struct(
                        [
                            pa.field("field1", pa.string()),
                            pa.field("field2", pa.int32()),
                            pa.field(
                                "deeper",
                                pa.struct([pa.field("field3", pa.string())]),
                            ),
                        ]
                    ),
                ),
                pa.field("list_field", pa.list_(pa.string())),
            ]
        )

        schema = create_schema(arrow_schema, include_split_column=True)

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


class TestNameMapping:
    """Tests for name mapping with nested structures."""

    def test_name_mapping_with_nested_structs(self, tmp_path):
        """Test that name mapping includes nested struct fields."""
        # Create schema with nested structs
        iceberg_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field(
                    "metadata",
                    pa.struct(
                        [
                            pa.field("author", pa.string()),
                            pa.field("year", pa.int32()),
                        ]
                    ),
                ),
            ]
        )

        # Create a test parquet file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        file_path = data_dir / "test.parquet"
        table = pa.table(
            {
                "id": ["1", "2"],
                "metadata": [
                    {"author": "Alice", "year": 2020},
                    {"author": "Bob", "year": 2021},
                ],
            },
            schema=iceberg_schema,
        )
        pq.write_table(table, file_path)

        files = [
            ParquetFile(
                uri=str(file_path),
                path=str(file_path),
                size=file_path.stat().st_size,
                blob_id="test",
            )
        ]

        # Write snapshot with schema
        metadata = write_snapshot(
            files=files,
            schema=iceberg_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
            include_split_column=False,
        )

        # Get name mapping from properties
        name_mapping = json.loads(metadata.properties["schema.name-mapping.default"])

        # Check top-level fields
        assert len(name_mapping) == 2
        assert name_mapping[0]["names"] == ["id"]
        assert name_mapping[1]["names"] == ["metadata"]

        # Check nested struct field
        metadata_mapping = name_mapping[1]
        assert "fields" in metadata_mapping
        assert len(metadata_mapping["fields"]) == 2

        # Check nested struct's child fields
        assert metadata_mapping["fields"][0]["names"] == ["author"]
        assert metadata_mapping["fields"][1]["names"] == ["year"]

    def test_name_mapping_with_lists(self, tmp_path):
        """Test that name mapping includes list element mappings."""
        # Create schema with list of strings and list of structs
        iceberg_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field(
                    "items",
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("name", pa.string()),
                                pa.field("value", pa.string()),
                            ]
                        )
                    ),
                ),
            ]
        )

        # Create test parquet file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        file_path = data_dir / "test.parquet"
        table = pa.table(
            {
                "id": ["1"],
                "tags": [["tag1", "tag2"]],
                "items": [[{"name": "item1", "value": "val1"}]],
            },
            schema=iceberg_schema,
        )
        pq.write_table(table, file_path)

        files = [
            ParquetFile(
                uri=str(file_path),
                path=str(file_path),
                size=file_path.stat().st_size,
                blob_id="test",
            )
        ]

        metadata = write_snapshot(
            files=files,
            schema=iceberg_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
            include_split_column=False,
        )

        name_mapping = json.loads(metadata.properties["schema.name-mapping.default"])

        # Check list of strings (tags)
        tags_mapping = name_mapping[1]
        assert tags_mapping["names"] == ["tags"]
        assert "fields" in tags_mapping
        assert len(tags_mapping["fields"]) == 1

        # Check element mapping for simple list
        element_mapping = tags_mapping["fields"][0]
        assert element_mapping["names"] == ["element"]

        # Check list of structs (items)
        items_mapping = name_mapping[2]
        assert items_mapping["names"] == ["items"]
        assert "fields" in items_mapping

        # Check element mapping for list of structs
        items_element = items_mapping["fields"][0]
        assert items_element["names"] == ["element"]
        assert "fields" in items_element

        # Check struct fields within list element
        assert len(items_element["fields"]) == 2
        assert items_element["fields"][0]["names"] == ["name"]
        assert items_element["fields"][1]["names"] == ["value"]

    def test_name_mapping_with_maps(self, tmp_path):
        """Test that name mapping includes map key and value mappings."""
        # Create schema with a map
        iceberg_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field(
                    "metadata",
                    pa.map_(
                        pa.string(),
                        pa.struct(
                            [
                                pa.field("count", pa.int32()),
                                pa.field("name", pa.string()),
                            ]
                        ),
                    ),
                ),
            ]
        )

        # Create test parquet file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        file_path = data_dir / "test.parquet"
        table = pa.table(
            {
                "id": ["1"],
                "metadata": [[("key1", {"count": 1, "name": "name1"})]],
            },
            schema=iceberg_schema,
        )
        pq.write_table(table, file_path)

        files = [
            ParquetFile(
                uri=str(file_path),
                path=str(file_path),
                size=file_path.stat().st_size,
                blob_id="test",
            )
        ]

        metadata = write_snapshot(
            files=files,
            schema=iceberg_schema,
            current_metadata=None,
            output_dir=tmp_path,
            base_uri=f"file://{tmp_path}",
            include_split_column=False,
        )

        name_mapping = json.loads(metadata.properties["schema.name-mapping.default"])

        # Check map field
        metadata_mapping = name_mapping[1]
        assert metadata_mapping["names"] == ["metadata"]
        assert "fields" in metadata_mapping
        assert len(metadata_mapping["fields"]) == 2

        # Check key mapping
        key_mapping = metadata_mapping["fields"][0]
        assert key_mapping["names"] == ["key"]

        # Check value mapping
        value_mapping = metadata_mapping["fields"][1]
        assert value_mapping["names"] == ["value"]
        assert "fields" in value_mapping

        # Check struct fields within map value
        assert len(value_mapping["fields"]) == 2
        assert value_mapping["fields"][0]["names"] == ["count"]
        assert value_mapping["fields"][1]["names"] == ["name"]

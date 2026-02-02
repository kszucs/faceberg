"""Tests for the convert module (Iceberg metadata generation)."""

from unittest.mock import Mock, patch

import pytest
from pyiceberg.schema import Schema
from pyiceberg.types import IntegerType, NestedField, StringType

from faceberg.bridge import FileInfo
from faceberg.convert import IcebergMetadataWriter


@pytest.fixture
def temp_table_path(tmp_path):
    """Create a temporary table path for testing."""
    table_path = tmp_path / "test_namespace" / "test_table"
    table_path.mkdir(parents=True, exist_ok=True)
    return table_path


@pytest.fixture
def simple_schema():
    """Create a simple Iceberg schema for testing."""
    return Schema(
        NestedField(field_id=1, name="id", field_type=StringType(), required=False),
        NestedField(field_id=2, name="value", field_type=IntegerType(), required=False),
    )


@pytest.fixture
def metadata_writer(temp_table_path, simple_schema):
    """Create a metadata writer instance for testing."""
    # Construct file:// URI for the temp path
    path_str = temp_table_path.absolute().as_posix()
    base_uri = f"file:///{path_str.lstrip('/')}"
    return IcebergMetadataWriter(
        table_path=temp_table_path, schema=simple_schema, base_uri=base_uri
    )


class TestGetHfFileSize:
    """Tests for the _get_hf_file_size method."""

    def test_get_hf_file_size_valid_url(self, metadata_writer):
        """Test getting file size from a valid HuggingFace URL."""
        test_url = "hf://datasets/deepmind/narrativeqa/data/train-00000-of-00024.parquet"

        with (
            patch("faceberg.convert.hf_hub_url") as mock_hf_hub_url,
            patch("faceberg.convert.get_hf_file_metadata") as mock_get_metadata,
        ):
            # Setup mocks
            mock_hf_hub_url.return_value = "https://huggingface.co/mock-url"

            mock_metadata = Mock()
            mock_metadata.size = 9799947
            mock_get_metadata.return_value = mock_metadata

            # Test
            file_size = metadata_writer._get_hf_file_size(test_url)

            # Verify
            assert file_size == 9799947
            mock_hf_hub_url.assert_called_once_with(
                repo_id="deepmind/narrativeqa",
                filename="data/train-00000-of-00024.parquet",
                repo_type="dataset",
                revision=None,
            )
            mock_get_metadata.assert_called_once()

    def test_get_hf_file_size_nested_path(self, metadata_writer):
        """Test getting file size from a URL with deeply nested path."""
        test_url = "hf://datasets/org/repo/path/to/deep/file.parquet"

        with (
            patch("faceberg.convert.hf_hub_url") as mock_hf_hub_url,
            patch("faceberg.convert.get_hf_file_metadata") as mock_get_metadata,
        ):
            mock_hf_hub_url.return_value = "https://mock.url"
            mock_metadata = Mock()
            mock_metadata.size = 12345678
            mock_get_metadata.return_value = mock_metadata

            file_size = metadata_writer._get_hf_file_size(test_url)

            assert file_size == 12345678
            mock_hf_hub_url.assert_called_once_with(
                repo_id="org/repo",
                filename="path/to/deep/file.parquet",
                repo_type="dataset",
                revision=None,
            )

    def test_get_hf_file_size_invalid_url_format(self, metadata_writer):
        """Test handling of invalid URL format."""
        import pytest

        # Not a hf:// URL
        with pytest.raises(ValueError, match="Invalid HuggingFace file path"):
            metadata_writer._get_hf_file_size("s3://bucket/file.parquet")

        # Invalid hf:// URL (too few parts)
        with pytest.raises(ValueError, match="Invalid HuggingFace file path format"):
            metadata_writer._get_hf_file_size("hf://datasets/repo")

    def test_get_hf_file_size_api_error(self, metadata_writer):
        """Test handling of HuggingFace API errors."""
        import pytest

        test_url = "hf://datasets/org/repo/file.parquet"

        with (
            patch("faceberg.convert.hf_hub_url") as mock_hf_hub_url,
            patch("faceberg.convert.get_hf_file_metadata") as mock_get_metadata,
        ):
            mock_hf_hub_url.return_value = "https://mock.url"
            # Simulate API error
            mock_get_metadata.side_effect = Exception("API Error")

            # Should raise the API error (fail-fast behavior)
            with pytest.raises(Exception, match="API Error"):
                metadata_writer._get_hf_file_size(test_url)


class TestReadFileMetadata:
    """Tests for the _read_file_metadata method."""

    def test_read_file_metadata_gets_file_size(self, metadata_writer):
        """Test that _read_file_metadata gets file size from HuggingFace when size_bytes is 0."""
        file_infos = [
            FileInfo(
                uri="hf://datasets/org/repo/file1.parquet",
                size_bytes=0,  # No size provided
                row_count=0,
                split="train",
            )
        ]

        with (
            patch("faceberg.convert.pq.read_metadata") as mock_read_metadata,
            patch.object(metadata_writer, "_get_hf_file_size") as mock_get_size,
        ):
            # Mock parquet metadata
            mock_metadata = Mock()
            mock_metadata.num_rows = 1000
            mock_read_metadata.return_value = mock_metadata

            # Mock file size from HuggingFace
            mock_get_size.return_value = 9876543

            # Test
            enriched = metadata_writer._read_file_metadata(file_infos)

            # Verify
            assert len(enriched) == 1
            assert enriched[0].uri == "hf://datasets/org/repo/file1.parquet"
            assert enriched[0].size_bytes == 9876543
            assert enriched[0].row_count == 1000
            mock_get_size.assert_called_once_with("hf://datasets/org/repo/file1.parquet")

    def test_read_file_metadata_preserves_provided_size(self, metadata_writer):
        """Test that _read_file_metadata preserves size_bytes when already provided."""
        file_infos = [
            FileInfo(
                uri="hf://datasets/org/repo/file1.parquet",
                size_bytes=5555555,  # Size already provided
                row_count=0,
                split="train",
            )
        ]

        with (
            patch("faceberg.convert.pq.read_metadata") as mock_read_metadata,
            patch.object(metadata_writer, "_get_hf_file_size") as mock_get_size,
        ):
            mock_metadata = Mock()
            mock_metadata.num_rows = 1000
            mock_read_metadata.return_value = mock_metadata

            enriched = metadata_writer._read_file_metadata(file_infos)

            # Should use provided size, not call _get_hf_file_size
            assert enriched[0].size_bytes == 5555555
            mock_get_size.assert_not_called()

    def test_read_file_metadata_multiple_files(self, metadata_writer):
        """Test enriching metadata for multiple files."""
        file_infos = [
            FileInfo(
                uri="hf://datasets/org/repo/file1.parquet",
                size_bytes=0,
                row_count=0,
                split="train",
            ),
            FileInfo(
                uri="hf://datasets/org/repo/file2.parquet",
                size_bytes=0,
                row_count=0,
                split="train",
            ),
            FileInfo(
                uri="hf://datasets/org/repo/file3.parquet",
                size_bytes=123456,  # Already has size
                row_count=500,  # This will be overwritten by reading parquet metadata
                split="test",
            ),
        ]

        with (
            patch("faceberg.convert.pq.read_metadata") as mock_read_metadata,
            patch.object(metadata_writer, "_get_hf_file_size") as mock_get_size,
        ):
            # Mock parquet metadata - return different row counts for each file
            def get_metadata_side_effect(path):
                mock_metadata = Mock()
                if "file1" in path:
                    mock_metadata.num_rows = 1000
                elif "file2" in path:
                    mock_metadata.num_rows = 2000
                else:  # file3
                    mock_metadata.num_rows = 3000
                return mock_metadata

            mock_read_metadata.side_effect = get_metadata_side_effect

            # Mock file sizes from HuggingFace for files without size
            mock_get_size.side_effect = [9999999, 8888888]

            enriched = metadata_writer._read_file_metadata(file_infos)

            assert len(enriched) == 3
            # File 1: no size, gets it from HuggingFace
            assert enriched[0].size_bytes == 9999999
            assert enriched[0].row_count == 1000
            # File 2: no size, gets it from HuggingFace
            assert enriched[1].size_bytes == 8888888
            assert enriched[1].row_count == 2000
            # File 3: has size, uses it, but row_count is read from parquet metadata
            assert enriched[2].size_bytes == 123456
            assert enriched[2].row_count == 3000  # Overwritten by parquet metadata
            # Should only call _get_hf_file_size for first two files (file3 has size)
            assert mock_get_size.call_count == 2

    def test_read_file_metadata_handles_read_error(self, metadata_writer):
        """Test that metadata read errors are raised (fail-fast behavior)."""
        import pytest

        file_infos = [
            FileInfo(
                uri="hf://datasets/org/repo/file1.parquet",
                size_bytes=0,
                row_count=0,
                split="train",
            )
        ]

        with patch("faceberg.convert.pq.read_metadata") as mock_read_metadata:
            # Simulate read error
            mock_read_metadata.side_effect = Exception("Cannot read metadata")

            # Should raise the error (fail-fast behavior)
            with pytest.raises(Exception, match="Cannot read metadata"):
                metadata_writer._read_file_metadata(file_infos)


class TestFileSizeRegression:
    """Regression tests to ensure the bug fix works correctly."""

    def test_file_size_not_using_serialized_size(self, metadata_writer):
        """Regression test: ensure we don't use metadata.serialized_size (the original bug)."""
        # This is the key regression test for the bug fix
        file_infos = [
            FileInfo(
                uri="hf://datasets/deepmind/narrativeqa/data/train-00000-of-00024.parquet",
                size_bytes=0,
                row_count=0,
                split="train",
            )
        ]

        with (
            patch("faceberg.convert.pq.read_metadata") as mock_read_metadata,
            patch.object(metadata_writer, "_get_hf_file_size") as mock_get_size,
        ):
            # The bug was using metadata.serialized_size which is ~500 bytes
            mock_metadata = Mock()
            mock_metadata.num_rows = 1365
            mock_metadata.serialized_size = 550  # This is the WRONG value that was used before

            mock_read_metadata.return_value = mock_metadata

            # The correct file size from HuggingFace API
            mock_get_size.return_value = 9799947

            enriched = metadata_writer._read_file_metadata(file_infos)

            # Verify we're using the correct file size, not serialized_size
            assert enriched[0].size_bytes == 9799947
            assert enriched[0].size_bytes != 550
            # The ratio should be reasonable (actual size vs metadata footer size)
            assert enriched[0].size_bytes / mock_metadata.serialized_size > 1000

    def test_file_sizes_match_real_world_ratios(self, metadata_writer):
        """Test that file sizes match expected ratios from real-world HuggingFace datasets."""
        # From the bug report, we saw ratios of 500-19000x between actual and serialized_size
        file_infos = [
            FileInfo(
                uri=f"hf://datasets/deepmind/narrativeqa/data/train-{i:05d}-of-00024.parquet",
                size_bytes=0,
                row_count=0,
                split="train",
            )
            for i in range(5)
        ]

        # Real-world compressed data sizes (excluding footer)
        compressed_sizes = [9766702, 67176993, 232523620, 27221729, 88315563]
        # Typical metadata.serialized_size values (footer size)
        serialized_sizes = [18853, 10532, 11971, 9938, 19011]

        with (
            patch("faceberg.convert.pq.read_metadata") as mock_read_metadata,
            patch.object(metadata_writer, "_get_hf_file_size") as mock_get_size,
        ):

            def get_metadata_side_effect(path):
                idx = int(path.split("train-")[1].split("-of")[0])
                mock_metadata = Mock()
                mock_metadata.num_rows = 1000
                mock_metadata.serialized_size = serialized_sizes[idx]
                mock_metadata.num_row_groups = 1

                # Mock row group with single column containing all compressed data
                mock_rg = Mock()
                mock_rg.num_columns = 1
                mock_col = Mock(total_compressed_size=compressed_sizes[idx])
                mock_rg.column = Mock(return_value=mock_col)
                mock_metadata.row_group = Mock(return_value=mock_rg)

                return mock_metadata

            mock_read_metadata.side_effect = get_metadata_side_effect
            # Mock the file size to return calculated size (compressed + footer + 8)
            mock_get_size.side_effect = lambda path: (
                compressed_sizes[int(path.split("train-")[1].split("-of")[0])]
                + serialized_sizes[int(path.split("train-")[1].split("-of")[0])]
                + 8
            )

            enriched = metadata_writer._read_file_metadata(file_infos)

            # Verify all files have correct sizes (compressed + footer + 8 bytes)
            for i, file_info in enumerate(enriched):
                expected = compressed_sizes[i] + serialized_sizes[i] + 8
                assert file_info.size_bytes == expected
                # Verify we're not using just the footer
                assert file_info.size_bytes != serialized_sizes[i]
                # Verify the ratio is in the expected range
                ratio = file_info.size_bytes / serialized_sizes[i]
                assert 500 <= ratio <= 20000  # Based on real-world observations


class TestGetPreviousManifests:
    """Tests for the _get_previous_manifests method for fast append optimization."""

    def test_no_snapshots_returns_none(self, metadata_writer):
        """Test that None is returned when metadata has no snapshots."""
        from pyiceberg.table.metadata import TableMetadataV2

        # Create metadata with no snapshots
        metadata = Mock(spec=TableMetadataV2)
        metadata.current_snapshot_id = None
        metadata.snapshots = []

        result = metadata_writer._get_previous_manifests(metadata)
        assert result is None

    def test_returns_manifest_files_without_reading_contents(self, metadata_writer):
        """Test that ManifestFile objects are returned without fetching their entries."""
        from pyiceberg.manifest import ManifestFile
        from pyiceberg.table.metadata import TableMetadataV2
        from pyiceberg.table.snapshots import Snapshot

        # Create mock manifest files
        mock_manifest_1 = Mock(spec=ManifestFile)
        mock_manifest_1.manifest_path = "hf://datasets/org/repo/metadata/manifest1.avro"

        mock_manifest_2 = Mock(spec=ManifestFile)
        mock_manifest_2.manifest_path = "hf://datasets/org/repo/metadata/manifest2.avro"

        # Create mock snapshot
        mock_snapshot = Mock(spec=Snapshot)
        mock_snapshot.snapshot_id = 1
        mock_snapshot.manifests.return_value = [mock_manifest_1, mock_manifest_2]

        # Create metadata
        metadata = Mock(spec=TableMetadataV2)
        metadata.current_snapshot_id = 1
        metadata.snapshots = [mock_snapshot]

        # Test
        result = metadata_writer._get_previous_manifests(metadata)

        # Verify - should return manifest files
        assert result is not None
        assert len(result) == 2
        assert result[0] == mock_manifest_1
        assert result[1] == mock_manifest_2

        # Critical: verify we did NOT call fetch_manifest_entry (no content reading)
        assert (
            not hasattr(mock_manifest_1, "fetch_manifest_entry")
            or not mock_manifest_1.fetch_manifest_entry.called
        )
        assert (
            not hasattr(mock_manifest_2, "fetch_manifest_entry")
            or not mock_manifest_2.fetch_manifest_entry.called
        )

        # Verify we called manifests() with file_io
        mock_snapshot.manifests.assert_called_once_with(metadata_writer.file_io)

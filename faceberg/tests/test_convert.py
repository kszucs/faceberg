"""Tests for the convert module (Iceberg metadata generation)."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from pyiceberg.schema import Schema
from pyiceberg.types import IntegerType, NestedField, StringType, StructType

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
    return IcebergMetadataWriter(table_path=temp_table_path, schema=simple_schema)


class TestGetHfFileSize:
    """Tests for the _get_hf_file_size method."""

    def test_get_hf_file_size_valid_url(self, metadata_writer):
        """Test getting file size from a valid HuggingFace URL."""
        test_url = "hf://datasets/deepmind/narrativeqa/data/train-00000-of-00024.parquet"

        with patch("faceberg.convert.hf_hub_url") as mock_hf_hub_url, patch(
            "faceberg.convert.get_hf_file_metadata"
        ) as mock_get_metadata:
            # Setup mocks
            mock_hf_hub_url.return_value = "https://huggingface.co/datasets/deepmind/narrativeqa/resolve/main/data/train-00000-of-00024.parquet"

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
            )
            mock_get_metadata.assert_called_once()

    def test_get_hf_file_size_nested_path(self, metadata_writer):
        """Test getting file size from a URL with deeply nested path."""
        test_url = "hf://datasets/org/repo/path/to/deep/file.parquet"

        with patch("faceberg.convert.hf_hub_url") as mock_hf_hub_url, patch(
            "faceberg.convert.get_hf_file_metadata"
        ) as mock_get_metadata:
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
            )

    def test_get_hf_file_size_invalid_url_format(self, metadata_writer):
        """Test handling of invalid URL format."""
        # Not a hf:// URL
        file_size = metadata_writer._get_hf_file_size("s3://bucket/file.parquet")
        assert file_size == 0

        # Invalid hf:// URL (too few parts)
        file_size = metadata_writer._get_hf_file_size("hf://datasets/repo")
        assert file_size == 0

    def test_get_hf_file_size_api_error(self, metadata_writer):
        """Test handling of HuggingFace API errors."""
        test_url = "hf://datasets/org/repo/file.parquet"

        with patch("faceberg.convert.hf_hub_url") as mock_hf_hub_url, patch(
            "faceberg.convert.get_hf_file_metadata"
        ) as mock_get_metadata:
            mock_hf_hub_url.return_value = "https://mock.url"
            # Simulate API error
            mock_get_metadata.side_effect = Exception("API Error")

            # Should return 0 and log warning
            file_size = metadata_writer._get_hf_file_size(test_url)
            assert file_size == 0

    def test_get_hf_file_size_repository_not_found(self, metadata_writer):
        """Test handling when repository doesn't exist."""
        test_url = "hf://datasets/nonexistent/repo/file.parquet"

        with patch("faceberg.convert.hf_hub_url") as mock_hf_hub_url, patch(
            "faceberg.convert.get_hf_file_metadata"
        ) as mock_get_metadata:
            mock_hf_hub_url.return_value = "https://mock.url"
            # Simulate any exception (generic error handling)
            mock_get_metadata.side_effect = RuntimeError("Repository not found")

            file_size = metadata_writer._get_hf_file_size(test_url)
            assert file_size == 0


class TestReadFileMetadata:
    """Tests for the _read_file_metadata method."""

    def test_read_file_metadata_uses_hf_file_size(self, metadata_writer):
        """Test that _read_file_metadata uses HuggingFace file size when size_bytes is 0."""
        file_infos = [
            FileInfo(
                path="hf://datasets/org/repo/file1.parquet",
                size_bytes=0,  # No size provided
                row_count=0,
                split="train",
            )
        ]

        with patch("faceberg.convert.pq.read_metadata") as mock_read_metadata, patch.object(
            metadata_writer, "_get_hf_file_size"
        ) as mock_get_size:
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
            assert enriched[0].path == "hf://datasets/org/repo/file1.parquet"
            assert enriched[0].size_bytes == 9876543
            assert enriched[0].row_count == 1000
            mock_get_size.assert_called_once_with("hf://datasets/org/repo/file1.parquet")

    def test_read_file_metadata_preserves_provided_size(self, metadata_writer):
        """Test that _read_file_metadata preserves size_bytes when already provided."""
        file_infos = [
            FileInfo(
                path="hf://datasets/org/repo/file1.parquet",
                size_bytes=5555555,  # Size already provided
                row_count=0,
                split="train",
            )
        ]

        with patch("faceberg.convert.pq.read_metadata") as mock_read_metadata, patch.object(
            metadata_writer, "_get_hf_file_size"
        ) as mock_get_size:
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
                path="hf://datasets/org/repo/file1.parquet",
                size_bytes=0,
                row_count=0,
                split="train",
            ),
            FileInfo(
                path="hf://datasets/org/repo/file2.parquet",
                size_bytes=0,
                row_count=0,
                split="train",
            ),
            FileInfo(
                path="hf://datasets/org/repo/file3.parquet",
                size_bytes=123456,  # Already has size
                row_count=500,  # This will be overwritten by reading parquet metadata
                split="test",
            ),
        ]

        with patch("faceberg.convert.pq.read_metadata") as mock_read_metadata, patch.object(
            metadata_writer, "_get_hf_file_size"
        ) as mock_get_size:
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

            # Mock file sizes for files without size
            mock_get_size.side_effect = [9999999, 8888888]

            enriched = metadata_writer._read_file_metadata(file_infos)

            assert len(enriched) == 3
            # File 1: no size, gets it from HF
            assert enriched[0].size_bytes == 9999999
            assert enriched[0].row_count == 1000
            # File 2: no size, gets it from HF
            assert enriched[1].size_bytes == 8888888
            assert enriched[1].row_count == 2000
            # File 3: has size, uses it, but row_count is read from parquet metadata
            assert enriched[2].size_bytes == 123456
            assert enriched[2].row_count == 3000  # Overwritten by parquet metadata
            # Should only call _get_hf_file_size for first two files (file3 has size)
            assert mock_get_size.call_count == 2

    def test_read_file_metadata_handles_read_error(self, metadata_writer):
        """Test that files with metadata read errors are kept with original info."""
        file_infos = [
            FileInfo(
                path="hf://datasets/org/repo/file1.parquet",
                size_bytes=0,
                row_count=0,
                split="train",
            )
        ]

        with patch("faceberg.convert.pq.read_metadata") as mock_read_metadata:
            # Simulate read error
            mock_read_metadata.side_effect = Exception("Cannot read metadata")

            enriched = metadata_writer._read_file_metadata(file_infos)

            # Should keep original file info
            assert len(enriched) == 1
            assert enriched[0].path == file_infos[0].path
            assert enriched[0].size_bytes == 0
            assert enriched[0].row_count == 0


class TestFileSizeRegression:
    """Regression tests to ensure the bug fix works correctly."""

    def test_file_size_not_using_serialized_size(self, metadata_writer):
        """Regression test: ensure we don't use metadata.serialized_size (the original bug)."""
        # This is the key regression test for the bug fix
        file_infos = [
            FileInfo(
                path="hf://datasets/deepmind/narrativeqa/data/train-00000-of-00024.parquet",
                size_bytes=0,
                row_count=0,
                split="train",
            )
        ]

        with patch("faceberg.convert.pq.read_metadata") as mock_read_metadata, patch.object(
            metadata_writer, "_get_hf_file_size"
        ) as mock_get_size:
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
                path=f"hf://datasets/deepmind/narrativeqa/data/train-{i:05d}-of-00024.parquet",
                size_bytes=0,
                row_count=0,
                split="train",
            )
            for i in range(5)
        ]

        # Real-world file sizes from the bug report
        real_sizes = [9766702, 67176993, 232523620, 27221729, 88315563]
        # Typical metadata.serialized_size values (around 500-600 bytes)
        serialized_sizes = [18853, 10532, 11971, 9938, 19011]

        with patch("faceberg.convert.pq.read_metadata") as mock_read_metadata, patch.object(
            metadata_writer, "_get_hf_file_size"
        ) as mock_get_size:

            def get_metadata_side_effect(path):
                idx = int(path.split("train-")[1].split("-of")[0])
                mock_metadata = Mock()
                mock_metadata.num_rows = 1000
                mock_metadata.serialized_size = serialized_sizes[idx]
                return mock_metadata

            mock_read_metadata.side_effect = get_metadata_side_effect
            mock_get_size.side_effect = real_sizes

            enriched = metadata_writer._read_file_metadata(file_infos)

            # Verify all files have correct sizes
            for i, file_info in enumerate(enriched):
                assert file_info.size_bytes == real_sizes[i]
                # Verify we're not using the wrong serialized_size
                assert file_info.size_bytes != serialized_sizes[i]
                # Verify the ratio is in the expected range
                ratio = file_info.size_bytes / serialized_sizes[i]
                assert 500 <= ratio <= 20000  # Based on real-world observations

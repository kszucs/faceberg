# tests/test_discover.py
import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from datasets import Features, Value

from faceberg.discover import DatasetInfo, ParquetFile, dataset_builder_safe, discover_dataset


def test_parquet_file_creation():
    """Test creating a ParquetFile with all fields."""
    pf = ParquetFile(
        uri="hf://datasets/squad@abc123/train-00000.parquet",
        path="train-00000.parquet",
        size=1024,
        blob_id="abc123def456",
        split="train",
    )
    assert pf.uri == "hf://datasets/squad@abc123/train-00000.parquet"
    assert pf.path == "train-00000.parquet"
    assert pf.size == 1024
    assert pf.blob_id == "abc123def456"
    assert pf.split == "train"


def test_parquet_file_optional_split():
    """Test ParquetFile with optional split."""
    pf = ParquetFile(
        uri="hf://datasets/squad@abc123/data.parquet",
        path="data.parquet",
        size=2048,
        blob_id="xyz789",
        split=None,
    )
    assert pf.split is None


def test_dataset_info_creation():
    """Test creating a DatasetInfo with all fields."""
    features = Features({"text": Value("string")})
    files = [
        ParquetFile(
            uri="hf://datasets/squad@abc123/train-00000.parquet",
            path="train-00000.parquet",
            size=1024,
            blob_id="blob1",
            split="train",
        )
    ]

    info = DatasetInfo(
        repo_id="squad",
        config="plain_text",
        revision="abc123",
        features=features,
        splits=["train", "test"],
        data_dir="data",
        files=files,
    )

    assert info.repo_id == "squad"
    assert info.config == "plain_text"
    assert info.revision == "abc123"
    assert info.features == features
    assert info.splits == ["train", "test"]
    assert info.data_dir == "data"
    assert len(info.files) == 1
    assert info.files[0].uri == "hf://datasets/squad@abc123/train-00000.parquet"


def test_dataset_builder_safe_changes_directory(tmp_path):
    """Test that dataset_builder_safe changes to temp directory."""
    original_cwd = os.getcwd()

    with patch("faceberg.discover.load_dataset_builder") as mock_load:
        mock_builder = MagicMock()
        mock_load.return_value = mock_builder

        result = dataset_builder_safe("squad", "plain_text", token="test_token")

        # Should be back in original directory
        assert os.getcwd() == original_cwd
        assert result == mock_builder
        mock_load.assert_called_once_with("squad", "plain_text", token="test_token")


def test_dataset_builder_safe_restores_directory_on_error(tmp_path):
    """Test that directory is restored even on error."""
    original_cwd = os.getcwd()

    with patch("faceberg.discover.load_dataset_builder") as mock_load:
        mock_load.side_effect = Exception("Load failed")

        try:
            dataset_builder_safe("squad", "plain_text")
        except Exception:
            pass

        # Should be back in original directory even after error
        assert os.getcwd() == original_cwd


def test_discover_dataset_basic():
    """Test basic dataset discovery with mocked APIs."""
    # Mock builder
    mock_builder = MagicMock()
    mock_builder.name = "parquet"
    mock_builder.hash = "abc123def456"
    mock_builder.info.features = Features({"text": Value("string")})
    mock_builder.config.name = "plain_text"
    mock_builder.config.data_files = {
        "train": ["hf://datasets/squad@abc123def456/data/train-00000.parquet"],
    }

    # Mock HuggingFace API response
    mock_sibling = MagicMock()
    mock_sibling.rfilename = "data/train-00000.parquet"
    mock_sibling.size = 1024
    mock_sibling.blob_id = "blob123"

    mock_dataset_info = MagicMock()
    mock_dataset_info.siblings = [mock_sibling]

    with patch("faceberg.discover.dataset_builder_safe", return_value=mock_builder):
        with patch("faceberg.discover.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.dataset_info.return_value = mock_dataset_info
            mock_api_class.return_value = mock_api

            result = discover_dataset("squad", "plain_text", token="test_token")

    # Verify result
    assert result.repo_id == "squad"
    assert result.config == "plain_text"
    assert result.revision == "abc123def456"
    assert result.splits == ["train"]
    assert result.data_dir == "data"
    assert len(result.files) == 1
    assert result.files[0].uri == "hf://datasets/squad@abc123def456/data/train-00000.parquet"
    assert result.files[0].size == 1024
    assert result.files[0].blob_id == "blob123"
    assert result.files[0].split == "train"


def test_discover_dataset_multiple_splits():
    """Test discovery with multiple splits."""
    # Mock builder
    mock_builder = MagicMock()
    mock_builder.name = "parquet"
    mock_builder.hash = "xyz789"
    mock_builder.info.features = Features({"text": Value("string")})
    mock_builder.config.data_files = {
        "train": ["hf://datasets/squad@xyz789/data/train-00000.parquet"],
        "test": ["hf://datasets/squad@xyz789/data/test-00000.parquet"],
        "validation": ["hf://datasets/squad@xyz789/data/validation-00000.parquet"],
    }

    # Mock HuggingFace API response
    mock_siblings = [
        Mock(rfilename="data/train-00000.parquet", size=1024, oid="blob1"),
        Mock(rfilename="data/test-00000.parquet", size=512, oid="blob2"),
        Mock(rfilename="data/validation-00000.parquet", size=256, oid="blob3"),
    ]

    mock_dataset_info = MagicMock()
    mock_dataset_info.siblings = mock_siblings

    with patch("faceberg.discover.dataset_builder_safe", return_value=mock_builder):
        with patch("faceberg.discover.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.dataset_info.return_value = mock_dataset_info
            mock_api_class.return_value = mock_api

            result = discover_dataset("squad", "plain_text")

    # Verify result
    assert result.splits == ["train", "test", "validation"]
    assert len(result.files) == 3

    # Check each file has correct split
    train_files = [f for f in result.files if f.split == "train"]
    test_files = [f for f in result.files if f.split == "test"]
    val_files = [f for f in result.files if f.split == "validation"]

    assert len(train_files) == 1
    assert len(test_files) == 1
    assert len(val_files) == 1

    assert train_files[0].size == 1024
    assert test_files[0].size == 512
    assert val_files[0].size == 256


def test_discover_dataset_empty():
    """Test discovery of dataset with no files."""
    # Mock builder with no data files
    mock_builder = MagicMock()
    mock_builder.name = "parquet"
    mock_builder.hash = "empty123"
    mock_builder.info.features = Features({"text": Value("string")})
    mock_builder.config.data_files = {}

    mock_dataset_info = MagicMock()
    mock_dataset_info.siblings = []

    with patch("faceberg.discover.dataset_builder_safe", return_value=mock_builder):
        with patch("faceberg.discover.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.dataset_info.return_value = mock_dataset_info
            mock_api_class.return_value = mock_api

            result = discover_dataset("empty", "default")

    # Verify result
    assert result.repo_id == "empty"
    assert result.splits == []
    assert result.files == []
    assert result.data_dir == ""


def test_discover_dataset_missing_file_metadata():
    """Test that missing file metadata raises ValueError."""
    # Mock builder with file that won't be in metadata
    mock_builder = MagicMock()
    mock_builder.name = "parquet"
    mock_builder.hash = "missing123"
    mock_builder.info.features = Features({"text": Value("string")})
    mock_builder.config.data_files = {
        "train": ["hf://datasets/squad@missing123/data/missing.parquet"],
    }

    # Mock HuggingFace API response without the file
    mock_dataset_info = MagicMock()
    mock_dataset_info.siblings = []  # No files in metadata

    with patch("faceberg.discover.dataset_builder_safe", return_value=mock_builder):
        with patch("faceberg.discover.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.dataset_info.return_value = mock_dataset_info
            mock_api_class.return_value = mock_api

            with pytest.raises(ValueError, match="not found in Hub API response"):
                discover_dataset("squad", "plain_text")


def test_discover_dataset_invalid_dataset():
    """Test that invalid dataset raises ValueError."""
    with patch("faceberg.discover.dataset_builder_safe") as mock_safe:
        mock_safe.side_effect = Exception("Dataset not found")

        with pytest.raises(ValueError, match="not found or not accessible"):
            discover_dataset("invalid", "config")


def test_discover_dataset_invalid_config():
    """Test that invalid config raises ValueError."""
    with patch("faceberg.discover.dataset_builder_safe") as mock_safe:
        mock_safe.side_effect = Exception("Config 'invalid' not found")

        with pytest.raises(ValueError, match="not found or not accessible"):
            discover_dataset("squad", "invalid")

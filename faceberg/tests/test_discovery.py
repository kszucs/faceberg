"""Tests for dataset discovery."""

import pytest

from faceberg.discovery import DatasetInfo


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

    print("\n✓ All tests passed!")

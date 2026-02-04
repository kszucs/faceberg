# Dataset Discovery Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a focused `discover.py` module for discovering HuggingFace dataset structure and files without any Iceberg-specific logic.

**Architecture:** Single-responsibility module with `discover_dataset()` function that queries HuggingFace Hub API to gather dataset metadata, features, splits, and Parquet files. Returns a simple `DatasetInfo` dataclass with all discovery results.

**Tech Stack:** Python 3.10+, datasets library, huggingface_hub, pytest

---

## Task 1: Create ParquetFile dataclass and helper

**Files:**
- Create: `faceberg/discover.py`

**Step 1: Write the test for ParquetFile creation**

Create the test file first to drive implementation:

```python
# tests/test_discover.py
from faceberg.discover import ParquetFile


def test_parquet_file_creation():
    """Test creating a ParquetFile with all fields."""
    pf = ParquetFile(
        uri="hf://datasets/squad@abc123/train-00000.parquet",
        size=1024,
        blob_id="abc123def456",
        split="train",
    )
    assert pf.uri == "hf://datasets/squad@abc123/train-00000.parquet"
    assert pf.size == 1024
    assert pf.blob_id == "abc123def456"
    assert pf.split == "train"


def test_parquet_file_optional_split():
    """Test ParquetFile with optional split."""
    pf = ParquetFile(
        uri="hf://datasets/squad@abc123/data.parquet",
        size=2048,
        blob_id="xyz789",
        split=None,
    )
    assert pf.split is None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_discover.py::test_parquet_file_creation -v
```

Expected: FAIL with "cannot import name 'ParquetFile'"

**Step 3: Implement ParquetFile dataclass**

```python
# faceberg/discover.py
"""HuggingFace dataset discovery.

This module discovers HuggingFace datasets and extracts metadata without
any Iceberg-specific conversions. It provides the foundation for converting
datasets to Iceberg tables.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from datasets import Features, load_dataset_builder
from huggingface_hub import HfApi


@dataclass
class ParquetFile:
    """A Parquet file discovered in a HuggingFace dataset.

    Attributes:
        uri: Full hf:// URI with revision (e.g., hf://datasets/repo@sha/file.parquet)
        size: File size in bytes
        blob_id: Git blob ID (oid) from HuggingFace
        split: Optional split name (train, test, validation, etc.)
    """

    uri: str
    size: int
    blob_id: str
    split: Optional[str] = None
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_discover.py::test_parquet_file_creation tests/test_discover.py::test_parquet_file_optional_split -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add faceberg/discover.py tests/test_discover.py
git commit -m "Add ParquetFile dataclass for dataset discovery"
```

---

## Task 2: Create DatasetInfo dataclass

**Files:**
- Modify: `faceberg/discover.py`
- Modify: `tests/test_discover.py`

**Step 1: Write the test for DatasetInfo creation**

```python
# tests/test_discover.py
from datasets import Features, Value
from faceberg.discover import DatasetInfo, ParquetFile


def test_dataset_info_creation():
    """Test creating a DatasetInfo with all fields."""
    features = Features({"text": Value("string")})
    files = [
        ParquetFile(
            uri="hf://datasets/squad@abc123/train-00000.parquet",
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_discover.py::test_dataset_info_creation -v
```

Expected: FAIL with "cannot import name 'DatasetInfo'"

**Step 3: Implement DatasetInfo dataclass**

```python
# faceberg/discover.py (add after ParquetFile)

@dataclass
class DatasetInfo:
    """Complete information about a discovered HuggingFace dataset.

    This represents the result of dataset discovery, containing all metadata
    needed to understand the dataset structure without any Iceberg conversions.

    Attributes:
        repo_id: HuggingFace repository ID (e.g., "squad")
        config: Configuration name
        revision: Git revision SHA
        features: HuggingFace Features object describing the schema
        splits: List of split names (e.g., ["train", "test"])
        data_dir: Common directory path containing data files
        files: List of all discovered Parquet files
    """

    repo_id: str
    config: str
    revision: str
    features: Features
    splits: List[str]
    data_dir: str
    files: List[ParquetFile]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_discover.py::test_dataset_info_creation -v
```

Expected: 1 passed

**Step 5: Commit**

```bash
git add faceberg/discover.py tests/test_discover.py
git commit -m "Add DatasetInfo dataclass for discovery results"
```

---

## Task 3: Add dataset_builder_safe helper

**Files:**
- Modify: `faceberg/discover.py`
- Modify: `tests/test_discover.py`

**Step 1: Write the test for dataset_builder_safe**

```python
# tests/test_discover.py
from unittest.mock import MagicMock, patch
from faceberg.discover import dataset_builder_safe


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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_discover.py::test_dataset_builder_safe_changes_directory -v
```

Expected: FAIL with "cannot import name 'dataset_builder_safe'"

**Step 3: Implement dataset_builder_safe**

```python
# faceberg/discover.py (add after imports, before dataclasses)

def dataset_builder_safe(
    repo_id: str,
    config: str,
    token: Optional[str] = None,
):
    """Load dataset builder while avoiding picking up local files.

    Changes to a temporary directory before loading to ensure the datasets
    library doesn't pick up local files in the current directory.

    Args:
        repo_id: HuggingFace dataset repository ID
        config: Configuration name
        token: Optional HuggingFace API token

    Returns:
        Dataset builder object

    Raises:
        Exception: If loading fails
    """
    original_cwd = os.getcwd()

    try:
        # Change to a temporary directory to avoid dataset library picking up local files
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            return load_dataset_builder(repo_id, config, token=token)
    finally:
        # Always restore the original directory
        os.chdir(original_cwd)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_discover.py::test_dataset_builder_safe_changes_directory tests/test_discover.py::test_dataset_builder_safe_restores_directory_on_error -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add faceberg/discover.py tests/test_discover.py
git commit -m "Add dataset_builder_safe helper function"
```

---

## Task 4: Implement discover_dataset function (basic structure)

**Files:**
- Modify: `faceberg/discover.py`
- Modify: `tests/test_discover.py`

**Step 1: Write the test for basic discovery with mocked APIs**

```python
# tests/test_discover.py
from unittest.mock import MagicMock, patch, Mock
from faceberg.discover import discover_dataset


def test_discover_dataset_basic():
    """Test basic dataset discovery with mocked APIs."""
    # Mock builder
    mock_builder = MagicMock()
    mock_builder.hash = "abc123def456"
    mock_builder.info.features = Features({"text": Value("string")})
    mock_builder.config.data_files = {
        "train": ["hf://datasets/squad@abc123def456/data/train-00000.parquet"],
    }

    # Mock HuggingFace API response
    mock_sibling = MagicMock()
    mock_sibling.rfilename = "data/train-00000.parquet"
    mock_sibling.size = 1024
    mock_sibling.oid = "blob123"

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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_discover.py::test_discover_dataset_basic -v
```

Expected: FAIL with "cannot import name 'discover_dataset'"

**Step 3: Implement discover_dataset function**

```python
# faceberg/discover.py (add at end of file)

def discover_dataset(
    repo_id: str,
    config: str,
    token: Optional[str] = None,
) -> DatasetInfo:
    """Discover structure and files in a HuggingFace dataset.

    Queries the HuggingFace Hub to gather dataset metadata, features, splits,
    and Parquet file information without any Iceberg-specific conversions.

    Args:
        repo_id: HuggingFace dataset repository ID (e.g., "squad")
        config: Configuration name to discover
        token: HuggingFace API token (uses HF_TOKEN env var if not provided)

    Returns:
        DatasetInfo with all files for the latest revision

    Raises:
        ValueError: If dataset not found, config doesn't exist, or metadata inconsistent
    """
    # Step 1: Load dataset builder
    try:
        builder = dataset_builder_safe(repo_id, config=config, token=token)
    except Exception as e:
        raise ValueError(
            f"Dataset {repo_id} config {config} not found or not accessible: {e}"
        ) from e

    revision = builder.hash
    features = builder.info.features

    # Step 2: Fetch file metadata from HuggingFace Hub
    api = HfApi(token=token)
    dataset_info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    file_metadata = {s.rfilename: s for s in dataset_info.siblings}

    # Step 3: Process data files
    files = []
    for split, file_uris in builder.config.data_files.items():
        for uri in file_uris:
            # Extract path from URI (format: hf://datasets/repo@revision/path)
            # Split by '@' to get revision part, then by '/' to get path after revision
            parts = uri.split("@", 1)[1].split("/", 1)
            path = parts[1] if len(parts) > 1 else ""

            # Get metadata (strict - fail if not found)
            if path not in file_metadata:
                raise ValueError(
                    f"File {uri} from dataset builder not found in Hub API response. "
                    f"This may indicate an inconsistent dataset state."
                )

            metadata = file_metadata[path]

            # Create ParquetFile
            files.append(
                ParquetFile(
                    uri=uri,
                    size=metadata.size,
                    blob_id=metadata.oid,
                    split=split,
                )
            )

    # Step 4: Extract common data directory
    if files:
        try:
            file_paths = [
                uri.split("@", 1)[1].split("/", 1)[1] for uri in [f.uri for f in files]
            ]
            file_dirs = [os.path.dirname(path) for path in file_paths]
            data_dir = os.path.commonpath(file_dirs) if file_dirs else ""
        except ValueError as e:
            raise ValueError(
                f"Unable to determine common data directory from files: {file_paths}"
            ) from e
    else:
        data_dir = ""

    # Step 5: Return DatasetInfo
    return DatasetInfo(
        repo_id=repo_id,
        config=config,
        revision=revision,
        features=features,
        splits=list(builder.config.data_files.keys()),
        data_dir=data_dir,
        files=files,
    )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_discover.py::test_discover_dataset_basic -v
```

Expected: 1 passed

**Step 5: Commit**

```bash
git add faceberg/discover.py tests/test_discover.py
git commit -m "Implement discover_dataset function"
```

---

## Task 5: Add test for multiple splits

**Files:**
- Modify: `tests/test_discover.py`

**Step 1: Write the test for multiple splits**

```python
# tests/test_discover.py (add to file)

def test_discover_dataset_multiple_splits():
    """Test discovery with multiple splits."""
    # Mock builder
    mock_builder = MagicMock()
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
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_discover.py::test_discover_dataset_multiple_splits -v
```

Expected: 1 passed

**Step 3: Commit**

```bash
git add tests/test_discover.py
git commit -m "Add test for multiple splits discovery"
```

---

## Task 6: Add test for empty dataset

**Files:**
- Modify: `tests/test_discover.py`

**Step 1: Write the test for empty dataset**

```python
# tests/test_discover.py (add to file)

def test_discover_dataset_empty():
    """Test discovery of dataset with no files."""
    # Mock builder with no data files
    mock_builder = MagicMock()
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
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_discover.py::test_discover_dataset_empty -v
```

Expected: 1 passed

**Step 3: Commit**

```bash
git add tests/test_discover.py
git commit -m "Add test for empty dataset discovery"
```

---

## Task 7: Add test for missing file metadata error

**Files:**
- Modify: `tests/test_discover.py`

**Step 1: Write the test for missing file metadata**

```python
# tests/test_discover.py (add to file)
import pytest


def test_discover_dataset_missing_file_metadata():
    """Test that missing file metadata raises ValueError."""
    # Mock builder with file that won't be in metadata
    mock_builder = MagicMock()
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
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_discover.py::test_discover_dataset_missing_file_metadata -v
```

Expected: 1 passed

**Step 3: Commit**

```bash
git add tests/test_discover.py
git commit -m "Add test for missing file metadata error"
```

---

## Task 8: Add test for invalid dataset error

**Files:**
- Modify: `tests/test_discover.py`

**Step 1: Write the test for invalid dataset**

```python
# tests/test_discover.py (add to file)

def test_discover_dataset_invalid_dataset():
    """Test that invalid dataset raises ValueError."""
    with patch("faceberg.discover.dataset_builder_safe") as mock_safe:
        mock_safe.side_effect = Exception("Dataset not found")

        with pytest.raises(ValueError, match="not found or not accessible"):
            discover_dataset("invalid", "config")
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_discover.py::test_discover_dataset_invalid_dataset -v
```

Expected: 1 passed

**Step 3: Commit**

```bash
git add tests/test_discover.py
git commit -m "Add test for invalid dataset error"
```

---

## Task 9: Add test for invalid config error

**Files:**
- Modify: `tests/test_discover.py`

**Step 1: Write the test for invalid config**

```python
# tests/test_discover.py (add to file)

def test_discover_dataset_invalid_config():
    """Test that invalid config raises ValueError."""
    with patch("faceberg.discover.dataset_builder_safe") as mock_safe:
        mock_safe.side_effect = Exception("Config 'invalid' not found")

        with pytest.raises(ValueError, match="not found or not accessible"):
            discover_dataset("squad", "invalid")
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_discover.py::test_discover_dataset_invalid_config -v
```

Expected: 1 passed

**Step 3: Commit**

```bash
git add tests/test_discover.py
git commit -m "Add test for invalid config error"
```

---

## Task 10: Add missing import for os in tests

**Files:**
- Modify: `tests/test_discover.py`

**Step 1: Add os import to test file**

```python
# tests/test_discover.py (at top of file)
import os
from unittest.mock import MagicMock, patch, Mock

import pytest
from datasets import Features, Value

from faceberg.discover import DatasetInfo, ParquetFile, dataset_builder_safe, discover_dataset
```

**Step 2: Run all tests to verify everything passes**

```bash
pytest tests/test_discover.py -v
```

Expected: All tests passed

**Step 3: Commit**

```bash
git add tests/test_discover.py
git commit -m "Add missing import in test_discover.py"
```

---

## Task 11: Update iceberg.py to remove ParquetFile

**Files:**
- Modify: `faceberg/iceberg.py`

**Step 1: Check for usage of ParquetFile in iceberg.py**

```bash
grep -n "class ParquetFile" faceberg/iceberg.py
grep -n "ParquetFile" faceberg/iceberg.py
```

**Step 2: Remove ParquetFile class definition**

Remove lines 125-132 from `faceberg/iceberg.py`:

```python
# DELETE THESE LINES:
@dataclass
class ParquetFile:
    """A parquet file to be added or removed from a snapshot."""

    uri: str
    size: int
    hash: str
```

**Step 3: Add import from discover module**

At the top of `faceberg/iceberg.py`, add:

```python
from faceberg.discover import ParquetFile
```

Update the import block to look like:

```python
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa

from faceberg.discover import ParquetFile
from pyiceberg.io import FileIO
# ... rest of imports
```

**Step 4: Update references to use blob_id instead of hash**

Find and update references in `iceberg.py`:
- Line 131: `hash: str` → already removed with class
- Line 190: `hash=""` → `blob_id=""`
- Line 198: `hash=""` → `blob_id=""`

Update these lines:

```python
# Line 190 (in diff_snapshot)
old_pf = ParquetFile(uri=pf.uri, size=prev_size, blob_id="")

# Line 198 (in diff_snapshot)
removed_pf = ParquetFile(uri=uri, size=size, blob_id="")
```

**Step 5: Run tests to verify iceberg module still works**

```bash
pytest tests/test_iceberg.py -v
```

Expected: All tests should pass (if they exist)

**Step 6: Commit**

```bash
git add faceberg/iceberg.py
git commit -m "Move ParquetFile to discover module and update references"
```

---

## Task 12: Run full test suite

**Files:**
- N/A (verification only)

**Step 1: Run all tests**

```bash
pytest tests/test_discover.py -v
```

Expected: All tests pass

**Step 2: Check test coverage (optional)**

```bash
pytest tests/test_discover.py --cov=faceberg.discover --cov-report=term-missing
```

Review coverage - should be high for the new module.

**Step 3: Final verification - import the module**

```bash
python -c "from faceberg.discover import discover_dataset, DatasetInfo, ParquetFile; print('Import successful')"
```

Expected: "Import successful"

---

## Summary

✅ Created `faceberg/discover.py` with:
- `ParquetFile` dataclass (moved from iceberg.py)
- `DatasetInfo` dataclass
- `dataset_builder_safe()` helper
- `discover_dataset()` function

✅ Created `tests/test_discover.py` with comprehensive tests:
- ParquetFile creation
- DatasetInfo creation
- Helper function behavior
- Basic discovery
- Multiple splits
- Empty datasets
- Error conditions (missing metadata, invalid dataset/config)

✅ Updated `faceberg/iceberg.py`:
- Removed ParquetFile class
- Added import from discover module
- Updated field name from `hash` to `blob_id`

**Module is ready for use!**

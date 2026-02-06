"""HuggingFace dataset discovery.

This module discovers HuggingFace datasets and extracts metadata without
any Iceberg-specific conversions. It provides the foundation for converting
datasets to Iceberg tables.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional

from datasets import Features, StreamingDownloadManager, load_dataset_builder
from huggingface_hub import HfApi


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


@dataclass
class ParquetFile:
    """A Parquet file discovered in a HuggingFace dataset.

    Attributes:
        uri: Full hf:// URI with revision (e.g., hf://datasets/repo@sha/file.parquet)
        path: File path within the dataset (e.g., data/train-00000.parquet)
        size: File size in bytes
        blob_id: Git blob ID (oid) from HuggingFace
        split: Optional split name (train, test, validation, etc.)
    """

    uri: str
    path: str
    size: int
    blob_id: str
    split: Optional[str] = None


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


def discover_dataset(
    repo_id: str,
    config: str,
    token: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
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
    if progress_callback is None:

        def progress(*args, **kwargs):
            pass
    else:
        progress = progress_callback

    # Step 1: Load dataset builder
    progress(state="in_progress", percent=0, stage="Loading dataset builder")
    try:
        builder = dataset_builder_safe(repo_id, config=config, token=token)
    except Exception as e:
        raise ValueError(
            f"Dataset {repo_id} config {config} not found or not accessible: {e}"
        ) from e

    # Step 1.1: Infer features if they are absent from the dataset card metadata
    if builder.info.features is None:
        dl_manager = StreamingDownloadManager()
        builder._split_generators(dl_manager)

    revision = builder.hash
    features = builder.info.features

    # Step 2: Fetch file metadata from HuggingFace Hub
    progress(state="in_progress", percent=5, stage="Fetching list of dataset files")
    api = HfApi(token=token)
    dataset_info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    # Build mapping from URI to sibling metadata
    file_metadata = {
        f"hf://datasets/{repo_id}@{revision}/{s.rfilename}": s for s in dataset_info.siblings
    }

    # Step 3: Process data files
    files = []
    for split, file_uris in builder.config.data_files.items():
        for uri in file_uris:
            # Get metadata (strict - fail if not found)
            if uri not in file_metadata:
                raise ValueError(
                    f"File {uri} from dataset builder not found in Hub API response. "
                    f"This may indicate an inconsistent dataset state."
                )

            metadata = file_metadata[uri]

            # Create ParquetFile
            files.append(
                ParquetFile(
                    uri=uri,
                    path=metadata.rfilename,
                    size=metadata.size,
                    blob_id=metadata.blob_id,
                    split=split,
                )
            )

    # Step 4: Extract common data directory
    if files:
        try:
            file_dirs = [os.path.dirname(f.path) for f in files]
            data_dir = os.path.commonpath(file_dirs) if file_dirs else ""
        except ValueError as e:
            file_paths = [f.path for f in files]
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

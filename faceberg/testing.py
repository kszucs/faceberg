"""Testing utilities for Faceberg.

This module provides mock implementations of external services for testing purposes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, BinaryIO, Literal

from huggingface_hub import CommitOperationAdd, CommitOperationDelete


class MockHfApi:
    """Mock implementation of HuggingFace Hub API for local testing.

    This mock provides a local filesystem-based implementation of the HfApi
    interface, allowing tests to run without network access or actual HuggingFace
    Hub interaction.

    Args:
        cache_dir: Base directory for storing mock repositories. If None, creates
            a temporary directory.
        token: Authentication token (ignored in mock, kept for compatibility).

    Example:
        >>> api = MockHfApi(cache_dir="/tmp/mock_hf")
        >>> api.create_repo("my-org/my-repo", repo_type="dataset")
        >>> api.upload_file("data.csv", "data.csv", "my-org/my-repo", repo_type="dataset")
    """

    def __init__(self, cache_dir: str | Path | None = None, token: str | None = None):
        """Initialize the mock HfApi.

        Args:
            cache_dir: Directory to use for mock storage
            token: Authentication token (ignored, for compatibility)
        """
        if cache_dir is None:
            import tempfile

            cache_dir = Path(tempfile.mkdtemp(prefix="mock_hf_"))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token = token

    def _get_repo_path(
        self, repo_id: str, repo_type: Literal["model", "dataset", "space"] = "model"
    ) -> Path:
        """Get the local filesystem path for a repository.

        Args:
            repo_id: Repository identifier (e.g., "org/repo")
            repo_type: Type of repository

        Returns:
            Path to the repository directory
        """
        # Sanitize repo_id for filesystem use
        safe_repo_id = repo_id.replace("/", "--")
        return self.cache_dir / repo_type / safe_repo_id

    def create_repo(
        self,
        repo_id: str,
        repo_type: Literal["model", "dataset", "space"] = "model",
        private: bool = False,
        exist_ok: bool = False,
        space_sdk: str | None = None,
    ) -> str:
        """Create a new repository.

        Args:
            repo_id: Repository identifier (e.g., "org/repo")
            repo_type: Type of repository
            private: Whether the repository should be private (ignored in mock)
            exist_ok: If True, don't raise error if repo exists
            space_sdk: SDK to use for spaces (e.g., "docker")

        Returns:
            Repository URL (mock URL)

        Raises:
            ValueError: If repository exists and exist_ok is False
        """
        repo_path = self._get_repo_path(repo_id, repo_type)

        if repo_path.exists() and not exist_ok:
            raise ValueError(f"Repository {repo_id} already exists")

        repo_path.mkdir(parents=True, exist_ok=True)

        # Store metadata
        metadata = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "private": private,
            "space_sdk": space_sdk,
        }
        (repo_path / ".hf_metadata.json").write_text(json.dumps(metadata, indent=2))

        return f"file://{repo_path}"

    def upload_file(
        self,
        path_or_fileobj: str | Path | BinaryIO,
        path_in_repo: str,
        repo_id: str,
        repo_type: Literal["model", "dataset", "space"] = "model",
        commit_message: str | None = None,
    ) -> str:
        """Upload a file to a repository.

        Args:
            path_or_fileobj: Local file path or file object to upload
            path_in_repo: Destination path in the repository
            repo_id: Repository identifier
            repo_type: Type of repository
            commit_message: Commit message (ignored in mock)

        Returns:
            File URL (mock URL)

        Raises:
            FileNotFoundError: If repository doesn't exist
        """
        repo_path = self._get_repo_path(repo_id, repo_type)

        if not repo_path.exists():
            raise FileNotFoundError(f"Repository {repo_id} not found")

        dest_path = repo_path / path_in_repo
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle file object or path
        if isinstance(path_or_fileobj, (str, Path)):
            shutil.copy2(path_or_fileobj, dest_path)
        else:
            # File object
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(path_or_fileobj, f)

        return f"file://{dest_path}"

    def create_commit(
        self,
        repo_id: str,
        operations: list[CommitOperationAdd | CommitOperationDelete],
        commit_message: str,
        repo_type: Literal["model", "dataset", "space"] = "model",
    ) -> str:
        """Create a commit with multiple operations.

        Args:
            repo_id: Repository identifier
            operations: List of file operations to perform
            commit_message: Commit message (ignored in mock)
            repo_type: Type of repository

        Returns:
            Commit URL (mock URL)

        Raises:
            FileNotFoundError: If repository doesn't exist
        """
        repo_path = self._get_repo_path(repo_id, repo_type)

        if not repo_path.exists():
            raise FileNotFoundError(f"Repository {repo_id} not found")

        for op in operations:
            if isinstance(op, CommitOperationAdd):
                dest_path = repo_path / op.path_in_repo
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Handle file object or path
                if isinstance(op.path_or_fileobj, (str, Path)):
                    shutil.copy2(op.path_or_fileobj, dest_path)
                else:
                    # File object
                    with open(dest_path, "wb") as f:
                        shutil.copyfileobj(op.path_or_fileobj, f)

            elif isinstance(op, CommitOperationDelete):
                file_path = repo_path / op.path_in_repo
                if file_path.exists():
                    file_path.unlink()

        return f"file://{repo_path}/commit/mock"

    def list_repo_files(
        self,
        repo_id: str,
        repo_type: Literal["model", "dataset", "space"] = "model",
    ) -> list[str]:
        """List all files in a repository.

        Args:
            repo_id: Repository identifier
            repo_type: Type of repository

        Returns:
            List of file paths relative to repository root

        Raises:
            FileNotFoundError: If repository doesn't exist
        """
        repo_path = self._get_repo_path(repo_id, repo_type)

        if not repo_path.exists():
            raise FileNotFoundError(f"Repository {repo_id} not found")

        files = []
        for path in repo_path.rglob("*"):
            if path.is_file() and not path.name.startswith(".hf_"):
                files.append(str(path.relative_to(repo_path)))
        return sorted(files)

    def repo_info(
        self,
        repo_id: str,
        repo_type: Literal["model", "dataset", "space"] = "model",
    ) -> dict[str, Any]:
        """Get repository information.

        Args:
            repo_id: Repository identifier
            repo_type: Type of repository

        Returns:
            Dictionary with repository metadata

        Raises:
            FileNotFoundError: If repository doesn't exist
        """
        repo_path = self._get_repo_path(repo_id, repo_type)

        if not repo_path.exists():
            raise FileNotFoundError(f"Repository {repo_id} not found")

        metadata_path = repo_path / ".hf_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        else:
            metadata = {"repo_id": repo_id, "repo_type": repo_type}

        return metadata

    def delete_repo(
        self,
        repo_id: str,
        repo_type: Literal["model", "dataset", "space"] = "model",
    ) -> None:
        """Delete a repository.

        Args:
            repo_id: Repository identifier
            repo_type: Type of repository

        Raises:
            FileNotFoundError: If repository doesn't exist
        """
        repo_path = self._get_repo_path(repo_id, repo_type)

        if not repo_path.exists():
            raise FileNotFoundError(f"Repository {repo_id} not found")

        shutil.rmtree(repo_path)

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        repo_type: Literal["model", "dataset", "space"] = "model",
    ) -> str:
        """Download a file from a repository.

        Args:
            repo_id: Repository identifier
            filename: Path to file in repository
            repo_type: Type of repository

        Returns:
            Local path to downloaded file

        Raises:
            FileNotFoundError: If repository or file doesn't exist
        """
        repo_path = self._get_repo_path(repo_id, repo_type)

        if not repo_path.exists():
            raise FileNotFoundError(f"Repository {repo_id} not found")

        file_path = repo_path / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"File {filename} not found in repository {repo_id}"
            )

        return str(file_path)


def hf_hub_download(
    repo_id: str,
    filename: str,
    repo_type: Literal["model", "dataset", "space"] = "model",
    cache_dir: str | Path | None = None,
) -> str:
    """Mock implementation of hf_hub_download as a standalone function.

    This mirrors the huggingface_hub.hf_hub_download interface for testing.

    Args:
        repo_id: Repository identifier
        filename: Path to file in repository
        repo_type: Type of repository
        cache_dir: Cache directory containing mock repositories

    Returns:
        Local path to file

    Raises:
        FileNotFoundError: If repository or file doesn't exist
    """
    api = MockHfApi(cache_dir=cache_dir)
    return api.hf_hub_download(repo_id, filename, repo_type)

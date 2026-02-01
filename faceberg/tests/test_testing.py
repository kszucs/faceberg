"""Tests for faceberg.testing module."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

from faceberg.testing import MockHfApi, hf_hub_download


@pytest.fixture
def mock_hf_cache():
    """Create a temporary directory for mock HuggingFace cache."""
    with tempfile.TemporaryDirectory(prefix="test_mock_hf_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_api(mock_hf_cache):
    """Create a MockHfApi instance with temporary cache."""
    return MockHfApi(cache_dir=mock_hf_cache)


class TestMockHfApiInit:
    """Tests for MockHfApi initialization."""

    def test_init_with_cache_dir(self, mock_hf_cache):
        """Test initialization with explicit cache directory."""
        api = MockHfApi(cache_dir=mock_hf_cache)
        assert api.cache_dir == mock_hf_cache
        assert api.cache_dir.exists()
        assert api.token is None

    def test_init_with_token(self, mock_hf_cache):
        """Test initialization with token (for compatibility)."""
        api = MockHfApi(cache_dir=mock_hf_cache, token="test_token")
        assert api.token == "test_token"

    def test_init_without_cache_dir(self):
        """Test initialization without cache directory creates temp dir."""
        api = MockHfApi()
        assert api.cache_dir.exists()
        assert "mock_hf_" in str(api.cache_dir)


class TestMockHfApiCreateRepo:
    """Tests for MockHfApi.create_repo method."""

    def test_create_repo_basic(self, mock_api):
        """Test basic repository creation."""
        url = mock_api.create_repo("test-org/test-repo", repo_type="dataset")
        assert "file://" in url
        assert "dataset" in url
        assert "test-org--test-repo" in url

        # Verify repository was created
        repo_path = mock_api._get_repo_path("test-org/test-repo", "dataset")
        assert repo_path.exists()
        assert repo_path.is_dir()

    def test_create_repo_with_metadata(self, mock_api):
        """Test repository creation stores metadata."""
        mock_api.create_repo(
            "test-org/test-repo", repo_type="dataset", private=True, space_sdk="docker"
        )

        repo_path = mock_api._get_repo_path("test-org/test-repo", "dataset")
        metadata_path = repo_path / ".hf_metadata.json"
        assert metadata_path.exists()

        metadata = json.loads(metadata_path.read_text())
        assert metadata["repo_id"] == "test-org/test-repo"
        assert metadata["repo_type"] == "dataset"
        assert metadata["private"] is True
        assert metadata["space_sdk"] == "docker"

    def test_create_repo_exist_ok_true(self, mock_api):
        """Test creating existing repository with exist_ok=True."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")
        # Should not raise error
        mock_api.create_repo("test-org/test-repo", repo_type="dataset", exist_ok=True)

    def test_create_repo_exist_ok_false(self, mock_api):
        """Test creating existing repository with exist_ok=False raises error."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")
        with pytest.raises(ValueError, match="already exists"):
            mock_api.create_repo("test-org/test-repo", repo_type="dataset", exist_ok=False)

    def test_create_repo_types(self, mock_api):
        """Test creating repositories of different types."""
        for repo_type in ["model", "dataset", "space"]:
            url = mock_api.create_repo(f"test-org/test-{repo_type}", repo_type=repo_type)
            assert repo_type in url


class TestMockHfApiUploadFile:
    """Tests for MockHfApi.upload_file method."""

    def test_upload_file_from_path(self, mock_api, tmp_path):
        """Test uploading a file from local path."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        url = mock_api.upload_file(
            test_file, "data/test.txt", "test-org/test-repo", repo_type="dataset"
        )
        assert "file://" in url

        # Verify file was uploaded
        repo_path = mock_api._get_repo_path("test-org/test-repo", "dataset")
        uploaded_file = repo_path / "data" / "test.txt"
        assert uploaded_file.exists()
        assert uploaded_file.read_text() == "test content"

    def test_upload_file_from_fileobj(self, mock_api):
        """Test uploading a file from file object."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")

        file_obj = io.BytesIO(b"test content from fileobj")
        mock_api.upload_file(
            file_obj, "data/test.txt", "test-org/test-repo", repo_type="dataset"
        )

        # Verify file was uploaded
        repo_path = mock_api._get_repo_path("test-org/test-repo", "dataset")
        uploaded_file = repo_path / "data" / "test.txt"
        assert uploaded_file.exists()
        assert uploaded_file.read_text() == "test content from fileobj"

    def test_upload_file_repo_not_found(self, mock_api, tmp_path):
        """Test uploading to non-existent repository raises error."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with pytest.raises(FileNotFoundError, match="not found"):
            mock_api.upload_file(
                test_file, "data/test.txt", "test-org/nonexistent", repo_type="dataset"
            )


class TestMockHfApiCreateCommit:
    """Tests for MockHfApi.create_commit method."""

    def test_create_commit_add_operations(self, mock_api, tmp_path):
        """Test creating commit with add operations."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")

        # Create test files
        file1 = tmp_path / "file1.txt"
        file1.write_text("content 1")
        file2 = tmp_path / "file2.txt"
        file2.write_text("content 2")

        operations = [
            CommitOperationAdd(path_in_repo="data/file1.txt", path_or_fileobj=str(file1)),
            CommitOperationAdd(path_in_repo="data/file2.txt", path_or_fileobj=str(file2)),
        ]

        url = mock_api.create_commit(
            "test-org/test-repo",
            operations,
            "Add test files",
            repo_type="dataset",
        )
        assert "file://" in url

        # Verify files were added
        repo_path = mock_api._get_repo_path("test-org/test-repo", "dataset")
        assert (repo_path / "data" / "file1.txt").read_text() == "content 1"
        assert (repo_path / "data" / "file2.txt").read_text() == "content 2"

    def test_create_commit_delete_operations(self, mock_api, tmp_path):
        """Test creating commit with delete operations."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")

        # First upload a file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        mock_api.upload_file(
            test_file, "data/test.txt", "test-org/test-repo", repo_type="dataset"
        )

        # Now delete it
        operations = [CommitOperationDelete(path_in_repo="data/test.txt")]
        mock_api.create_commit(
            "test-org/test-repo",
            operations,
            "Delete test file",
            repo_type="dataset",
        )

        # Verify file was deleted
        repo_path = mock_api._get_repo_path("test-org/test-repo", "dataset")
        assert not (repo_path / "data" / "test.txt").exists()

    def test_create_commit_mixed_operations(self, mock_api, tmp_path):
        """Test creating commit with mixed add/delete operations."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")

        # Upload initial file
        file1 = tmp_path / "file1.txt"
        file1.write_text("content 1")
        mock_api.upload_file(
            file1, "data/file1.txt", "test-org/test-repo", repo_type="dataset"
        )

        # Create commit with add and delete
        file2 = tmp_path / "file2.txt"
        file2.write_text("content 2")
        operations = [
            CommitOperationDelete(path_in_repo="data/file1.txt"),
            CommitOperationAdd(path_in_repo="data/file2.txt", path_or_fileobj=str(file2)),
        ]
        mock_api.create_commit(
            "test-org/test-repo",
            operations,
            "Replace file1 with file2",
            repo_type="dataset",
        )

        # Verify changes
        repo_path = mock_api._get_repo_path("test-org/test-repo", "dataset")
        assert not (repo_path / "data" / "file1.txt").exists()
        assert (repo_path / "data" / "file2.txt").read_text() == "content 2"

    def test_create_commit_with_fileobj(self, mock_api):
        """Test creating commit with file objects."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")

        file_obj = io.BytesIO(b"test content")
        operations = [
            CommitOperationAdd(path_in_repo="data/test.txt", path_or_fileobj=file_obj)
        ]
        mock_api.create_commit(
            "test-org/test-repo",
            operations,
            "Add file from fileobj",
            repo_type="dataset",
        )

        # Verify file was added
        repo_path = mock_api._get_repo_path("test-org/test-repo", "dataset")
        assert (repo_path / "data" / "test.txt").read_text() == "test content"

    def test_create_commit_repo_not_found(self, mock_api, tmp_path):
        """Test creating commit for non-existent repository raises error."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("content 1")
        operations = [
            CommitOperationAdd(path_in_repo="data/file1.txt", path_or_fileobj=str(file1))
        ]

        with pytest.raises(FileNotFoundError, match="not found"):
            mock_api.create_commit(
                "test-org/nonexistent",
                operations,
                "Test commit",
                repo_type="dataset",
            )


class TestMockHfApiHfHubDownload:
    """Tests for MockHfApi.hf_hub_download method."""

    def test_hf_hub_download_success(self, mock_api, tmp_path):
        """Test downloading a file from repository."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")

        # Upload a file first
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        mock_api.upload_file(
            test_file, "data/test.txt", "test-org/test-repo", repo_type="dataset"
        )

        # Download the file
        downloaded_path = mock_api.hf_hub_download(
            "test-org/test-repo", "data/test.txt", repo_type="dataset"
        )

        assert Path(downloaded_path).exists()
        assert Path(downloaded_path).read_text() == "test content"

    def test_hf_hub_download_repo_not_found(self, mock_api):
        """Test downloading from non-existent repository raises error."""
        with pytest.raises(FileNotFoundError, match="not found"):
            mock_api.hf_hub_download(
                "test-org/nonexistent", "data/test.txt", repo_type="dataset"
            )

    def test_hf_hub_download_file_not_found(self, mock_api):
        """Test downloading non-existent file raises error."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")

        with pytest.raises(FileNotFoundError, match="not found"):
            mock_api.hf_hub_download(
                "test-org/test-repo", "nonexistent.txt", repo_type="dataset"
            )


class TestMockHfApiListRepoFiles:
    """Tests for MockHfApi.list_repo_files method."""

    def test_list_repo_files_empty(self, mock_api):
        """Test listing files in empty repository."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")
        files = mock_api.list_repo_files("test-org/test-repo", repo_type="dataset")
        assert files == []

    def test_list_repo_files_with_files(self, mock_api, tmp_path):
        """Test listing files in repository with files."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")

        # Upload multiple files
        for i in range(3):
            test_file = tmp_path / f"test{i}.txt"
            test_file.write_text(f"content {i}")
            mock_api.upload_file(
                test_file,
                f"data/test{i}.txt",
                "test-org/test-repo",
                repo_type="dataset",
            )

        files = mock_api.list_repo_files("test-org/test-repo", repo_type="dataset")
        assert len(files) == 3
        assert "data/test0.txt" in files
        assert "data/test1.txt" in files
        assert "data/test2.txt" in files

    def test_list_repo_files_excludes_metadata(self, mock_api, tmp_path):
        """Test that list_repo_files excludes .hf_ metadata files."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        mock_api.upload_file(
            test_file, "test.txt", "test-org/test-repo", repo_type="dataset"
        )

        files = mock_api.list_repo_files("test-org/test-repo", repo_type="dataset")
        # Should only include test.txt, not .hf_metadata.json
        assert files == ["test.txt"]


class TestMockHfApiRepoInfo:
    """Tests for MockHfApi.repo_info method."""

    def test_repo_info_with_metadata(self, mock_api):
        """Test getting repository info with metadata."""
        mock_api.create_repo(
            "test-org/test-repo",
            repo_type="dataset",
            private=True,
            space_sdk="docker",
        )

        info = mock_api.repo_info("test-org/test-repo", repo_type="dataset")
        assert info["repo_id"] == "test-org/test-repo"
        assert info["repo_type"] == "dataset"
        assert info["private"] is True
        assert info["space_sdk"] == "docker"

    def test_repo_info_without_metadata(self, mock_api):
        """Test getting repository info without metadata file."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")
        repo_path = mock_api._get_repo_path("test-org/test-repo", "dataset")
        # Remove metadata file
        (repo_path / ".hf_metadata.json").unlink()

        info = mock_api.repo_info("test-org/test-repo", repo_type="dataset")
        assert info["repo_id"] == "test-org/test-repo"
        assert info["repo_type"] == "dataset"

    def test_repo_info_not_found(self, mock_api):
        """Test getting info for non-existent repository raises error."""
        with pytest.raises(FileNotFoundError, match="not found"):
            mock_api.repo_info("test-org/nonexistent", repo_type="dataset")


class TestMockHfApiDeleteRepo:
    """Tests for MockHfApi.delete_repo method."""

    def test_delete_repo_success(self, mock_api):
        """Test deleting a repository."""
        mock_api.create_repo("test-org/test-repo", repo_type="dataset")
        repo_path = mock_api._get_repo_path("test-org/test-repo", "dataset")
        assert repo_path.exists()

        mock_api.delete_repo("test-org/test-repo", repo_type="dataset")
        assert not repo_path.exists()

    def test_delete_repo_not_found(self, mock_api):
        """Test deleting non-existent repository raises error."""
        with pytest.raises(FileNotFoundError, match="not found"):
            mock_api.delete_repo("test-org/nonexistent", repo_type="dataset")


class TestHfHubDownloadStandalone:
    """Tests for standalone hf_hub_download function."""

    def test_standalone_function(self, mock_hf_cache, tmp_path):
        """Test standalone hf_hub_download function."""
        # Create a repo using MockHfApi
        api = MockHfApi(cache_dir=mock_hf_cache)
        api.create_repo("test-org/test-repo", repo_type="dataset")

        # Upload a file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        api.upload_file(
            test_file, "data/test.txt", "test-org/test-repo", repo_type="dataset"
        )

        # Download using standalone function
        downloaded_path = hf_hub_download(
            "test-org/test-repo",
            "data/test.txt",
            repo_type="dataset",
            cache_dir=mock_hf_cache,
        )

        assert Path(downloaded_path).exists()
        assert Path(downloaded_path).read_text() == "test content"


class TestCatalogIntegration:
    """Integration tests simulating usage patterns from catalog.py."""

    def test_catalog_workflow(self, mock_api, tmp_path):
        """Test workflow similar to RemoteCatalog usage in catalog.py."""
        # Simulate RemoteCatalog._init() - line 1385
        repo_id = "test-org/test-catalog"
        repo_type = "dataset"

        mock_api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            space_sdk=None,  # Not a space
            exist_ok=False,
        )

        # Verify repo was created
        assert mock_api._get_repo_path(repo_id, repo_type).exists()

        # Simulate RemoteCatalog._commit() - line 1429
        # Create some test files to commit
        file1 = tmp_path / "metadata.json"
        file1.write_text('{"version": "1.0"}')
        file2 = tmp_path / "schema.json"
        file2.write_text('{"fields": []}')

        operations = [
            CommitOperationAdd(
                path_in_repo="metadata.json", path_or_fileobj=str(file1)
            ),
            CommitOperationAdd(path_in_repo="schema.json", path_or_fileobj=str(file2)),
        ]

        commit_url = mock_api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message="Sync catalog metadata",
        )

        assert "file://" in commit_url

        # Simulate RemoteCatalog._checkout() - line 1451
        local_path = mock_api.hf_hub_download(
            repo_id=repo_id, filename="metadata.json", repo_type=repo_type
        )

        assert Path(local_path).exists()
        assert Path(local_path).read_text() == '{"version": "1.0"}'

    def test_catalog_update_workflow(self, mock_api, tmp_path):
        """Test updating files in catalog."""
        repo_id = "test-org/test-catalog"
        repo_type = "dataset"

        # Create repo
        mock_api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=False)

        # Initial commit
        file1 = tmp_path / "config.json"
        file1.write_text('{"version": "1.0"}')
        operations = [
            CommitOperationAdd(path_in_repo="config.json", path_or_fileobj=str(file1))
        ]
        mock_api.create_commit(
            repo_id, operations, "Initial commit", repo_type=repo_type
        )

        # Update commit (delete old, add new)
        file2 = tmp_path / "config_v2.json"
        file2.write_text('{"version": "2.0"}')
        operations = [
            CommitOperationDelete(path_in_repo="config.json"),
            CommitOperationAdd(
                path_in_repo="config_v2.json", path_or_fileobj=str(file2)
            ),
        ]
        mock_api.create_commit(
            repo_id, operations, "Update to v2", repo_type=repo_type
        )

        # Verify
        files = mock_api.list_repo_files(repo_id, repo_type=repo_type)
        assert "config.json" not in files
        assert "config_v2.json" in files

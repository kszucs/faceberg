"""Tests for custom FileIO implementations."""

import pytest


class TestHfFileIO:
    """Tests for HfFileIO custom FileIO implementation."""

    def test_hffileio_import(self):
        """Test that HfFileIO can be imported."""
        from faceberg.catalog import HfFileIO

        assert HfFileIO is not None

    def test_hffileio_initialization(self):
        """Test that HfFileIO can be initialized with properties."""
        from faceberg.catalog import HfFileIO

        io = HfFileIO(
            properties={
                "hf.endpoint": "https://huggingface.co",
                "hf.token": "test_token",
            }
        )

        assert io is not None
        assert io.properties["hf.endpoint"] == "https://huggingface.co"
        assert io.properties["hf.token"] == "test_token"

    def test_hffileio_creates_hf_filesystem(self):
        """Test that HfFileIO creates HfFileSystem for hf:// scheme."""
        from huggingface_hub import HfFileSystem

        from faceberg.catalog import HfFileIO

        io = HfFileIO(properties={"hf.endpoint": "https://huggingface.co"})
        fs = io.get_fs("hf")

        assert isinstance(fs, HfFileSystem)

    def test_hffileio_uses_skip_instance_cache(self):
        """Test that HfFileIO creates multiple distinct HfFileSystem instances.

        When skip_instance_cache=True, each call to get_fs('hf') should create
        a new HfFileSystem instance (after cache eviction). This test verifies
        that our custom factory uses skip_instance_cache correctly.
        """
        from faceberg.catalog import HfFileIO

        io = HfFileIO(properties={"hf.endpoint": "https://huggingface.co"})

        # First call creates and caches filesystem
        fs1 = io.get_fs("hf")

        # Clear the thread-local cache to force recreation
        io._thread_locals.get_fs_cached.cache_clear()

        # Second call should create a new instance (not from HfFileSystem's global cache)
        fs2 = io.get_fs("hf")

        # With skip_instance_cache=True, these should be different instances
        # (Without it, HfFileSystem would return the same cached instance)
        assert fs1 is not fs2, (
            "Expected different HfFileSystem instances with skip_instance_cache=True"
        )

    def test_hffileio_extends_fsspec_fileio(self):
        """Test that HfFileIO properly extends FsspecFileIO."""
        from pyiceberg.io.fsspec import FsspecFileIO

        from faceberg.catalog import HfFileIO

        io = HfFileIO(properties={})

        assert isinstance(io, FsspecFileIO)
        # Should have all standard FileIO methods
        assert hasattr(io, "new_input")
        assert hasattr(io, "new_output")
        assert hasattr(io, "delete")
        assert hasattr(io, "get_fs")

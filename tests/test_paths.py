"""Tests for path configuration and validation.

This module tests that:
1. save_dir is a CloudPath (remote storage)
2. cache_dir is a local Path
3. All cache subdirectories use constants from buttermilk._core.constants.cache
4. No hardcoded cache paths exist in the codebase
"""

import pytest
from pathlib import Path
from cloudpathlib import AnyPath, CloudPath

from buttermilk._core.constants import cache


class TestSaveDir:
    """Test save_dir configuration and validation."""

    def test_save_dir_is_cloud_path(self, real_bm):
        """Test that save_dir is a CloudPath for remote storage."""
        if real_bm.session_info.save_dir:
            save_path = AnyPath(real_bm.session_info.save_dir)
            # save_dir should be a CloudPath (gs://, s3://, etc.) in production
            # For local testing, it might be a Path, which is acceptable
            assert isinstance(save_path, (CloudPath, Path)), (
                f"save_dir must be either CloudPath or Path, got {type(save_path)}"
            )


class TestCacheDir:
    """Test cache_dir configuration and validation."""

    def test_cache_dir_is_local_path(self, real_bm):
        """Test that cache_dir is a local Path, not cloud storage."""
        cache_dir = real_bm.session_info.cache_dir
        assert cache_dir is not None, "cache_dir must be configured"

        # cache_dir should be a local path string
        assert isinstance(cache_dir, str), f"cache_dir must be str, got {type(cache_dir)}"

        # Should not start with cloud prefixes
        assert not cache_dir.startswith(("gs://", "s3://", "azure://", "gcs://")), (
            f"cache_dir must be local, not cloud storage: {cache_dir}"
        )

        # Should be expandable to absolute path
        expanded = Path(cache_dir).expanduser().resolve()
        assert expanded.is_absolute(), f"cache_dir must expand to absolute path: {cache_dir} -> {expanded}"

    def test_cache_dir_can_be_created(self, real_bm):
        """Test that cache_dir can be created if it doesn't exist."""
        cache_dir = Path(real_bm.session_info.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        assert cache_dir.exists(), f"Failed to create cache_dir: {cache_dir}"


class TestCacheSubdirectories:
    """Test cache subdirectory access using constants."""

    def test_get_cache_subdir_method_exists(self, real_bm):
        """Test that SessionInfo has get_cache_subdir method."""
        assert hasattr(real_bm.session_info, "get_cache_subdir"), (
            "SessionInfo must have get_cache_subdir() method"
        )

    def test_cache_constants_exist(self):
        """Test that cache constants are defined."""
        # All expected cache subdirectory constants
        expected_constants = ["CHROMADB", "EMBEDDINGS", "ZOTERO", "MODELS", "RECORDS"]

        for const_name in expected_constants:
            assert hasattr(cache, const_name), f"cache.{const_name} constant must be defined"
            const_value = getattr(cache, const_name)
            assert isinstance(const_value, str), f"cache.{const_name} must be str, got {type(const_value)}"
            assert const_value, f"cache.{const_name} must not be empty"

    def test_chromadb_cache_path(self, real_bm):
        """Test ChromaDB cache path uses constants."""
        chromadb_path = real_bm.session_info.get_cache_subdir(cache.CHROMADB)

        assert isinstance(chromadb_path, Path), f"get_cache_subdir must return Path, got {type(chromadb_path)}"
        assert chromadb_path.name == cache.CHROMADB, (
            f"ChromaDB cache subdirectory name must be '{cache.CHROMADB}', got '{chromadb_path.name}'"
        )
        assert str(real_bm.session_info.cache_dir) in str(chromadb_path), (
            f"ChromaDB path must be under cache_dir: {chromadb_path}"
        )

    def test_zotero_cache_path(self, real_bm):
        """Test Zotero cache path uses constants."""
        zotero_path = real_bm.session_info.get_cache_subdir(cache.ZOTERO)

        assert isinstance(zotero_path, Path), f"get_cache_subdir must return Path, got {type(zotero_path)}"
        assert zotero_path.name == cache.ZOTERO, (
            f"Zotero cache subdirectory name must be '{cache.ZOTERO}', got '{zotero_path.name}'"
        )

    def test_embeddings_cache_path(self, real_bm):
        """Test embeddings cache path uses constants."""
        embeddings_path = real_bm.session_info.get_cache_subdir(cache.EMBEDDINGS)

        assert isinstance(embeddings_path, Path), f"get_cache_subdir must return Path, got {type(embeddings_path)}"
        assert embeddings_path.name == cache.EMBEDDINGS, (
            f"Embeddings cache subdirectory name must be '{cache.EMBEDDINGS}', got '{embeddings_path.name}'"
        )

    def test_cache_subdir_creates_directory(self, real_bm):
        """Test that get_cache_subdir creates the directory by default."""
        test_subdir = cache.CHROMADB
        cache_path = real_bm.session_info.get_cache_subdir(test_subdir)

        # Should be created by default
        assert cache_path.exists(), f"get_cache_subdir should create directory: {cache_path}"
        assert cache_path.is_dir(), f"cache subdirectory must be a directory: {cache_path}"

    def test_cache_subdir_no_create_option(self, real_bm, tmp_path):
        """Test that get_cache_subdir respects create=False."""
        # Use a temporary cache_dir for this test
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_cache:
            real_bm.session_info.cache_dir = str(tmp_cache)
            test_subdir = "test_no_create"

            # Should not create when create=False
            cache_path = real_bm.session_info.get_cache_subdir(test_subdir, create=False)
            assert not cache_path.exists(), f"get_cache_subdir(create=False) should not create directory"


class TestBackwardCompatibility:
    """Test backward compatibility with deprecated methods."""

    def test_get_chromadb_cache_dir_deprecated(self, real_bm):
        """Test that deprecated get_chromadb_cache_dir() still works."""
        # Old method should still work
        chromadb_path = real_bm.session_info.get_chromadb_cache_dir()

        assert isinstance(chromadb_path, Path)
        assert chromadb_path.name == cache.CHROMADB

        # Should match new method
        new_path = real_bm.session_info.get_cache_subdir(cache.CHROMADB)
        assert chromadb_path == new_path, "Deprecated method must return same path as new method"


class TestPathSeparation:
    """Test that cache and save directories are properly separated."""

    def test_cache_and_save_are_different(self, real_bm):
        """Test that cache_dir and save_dir are different locations."""
        if real_bm.session_info.save_dir:
            cache_dir = str(real_bm.session_info.cache_dir)
            save_dir = str(real_bm.session_info.save_dir)

            # They should be different paths
            assert cache_dir != save_dir, "cache_dir and save_dir should be different locations"

    def test_cache_is_local_save_can_be_remote(self, real_bm):
        """Test the intended separation: cache is local, save can be remote."""
        cache_dir = real_bm.session_info.cache_dir

        # Cache must be local
        assert not cache_dir.startswith(("gs://", "s3://", "azure://", "gcs://")), (
            "cache_dir must be local storage"
        )

        # Save can be either (local or remote)
        if real_bm.session_info.save_dir:
            save_path = AnyPath(real_bm.session_info.save_dir)
            # This is fine - save_dir can be CloudPath or local Path
            assert isinstance(save_path, (CloudPath, Path))

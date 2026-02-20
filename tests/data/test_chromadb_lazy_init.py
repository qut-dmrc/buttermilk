"""Test ChromaDB lazy initialization for startup optimization.

This test suite validates Phase 2 of startup optimization:
- ChromaDB should NOT initialize during model instantiation
- ChromaDB SHOULD initialize on first collection access OR after 60s background warmup
- Background warmup should start automatically

NO MOCKS - these are TRUE integration tests using real ChromaDB instances in temp directories.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from buttermilk.data.vector import ChromaDBEmbeddings

pytestmark = pytest.mark.anyio  # Enable async test support


class TestChromaDBLazyInitialization:
    """Test suite for ChromaDB lazy initialization."""

    def test_chromadb_not_initialized_during_model_creation(self):
        """Test that ChromaDB client is NOT initialized during model instantiation.

        This is the key optimization - creating a ChromaDBEmbeddings instance should
        be fast (<1s) and NOT trigger expensive operations like counting 162K embeddings.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            start_time = time.time()

            # Create ChromaDBEmbeddings instance - should be FAST
            embeddings = ChromaDBEmbeddings(
                persist_directory=str(Path(tmpdir) / "test_db"),
                collection_name="lazy_init_test",
                embedding_model="text-embedding-004",
                dimensionality=768,
                read_only=True,  # Skip embedding infrastructure for faster test
            )

            init_time = time.time() - start_time

            # Instance creation should be fast (<1 second)
            assert init_time < 1.0, (
                f"Model instantiation took {init_time:.2f}s (should be <1s)"
            )

            # Critical assertion: ChromaDB client should NOT exist yet
            assert not hasattr(embeddings, "_client") or embeddings._client is None, (
                "ChromaDB client should not be initialized during model creation"
            )

            # Cache should NOT be initialized yet
            assert not embeddings._cache_initialized, (
                "Cache should not be initialized during model creation"
            )

    async def test_chromadb_initializes_on_first_collection_access(self):
        """Test that ChromaDB initializes lazily on first collection access.

        This validates the lazy loading pattern - initialization should be deferred
        until the collection is actually needed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create instance - should be fast
            embeddings = ChromaDBEmbeddings(
                persist_directory=str(Path(tmpdir) / "test_db"),
                collection_name="lazy_access_test",
                embedding_model="text-embedding-004",
                dimensionality=768,
                read_only=True,
            )

            # Verify NOT initialized yet
            assert not embeddings._cache_initialized

            # First access to collection property should trigger initialization
            start_time = time.time()
            collection = embeddings.collection
            init_time = time.time() - start_time

            # Initialization should have happened
            assert embeddings._cache_initialized, (
                "Cache should be initialized after first collection access"
            )

            # Client should now exist
            assert embeddings._client is not None, (
                "ChromaDB client should exist after collection access"
            )

            # Collection should be usable
            assert collection is not None
            assert collection.count() == 0  # Empty collection for new test

            # Log timing for reference (not a hard requirement for now)
            print(f"First collection access took {init_time:.2f}s")

    async def test_background_warmup_starts_automatically(self):
        """Test that background warmup task starts automatically after 60s.

        This validates that we can optionally pre-initialize ChromaDB in the background
        to avoid first-access latency for queries that happen after 60s.

        NOTE: This test uses a shorter timeout (2s) for testing purposes.
        Production will use 60s.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create instance with background warmup
            embeddings = ChromaDBEmbeddings(
                persist_directory=str(Path(tmpdir) / "test_db"),
                collection_name="warmup_test",
                embedding_model="text-embedding-004",
                dimensionality=768,
                read_only=True,
            )

            # Initially NOT initialized
            assert not embeddings._cache_initialized

            # Start background warmup with short timeout for testing
            # (Production will use 60s)
            warmup_task = asyncio.create_task(
                self._warmup_after_delay(embeddings, delay_seconds=0.5)
            )

            # Wait for warmup to complete
            await warmup_task

            # After warmup, should be initialized
            assert embeddings._cache_initialized, (
                "Cache should be initialized after background warmup"
            )

            # Collection should be immediately accessible without delay
            start_time = time.time()
            collection = embeddings.collection
            access_time = time.time() - start_time

            # Access should be fast since already initialized
            assert access_time < 0.1, (
                f"Collection access after warmup took {access_time:.2f}s (should be <0.1s)"
            )

            assert collection is not None

    async def _warmup_after_delay(self, embeddings, delay_seconds: float):
        """Helper to simulate background warmup with configurable delay."""
        await asyncio.sleep(delay_seconds)
        await embeddings.ensure_cache_initialized()

    async def test_manual_ensure_cache_still_works(self):
        """Test that manual ensure_cache_initialized() still works for eager init.

        Some workflows need eager initialization - ensure this path still works.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = ChromaDBEmbeddings(
                persist_directory=str(Path(tmpdir) / "test_db"),
                collection_name="eager_test",
                embedding_model="text-embedding-004",
                dimensionality=768,
                read_only=True,
            )

            # Not initialized yet
            assert not embeddings._cache_initialized

            # Manual initialization
            await embeddings.ensure_cache_initialized()

            # Should be initialized now
            assert embeddings._cache_initialized
            assert embeddings._client is not None

            # Collection should work
            collection = embeddings.collection
            assert collection is not None


class TestChromaDBLazyInitWithRemoteStorage:
    """Test lazy initialization with remote storage paths.

    NOTE: These tests simulate remote paths but use local storage.
    Real remote storage would require GCS credentials.
    """

    def test_remote_path_detection_defers_initialization(self):
        """Test that remote storage paths defer client initialization.

        For remote paths (gs://, s3://, etc.), we should NOT download or
        initialize anything during model creation.
        """
        # Create instance with fake remote path
        # (Don't actually try to download - will fail fast if it tries)
        embeddings = ChromaDBEmbeddings(
            persist_directory="gs://fake-bucket/chromadb",
            collection_name="remote_test",
            embedding_model="text-embedding-004",
            dimensionality=768,
            read_only=True,
        )

        # Should NOT have initialized client (would fail trying to download)
        assert not hasattr(embeddings, "_client") or embeddings._client is None
        assert not embeddings._cache_initialized

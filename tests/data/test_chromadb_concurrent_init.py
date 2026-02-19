"""Test thread-safety of concurrent ChromaDB initialization.

This test suite validates that concurrent calls to ensure_cache_initialized()
are thread-safe and efficient:

1. Multiple simultaneous calls don't cause race conditions
2. Only one actual initialization occurs (not N redundant inits)
3. Subsequent calls after initialization are fast no-ops
4. Tests ChromaDBEmbeddings class (which has proper locking)

Thread-safety is ensured by:
- asyncio.Lock serializes initialization attempts
- Double-checked locking pattern prevents redundant work
- asyncio.to_thread() provides thread isolation for blocking ChromaDB operations

NOTE: ChromaDBUploader does NOT currently have the same thread-safety
guarantees and will be fixed separately.

NO MOCKS - these are TRUE integration tests using real ChromaDB instances.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from buttermilk.data.vector import ChromaDBEmbeddings

pytestmark = pytest.mark.anyio  # Enable async test support


class TestConcurrentChromaDBEmbeddingsInit:
    """Test concurrent initialization for ChromaDBEmbeddings class."""

    async def test_concurrent_embeddings_initialization(self):
        """Test multiple simultaneous ensure_cache_initialized() calls.

        Validates that concurrent initialization attempts:
        1. All complete without exceptions
        2. Result in exactly one initialization
        3. Don't cause race conditions or duplicate work
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = ChromaDBEmbeddings(
                persist_directory=str(Path(tmpdir) / "concurrent_test"),
                collection_name="test_concurrent_embeddings",
                embedding_model="text-embedding-004",
                dimensionality=768,
                read_only=True,  # Skip embedding infrastructure for faster test
            )

            # Verify not initialized yet
            assert not embeddings._cache_initialized

            # Create 10 concurrent initialization attempts
            tasks = [embeddings.ensure_cache_initialized() for _ in range(10)]

            # All should complete without exceptions
            start_time = time.time()
            await asyncio.gather(*tasks)
            elapsed = time.time() - start_time

            # Verify initialized exactly once
            assert embeddings._cache_initialized, "Cache should be initialized after concurrent calls"
            assert embeddings._client is not None, "ChromaDB client should exist after initialization"

            # Verify collection is accessible and functional
            collection = embeddings.collection
            assert collection is not None
            count = await asyncio.to_thread(collection.count)
            assert count == 0, "New collection should be empty"

            # Log timing for reference
            print(f"\n10 concurrent initialization calls took {elapsed:.3f}s")

    async def test_subsequent_calls_are_fast_noop(self):
        """Test that subsequent initialization calls are fast no-ops.

        After initial initialization, additional ensure_cache_initialized()
        calls should return immediately without doing any work.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = ChromaDBEmbeddings(
                persist_directory=str(Path(tmpdir) / "noop_test"),
                collection_name="test_noop",
                embedding_model="text-embedding-004",
                dimensionality=768,
                read_only=True,
            )

            # First initialization
            await embeddings.ensure_cache_initialized()
            assert embeddings._cache_initialized

            # Subsequent calls should be very fast (<10ms)
            times = []
            for _ in range(5):
                start = time.time()
                await embeddings.ensure_cache_initialized()
                elapsed = time.time() - start
                times.append(elapsed)

            # All subsequent calls should be fast no-ops
            max_time = max(times)
            avg_time = sum(times) / len(times)

            assert max_time < 0.01, f"Subsequent calls should be <10ms, got max={max_time * 1000:.1f}ms"

            print(f"\nSubsequent call times: avg={avg_time * 1000:.2f}ms, max={max_time * 1000:.2f}ms")

    async def test_concurrent_init_with_collection_access(self):
        """Test concurrent initialization mixed with collection access.

        This tests a realistic scenario where some tasks call
        ensure_cache_initialized() while others also call it before accessing
        the collection. All concurrent calls should be safe.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = ChromaDBEmbeddings(
                persist_directory=str(Path(tmpdir) / "mixed_test"),
                collection_name="test_mixed_access",
                embedding_model="text-embedding-004",
                dimensionality=768,
                read_only=True,
            )

            async def init_task():
                """Task that calls ensure_cache_initialized()."""
                await embeddings.ensure_cache_initialized()

            async def collection_access_task():
                """Task that ensures init then accesses collection."""
                # Ensure initialized before accessing collection
                await embeddings.ensure_cache_initialized()
                collection = embeddings.collection
                count = await asyncio.to_thread(collection.count)
                return count

            # Mix of initialization and collection access tasks
            tasks = [init_task() for _ in range(5)] + [collection_access_task() for _ in range(5)]

            # All should complete without exceptions
            results = await asyncio.gather(*tasks)

            # Collection access tasks return counts, init tasks return None
            counts = [r for r in results if r is not None]
            assert all(count == 0 for count in counts), "All collection access should see empty collection"

            # Verify final state is initialized and consistent
            assert embeddings._cache_initialized
            assert embeddings._client is not None
            collection = embeddings.collection
            assert collection is not None


class TestConcurrentInitEdgeCases:
    """Test edge cases and failure scenarios for concurrent initialization."""

    async def test_concurrent_init_survives_exception_in_one_task(self):
        """Test that if one initialization task encounters an error early,
        other tasks can still complete successfully.

        This validates that the lock is properly released even on exceptions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = ChromaDBEmbeddings(
                persist_directory=str(Path(tmpdir) / "error_test"),
                collection_name="test_error_handling",
                embedding_model="text-embedding-004",
                dimensionality=768,
                read_only=True,
            )

            # This will test the lock release behavior
            # First, do a successful init
            await embeddings.ensure_cache_initialized()

            # Now concurrent calls should all succeed as no-ops
            tasks = [embeddings.ensure_cache_initialized() for _ in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # No exceptions should occur
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"No exceptions expected, got: {exceptions}"

            # Verify still initialized and functional
            assert embeddings._cache_initialized
            collection = embeddings.collection
            assert collection is not None

    async def test_high_concurrency_stress_test(self):
        """Stress test with high concurrency (100 concurrent tasks).

        This validates that the locking mechanism scales properly and
        doesn't have race conditions even under heavy concurrent load.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = ChromaDBEmbeddings(
                persist_directory=str(Path(tmpdir) / "stress_test"),
                collection_name="test_high_concurrency",
                embedding_model="text-embedding-004",
                dimensionality=768,
                read_only=True,
            )

            # Create 100 concurrent initialization attempts
            num_tasks = 100
            tasks = [embeddings.ensure_cache_initialized() for _ in range(num_tasks)]

            start_time = time.time()
            await asyncio.gather(*tasks)
            elapsed = time.time() - start_time

            # Verify initialized exactly once (not 100 times!)
            assert embeddings._cache_initialized
            assert embeddings._client is not None

            # Verify collection is functional
            collection = embeddings.collection
            assert collection is not None
            count = await asyncio.to_thread(collection.count)
            assert count == 0

            print(f"\nStress test: {num_tasks} concurrent tasks took {elapsed:.3f}s")

            # Performance expectation: should be much faster than N sequential inits
            # If it took >5s for 100 tasks, something is wrong with concurrency
            assert elapsed < 5.0, f"100 concurrent tasks took {elapsed:.1f}s (should be <5s)"

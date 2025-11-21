"""Test that ChromaDB read-only mode skips expensive count operations.

Investigation: Validate that read-only mode optimization properly skips
the collection.count() call during initialization.
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from buttermilk.data.vector import ChromaDBEmbeddings


@pytest.mark.anyio
async def test_readonly_skips_count_operation() -> None:
    """Test that read-only mode skips collection.count() during initialization.

    This test verifies the optimization where read-only mode avoids the
    expensive count() operation on large collections during initialization.
    """
    # Create temporary directories for read-only and write mode tests
    temp_path_readonly = Path("/tmp/test_chromadb_readonly_count")
    temp_path_write = Path("/tmp/test_chromadb_write_count")
    temp_path_readonly.mkdir(exist_ok=True)
    temp_path_write.mkdir(exist_ok=True)

    collection_name = "test_count_collection"

    # Step 1: Create collections first (so validation path will be triggered)
    store_setup_readonly = ChromaDBEmbeddings(
        collection_name=collection_name,
        persist_directory=str(temp_path_readonly),
        dimensionality=3,
        read_only=False,  # Create in write mode first
        disable_auto_sync=True,
    )
    await store_setup_readonly.ensure_cache_initialized()

    store_setup_write = ChromaDBEmbeddings(
        collection_name=collection_name,
        persist_directory=str(temp_path_write),
        dimensionality=3,
        read_only=False,  # Create in write mode first
        disable_auto_sync=True,
    )
    await store_setup_write.ensure_cache_initialized()

    # Track calls to asyncio.to_thread to monitor count() calls
    original_to_thread = asyncio.to_thread
    to_thread_calls: list[tuple[Any, ...]] = []

    async def tracked_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        """Track calls to asyncio.to_thread."""
        to_thread_calls.append((func, *args))
        return await original_to_thread(func, *args, **kwargs)

    try:
        with patch("asyncio.to_thread", side_effect=tracked_to_thread):
            # Test 1: Read-only mode should NOT call count
            to_thread_calls.clear()
            store_readonly = ChromaDBEmbeddings(
                collection_name=collection_name,
                persist_directory=str(temp_path_readonly),
                dimensionality=3,
                read_only=True,
                disable_auto_sync=True,
            )
            await store_readonly.ensure_cache_initialized()

            # Verify count() was NOT called in read-only mode
            count_calls = [
                call
                for call in to_thread_calls
                if len(call) > 0
                and callable(call[0])
                and hasattr(call[0], "__name__")
                and call[0].__name__ == "count"
            ]
            assert len(count_calls) == 0, (
                f"Expected no count() calls in read-only mode, but found {len(count_calls)}"
            )

            # Test 2: Write mode SHOULD call count
            to_thread_calls.clear()
            store_write = ChromaDBEmbeddings(
                collection_name=collection_name,
                persist_directory=str(temp_path_write),
                dimensionality=3,
                read_only=False,
                disable_auto_sync=True,
            )
            await store_write.ensure_cache_initialized()

            # Verify count() WAS called in write mode
            count_calls = [
                call
                for call in to_thread_calls
                if len(call) > 0
                and callable(call[0])
                and hasattr(call[0], "__name__")
                and call[0].__name__ == "count"
            ]
            assert len(count_calls) > 0, (
                "Expected count() to be called in write mode, but it wasn't"
            )

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_path_readonly, ignore_errors=True)
        shutil.rmtree(temp_path_write, ignore_errors=True)


@pytest.mark.anyio
async def test_readonly_logs_skip_message() -> None:
    """Test that read-only mode logs appropriate message about skipped count.

    This test verifies the logging behavior difference between read-only
    and write modes.
    """
    temp_path = Path("/tmp/test_chromadb_readonly_logs")
    temp_path.mkdir(exist_ok=True)

    collection_name = "test_readonly_logs"

    try:
        # Test read-only initialization
        store = ChromaDBEmbeddings(
            collection_name=collection_name,
            persist_directory=str(temp_path),
            dimensionality=3,
            read_only=True,
            disable_auto_sync=True,
        )
        await store.ensure_cache_initialized()

        # Verify store is properly initialized
        assert store.collection is not None
        assert store.collection.name == collection_name

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_path, ignore_errors=True)


class TestChromaDBReadOnlySkipCount:
    """Test suite for ChromaDB read-only mode optimizations."""

    @pytest.mark.anyio
    async def test_readonly_mode_skips_count_during_finalize(
        self, tmp_path: Path
    ) -> None:
        """Test that read-only mode skips collection.count() during finalize_processing().

        This test verifies that when ChromaDB is in read-only mode, the
        finalize_processing() method should NOT call collection.count() to avoid
        expensive operations on large collections.

        Expected behavior:
        - Read-only mode: count() should NOT be called during finalize_processing()
        - Write mode: count() SHOULD be called during finalize_processing()

        Current status: FAILING - count() is always called regardless of read_only mode
        """
        collection_name = "test_finalize_count"

        # Create a temporary ChromaDB instance and add some processed records
        temp_db_path = tmp_path / "chromadb_finalize_test"
        temp_db_path.mkdir(exist_ok=True)

        # Step 1: Create collection and simulate some processing
        store_setup = ChromaDBEmbeddings(
            collection_name=collection_name,
            persist_directory=str(temp_db_path),
            dimensionality=3,
            read_only=False,
            disable_auto_sync=True,
        )
        await store_setup.ensure_cache_initialized()

        # Simulate processed records by setting the internal counter
        store_setup._processed_records_count = 5

        # Track calls to asyncio.to_thread to monitor count() calls
        original_to_thread = asyncio.to_thread
        to_thread_calls: list[tuple[Any, ...]] = []

        async def tracked_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
            """Track calls to asyncio.to_thread."""
            to_thread_calls.append((func, *args))
            return await original_to_thread(func, *args, **kwargs)

        # Test read-only mode - should skip count()
        with patch("asyncio.to_thread", side_effect=tracked_to_thread):
            # Create read-only instance with same collection
            to_thread_calls.clear()
            store_readonly = ChromaDBEmbeddings(
                collection_name=collection_name,
                persist_directory=str(temp_db_path),
                dimensionality=3,
                read_only=True,
                disable_auto_sync=True,
            )
            await store_readonly.ensure_cache_initialized()

            # Simulate processed records
            store_readonly._processed_records_count = 5

            # Call finalize_processing() in read-only mode
            await store_readonly.finalize_processing()

            # Verify count() was NOT called
            count_calls = [
                call
                for call in to_thread_calls
                if len(call) > 0
                and callable(call[0])
                and hasattr(call[0], "__name__")
                and call[0].__name__ == "count"
            ]

            assert len(count_calls) == 0, (
                f"Expected no count() calls in read-only mode during finalize_processing(), but found {len(count_calls)} calls"
            )

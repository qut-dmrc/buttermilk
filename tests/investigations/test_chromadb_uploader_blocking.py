"""Test ChromaDBUploader for event loop blocking during initialization.

This investigative test verifies that ChromaDBUploader._ensure_cache_initialized()
doesn't block the event loop during:
1. chromadb.PersistentClient() initialization (line 185)
2. client.get_collection() (line 191)
3. client.create_collection() (line 199)

Uses heartbeat pattern to detect blocking. NO MOCKS - uses REAL ChromaDB.

Expected behavior:
- BEFORE fixes: Test should FAIL with max_heartbeat_gap > 200ms (on slower systems)
- AFTER fixes: Test should PASS with max_heartbeat_gap < 200ms

Note: On fast systems with SSDs, individual operations may complete in <10ms,
making blocking hard to measure. The test uses 10 iterations to accumulate
blocking time. Check the "Blocking %" metric - high percentage indicates
unwrapped synchronous operations even if max gap is below threshold.

Current status (as of 2025-11-12):
- Lines 185, 191, 199 are NOT wrapped with asyncio.to_thread()
- Test passes on fast dev system but shows 82.6% blocking
- Will fail in production with slower disks/larger databases
"""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from buttermilk.processors.chromadb_uploader import ChromaDBUploader

pytestmark = pytest.mark.anyio


class TestChromaDBUploaderBlocking:
    """Test ChromaDBUploader for event loop blocking."""

    @pytest.mark.slow
    async def test_ensure_cache_initialized_blocking(self):
        """Verify _ensure_cache_initialized doesn't block event loop.

        This test measures blocking during ChromaDBUploader initialization which calls:
        - chromadb.PersistentClient() (line 185)
        - client.get_collection() (line 191)
        - client.create_collection() (line 199)

        Uses heartbeat task to detect event loop blocking.
        To make blocking measurable, we test multiple initialization cycles.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\nSetup: Using temp directory {tmpdir}")

            # Heartbeat monitoring variables
            heartbeat_count = 0
            max_heartbeat_gap = 0.0
            last_heartbeat_time = time.perf_counter()

            async def heartbeat():
                """Monitor event loop responsiveness."""
                nonlocal heartbeat_count, max_heartbeat_gap, last_heartbeat_time
                while True:
                    current_time = time.perf_counter()
                    gap = current_time - last_heartbeat_time
                    max_heartbeat_gap = max(max_heartbeat_gap, gap)
                    last_heartbeat_time = current_time
                    heartbeat_count += 1
                    await asyncio.sleep(0.01)  # 10ms heartbeat

            # Start heartbeat monitoring
            heartbeat_task = asyncio.create_task(heartbeat())
            await asyncio.sleep(0.05)  # Let heartbeat start
            start_time = time.perf_counter()

            # Execute multiple initialization cycles to make blocking measurable
            # Single operations may be too fast (<2ms) to reliably detect blocking
            iterations = 10
            print(f"Testing {iterations} initialization cycles...")

            for i in range(iterations):
                db_path = str(Path(tmpdir) / f"test_chromadb_{i}")

                # Create new uploader instance for each iteration
                uploader = ChromaDBUploader(
                    collection_name=f"blocking_test_{i}",
                    persist_directory=db_path,
                    sync_batch_size=50,
                    sync_interval_minutes=10,
                )

                # This calls UNWRAPPED blocking operations:
                # 1. chromadb.PersistentClient() - BLOCKING (line 185)
                # 2. client.get_collection() - BLOCKING (line 191, will fail first time)
                # 3. client.create_collection() - BLOCKING (line 199)
                await uploader._ensure_cache_initialized()

                # Verify initialization succeeded
                assert uploader._cache_initialized
                assert uploader._client is not None
                assert uploader._collection is not None

            duration = time.perf_counter() - start_time

            # Stop heartbeat monitoring
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

            # Calculate metrics
            expected_beats = duration / 0.01
            blocking_pct = (1 - (heartbeat_count / expected_beats)) * 100 if expected_beats > 0 else 0
            max_gap_ms = max_heartbeat_gap * 1000
            duration_ms = duration * 1000

            # Report results
            print("\n" + "=" * 80)
            print(f"ChromaDBUploader._ensure_cache_initialized() x{iterations} Blocking Test")
            print("=" * 80)
            print(f"Duration:        {duration_ms:.2f}ms")
            print(f"Heartbeats:      {heartbeat_count} (expected ~{expected_beats:.0f})")
            print(f"Blocking:        {blocking_pct:.1f}%")
            print(f"Max gap:         {max_gap_ms:.2f}ms")
            print("=" * 80)

            # Analysis
            print("\nAnalysis:")
            if max_gap_ms > 200:
                print(f"  ❌ BLOCKING DETECTED: Max heartbeat gap is {max_gap_ms:.2f}ms (threshold: 200ms)")
                print("  Operations blocking the event loop:")
                print("    - Line 185: chromadb.PersistentClient() - NOT WRAPPED")
                print("    - Line 191: client.get_collection() - NOT WRAPPED")
                print("    - Line 199: client.create_collection() - NOT WRAPPED")
                print("\n  Solution: Wrap these operations with asyncio.to_thread()")
            else:
                print(f"  ✅ NON-BLOCKING: Max heartbeat gap is {max_gap_ms:.2f}ms (threshold: 200ms)")
                print("  All ChromaDB operations properly wrapped with asyncio.to_thread()")

            # Note about fast systems
            if max_gap_ms < 50 and duration_ms < 500:
                print("\n  NOTE: ChromaDB operations are very fast on this system.")
                print("  Individual operations complete in <5ms, too fast for reliable blocking detection.")
                print("  In production with slower disks/larger databases, these operations WILL block")
                print("  and MUST be wrapped with asyncio.to_thread().")

            # Assert non-blocking behavior
            # This will FAIL before fixes (showing >200ms blocking)
            # This will PASS after fixes (showing <200ms blocking)
            assert max_gap_ms < 200, (
                f"_ensure_cache_initialized blocks event loop for {max_gap_ms:.2f}ms "
                f"(should be <200ms). Unwrapped blocking operations detected:\n"
                f"  - Line 185: chromadb.PersistentClient() needs asyncio.to_thread()\n"
                f"  - Line 191: client.get_collection() needs asyncio.to_thread()\n"
                f"  - Line 199: client.create_collection() needs asyncio.to_thread()"
            )

            print("\nTest complete - ChromaDBUploader initialization is non-blocking ✅")

    async def test_ensure_cache_initialized_creates_collection(self):
        """Verify _ensure_cache_initialized properly creates collection.

        This is a secondary test to ensure the blocking fix doesn't break functionality.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_chromadb_uploader_creation")

            # Create uploader for new collection
            uploader = ChromaDBUploader(
                collection_name="new_collection_test",
                persist_directory=db_path,
            )

            # Initialize cache
            await uploader._ensure_cache_initialized()

            # Verify collection was created
            assert uploader._cache_initialized
            assert uploader._client is not None
            assert uploader._collection is not None

            # Verify collection name
            assert uploader._collection.name == "new_collection_test"

            # Verify we can use the collection (count should be 0 for new collection)
            count = await asyncio.to_thread(uploader._collection.count)
            assert count == 0

            print("\n✅ Collection creation verified")

    async def test_ensure_cache_initialized_gets_existing_collection(self):
        """Verify _ensure_cache_initialized properly gets existing collection.

        This ensures the blocking fix works for both get_collection() and create_collection() paths.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_chromadb_uploader_existing")

            # First uploader - creates collection
            uploader1 = ChromaDBUploader(
                collection_name="existing_collection_test",
                persist_directory=db_path,
            )
            await uploader1._ensure_cache_initialized()

            # Second uploader - should get existing collection
            uploader2 = ChromaDBUploader(
                collection_name="existing_collection_test",
                persist_directory=db_path,
            )
            await uploader2._ensure_cache_initialized()

            # Verify both use the same collection
            assert uploader1._collection.name == uploader2._collection.name
            assert uploader1._collection.name == "existing_collection_test"

            print("\n✅ Existing collection retrieval verified")

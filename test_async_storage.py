#!/usr/bin/env python3
"""Test script to verify Storage async iterator implementation works with PipelineOrchestrator."""

import asyncio
import sys
from pathlib import Path

# Add buttermilk to path
sys.path.insert(0, str(Path(__file__).parent))

from buttermilk._core.types import Record
from buttermilk.storage.file import FileStorage
from buttermilk._core.storage_config import StorageConfig


async def test_storage_async_iterator():
    """Test that Storage objects can be used as async iterators with the pipeline pattern."""

    # Create a simple test data file
    test_file = Path("/tmp/test_records.jsonl")
    test_data = [
        {"record_id": "1", "content": "First record", "dataset_name": "test", "split_type": "train"},
        {"record_id": "2", "content": "Second record", "dataset_name": "test", "split_type": "train"},
        {"record_id": "3", "content": "Third record", "dataset_name": "test", "split_type": "train"}
    ]

    # Write test data
    import json
    with open(test_file, 'w') as f:
        for record in test_data:
            json.dump(record, f)
            f.write('\n')

    try:
        # Create storage
        config = StorageConfig(
            type="file",
            path=str(test_file),
            dataset_name="test",
            split_type="train"
        )
        storage = FileStorage(config)

        print("Testing Storage as async iterator...")

        # Test 1: Direct async iteration (what pipeline expects)
        print("\n1. Testing hasattr(__anext__):", hasattr(storage, "__anext__"))
        print("2. Testing hasattr(__aiter__):", hasattr(storage, "__aiter__"))

        # Test 2: Async iteration pattern from pipeline.py:179
        source_iter = storage if hasattr(storage, "__anext__") else storage.__aiter__()
        print("3. Source iter type:", type(source_iter))

        # Test 3: Actual async iteration
        count = 0
        async for record in storage:
            print(f"4. Got record {count + 1}: {record.record_id} - {record.content}")
            count += 1
            if count >= 3:  # Safety limit
                break

        print(f"\n✅ Successfully processed {count} records via async iteration")

        # Test 4: Multiple iterations should work
        print("\n5. Testing second iteration...")
        count2 = 0
        async for record in storage:
            count2 += 1
            if count2 >= 2:  # Just test a couple
                break
        print(f"✅ Second iteration worked, got {count2} records")

    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    asyncio.run(test_storage_async_iterator())
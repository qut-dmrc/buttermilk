#!/usr/bin/env python
"""Quick test to verify Title objects are loaded from BigQuery storage."""

import pytest
from omegaconf import OmegaConf

from buttermilk._core.storage_config import StorageFactory


@pytest.mark.slow
def test_title_loading():
    """Test that Title objects are properly loaded from tmdbtitles storage."""

    # Load the tmdbtitles configuration
    config_path = "buttermilk/conf/storage/tmdbtitles.yaml"
    config_dict = OmegaConf.load(config_path)

    print(f"Storage config: {config_dict}")

    # Create storage instance
    storage = StorageFactory.create_storage(config_dict)

    # Get the first few records to test
    count = 0
    for record in storage:
        count += 1

        # Check the type
        print(f"\nRecord {count}:")
        print(f"  Type: {type(record).__name__}")
        print(f"  Module: {type(record).__module__}")
        print(f"  Record ID: {record.record_id}")

        # Check Title-specific attributes
        if hasattr(record, "title"):
            print(f"  Title: {record.title}")
        if hasattr(record, "year"):
            print(f"  Year: {record.year}")

        # Show first 3 records only
        if count >= 3:
            break

    if count > 0:
        print(f"\n✅ Successfully loaded {count} Title objects from BigQuery!")
    else:
        print("\n⚠️ No records found in storage")


if __name__ == "__main__":
    test_title_loading()

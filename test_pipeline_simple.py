#!/usr/bin/env python
"""Simple test to verify the pipeline works with TMDBTool and uploader."""

import asyncio
from buttermilk.tools.catalog_test import Title, TMDBTool
from buttermilk.utils.uploader import AsyncDataUploader
from unittest.mock import MagicMock, AsyncMock

async def test_pipeline():
    """Test that TMDBTool yields observations and uploader passes them through."""

    # Create a test Title
    title = Title(
        record_id="123",
        title="Test Movie",
        year=2024
    )

    # Create TMDBTool with mocked API
    tool = TMDBTool(api_key="fake_key", region="US")

    # Mock get_availability to return a simple async generator
    async def mock_get_availability(title):
        # Simulate no providers found - yields one null observation
        yield {
            "record_id": "123",
            "title": "Test Movie",
            "year": 2024,
            "region": None,
            "available": False,
            "source": "TMDB",
            "provider_name": None,
            "provider_id": None,
            "provider_type": None
        }

    tool.get_availability = mock_get_availability

    # Create mock uploader
    mock_storage = MagicMock()
    uploader = AsyncDataUploader(storage=mock_storage, buffer_size=1)

    # Process through TMDBTool
    print("Processing through TMDBTool...")
    observations = []
    async for obs in tool.process(title):
        print(f"  TMDBTool yielded: {type(obs).__name__}")
        observations.append(obs)

        # Pass through uploader
        print("  Processing through uploader...")
        async for uploaded in uploader.process(obs):
            print(f"    Uploader yielded: {type(uploaded).__name__}")

    print(f"\nTotal observations: {len(observations)}")

    # Shutdown uploader
    uploader.shutdown()

    return observations

if __name__ == "__main__":
    results = asyncio.run(test_pipeline())
    print(f"\nSuccess! Processed {len(results)} observations")

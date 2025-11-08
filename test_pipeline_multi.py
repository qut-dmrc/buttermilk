#!/usr/bin/env python
"""Test multi-processor pipeline orchestrator."""

import asyncio

from buttermilk._core.types import BaseRecord
from buttermilk.pipeline import PipelineOrchestrator
from buttermilk.tools.catalog_test import Observation, Title


# Create mock processors
class MockTMDBProcessor:
    """Mock processor that transforms Title to Observations."""

    async def process(self, record: Title):
        """Transform Title to Observations."""
        print(f"  TMDB processing: {record.title}")
        # Yield 2 observations for each title
        yield Observation(
            record_id=record.record_id,
            title=record.title,
            year=record.year,
            provider_name="Netflix",
            region="US",
            available=True,
            source="TMDB",
        )
        yield Observation(
            record_id=record.record_id,
            title=record.title,
            year=record.year,
            provider_name="Amazon",
            region="US",
            available=True,
            source="TMDB",
        )


class MockUploader:
    """Mock uploader that passes through records."""

    def __init__(self):
        self.uploaded = []

    async def process(self, record: BaseRecord):
        """Pass through and track."""
        print(f"  Uploading: {type(record).__name__}")
        self.uploaded.append(record)
        yield record

    def shutdown(self):
        print(f"  Uploader shutdown: {len(self.uploaded)} records uploaded")


async def test_multi_processor_pipeline():
    """Test pipeline with multiple processors."""

    # Create test data source
    async def source():
        titles = [
            Title(record_id="1", title="Movie 1", year=2020),
            Title(record_id="2", title="Movie 2", year=2021),
        ]
        for title in titles:
            yield title

    # Create processors
    tmdb = MockTMDBProcessor()
    uploader = MockUploader()

    # Create orchestrator with multiple processors
    orchestrator = PipelineOrchestrator(
        stage_name="test_pipeline",
        source=source(),
        processors=[tmdb, uploader],  # Chain of processors
        concurrency=2,
    )

    # Run pipeline
    print("Running pipeline...")
    results = []
    async for record in orchestrator():
        print(f"Final output: {type(record).__name__} - {record.title}")
        results.append(record)

    # Shutdown
    uploader.shutdown()

    print("\nPipeline complete!")
    print("Input: 2 Titles")
    print(f"Output: {len(results)} Observations")
    print(f"Uploaded: {len(uploader.uploaded)} records")

    return results


if __name__ == "__main__":
    results = asyncio.run(test_multi_processor_pipeline())
    print(f"\nSuccess! Pipeline processed {len(results)} records")

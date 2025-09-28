#!/usr/bin/env python
"""Test multi-processor pipeline orchestrator."""

from typing import Any

import pytest

from buttermilk._core.types import BaseRecord
from buttermilk.pipeline import PipelineOrchestrator
from buttermilk.tools.catalog_test import Observation, Title


# Create fake processors
class FakeTMDBProcessor:
    """Mock processor that transforms Title to Observations."""

    async def process(self, inputs: dict[str, Any]):
        """Transform Title to Observations."""
        _ = self  # reference self to satisfy linter
        record = inputs["record"]  # Extract record from inputs dict
        print(f"  TMDB processing: {record.title}")
        # Yield 2 observations for each title, wrapped in dict
        yield {"record": Observation(
            record_id=record.record_id,
            title=record.title,
            year=record.year,
            provider_name="Netflix",
            region="US",
            available=True,
            source="TMDB",
            provider_type="flatrate",
            price=None,
            currency=None,
            format=None,
        )}
        yield {"record": Observation(
            record_id=record.record_id,
            title=record.title,
            year=record.year,
            provider_name="Amazon",
            region="US",
            available=True,
            source="TMDB",
            provider_type="flatrate",
            price=None,
            currency=None,
            format=None,
        )}


class FakeUploader:
    """Mock uploader that passes through records."""

    def __init__(self):
        self.uploaded = []

    async def process(self, inputs: dict[str, Any]):
        """Pass through and track."""
        record = inputs["record"]  # Extract record from inputs dict
        print(f"  Uploading: {type(record).__name__}")
        self.uploaded.append(record)
        yield inputs  # Pass through unchanged

    def shutdown(self):
        print(f"  Uploader shutdown: {len(self.uploaded)} records uploaded")


@pytest.mark.asyncio
async def test_multi_processor_pipeline():
    """Test pipeline with multiple processors."""

    # Create test data source
    async def source():
        titles = [
            Title(record_id="1", title="Movie 1", year=2020),
            Title(record_id="2", title="Movie 2", year=2021),
        ]
        for title in titles:
            yield {"record": title}

    # Create processors
    tmdb = FakeTMDBProcessor()
    uploader = FakeUploader()

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
    async for result_dict in orchestrator():
        print(f"Got result_dict: {result_dict}")
        record = result_dict["record"]
        print(f"Final output: {type(record).__name__} - {record.record_id}")
        results.append(record)

    # Shutdown
    uploader.shutdown()

    # Basic assertions: all outputs are Observations and uploader saw the same
    assert all(isinstance(r, Observation) for r in results)
    assert len(uploader.uploaded) == len(results)

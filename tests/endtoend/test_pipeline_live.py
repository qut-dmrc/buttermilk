#!/usr/bin/env python
"""Test multi-processor pipeline orchestrator."""

from typing import Any

import pytest

from buttermilk.pipeline import PipelineOrchestrator
from buttermilk.tools.catalog_test import THEMOVIEDB_AVAILABLE, Observation

# Skip entire module if themoviedb is not installed
pytestmark = pytest.mark.skipif(
    not THEMOVIEDB_AVAILABLE,
    reason="themoviedb package not installed - install with: pip install themoviedb.py",
)


# Create fake processors
class FakeTMDBProcessor:
    """Mock processor that transforms Title to Observations."""

    async def process(self, inputs: dict[str, Any]):
        """Transform Title to Observations."""
        _ = self  # reference self to satisfy linter
        record = inputs["record"]  # Extract record from inputs dict
        print(f"  TMDB processing: {record.title}")
        # Yield 2 observations for each title, wrapped in dict
        yield {
            "record": Observation(
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
            )
        }
        yield {
            "record": Observation(
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
            )
        }


class FakeUploader:
    """Mock uploader that passes through records."""

    def __init__(self):
        self.uploaded = []

    async def process(self, inputs: Any):
        """Pass through and track."""
        record = inputs
        print(f"  Uploading: {type(record).__name__}")
        self.uploaded.append(record)
        yield record  # Pass through unchanged

    def shutdown(self):
        print(f"  Uploader shutdown: {len(self.uploaded)} records uploaded")


@pytest.mark.anyio
async def test_live_pipeline(real_bm, real_conf):
    """Test pipeline with multiple processors using storage-backed source.

    The source storage is configured in testing.yaml and loaded via conftest.py.
    We read a small batch (2) of Title records from BigQuery-backed storage.
    """

    # Build storage from testing config and create an async source of Title records
    storage = real_bm.get_storage(real_conf.storage.titles)
    source = storage(batch_size=2)  # limit to small, deterministic batch

    # Create processors
    tmdb = FakeTMDBProcessor()
    uploader = FakeUploader()

    # Create orchestrator with multiple processors
    orchestrator = PipelineOrchestrator(
        stage_name="test_pipeline",
        source=source,
        processors=[tmdb, uploader],  # Chain of processors
        concurrency=2,
    )

    # Run pipeline
    print("Running pipeline...")
    results = []
    async for result_dict in orchestrator():
        record = result_dict["record"]
        print(f"Final output: {type(record).__name__} - {record.record_id}")
        results.append(record)

    # Shutdown
    uploader.shutdown()

    print("\nPipeline complete!")
    print("Input: 2 Titles (from storage)")
    print(f"Output: {len(results)} Observations")
    print(f"Uploaded: {len(uploader.uploaded)} records")

    # Basic assertions: all outputs are Observations and uploader saw the same
    assert all(isinstance(r, Observation) for r in results)
    assert len(uploader.uploaded) == len(results)

    return results

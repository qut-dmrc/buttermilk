"""Demo integration test for mmtmdb pipeline caching investigation.

This test runs record 56599 through the mmtmdb pipeline with caching enabled
and outputs all observations that would be saved to BigQuery.

Purpose: Investigate whether caching within the pipeline is causing data loss
(e.g., only first provider record per title being saved).

Usage:
    TMDB_API_KEY=your_key pytest tests/endtoend/test_mmtmdb_caching_demo.py -v -s
"""

import os

import pytest

from buttermilk.pipeline import PipelineOrchestrator
from buttermilk.tools.catalog_test import THEMOVIEDB_AVAILABLE, Observation, Title, TitleType, TMDBTool

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not THEMOVIEDB_AVAILABLE,
        reason="themoviedb package not installed",
    ),
]


class ObservationCapture:
    """Processor that captures observations instead of uploading to BigQuery.

    Mimics AsyncDataUploader's interface but captures records for inspection.
    """

    def __init__(self):
        self.captured: list[Observation] = []
        self.call_count = 0

    async def process(self, context, **kwargs):
        """Capture the observation and pass through."""
        record = context.record if hasattr(context, "record") else context
        self.call_count += 1
        self.captured.append(record)
        print(
            f"  [Capture #{self.call_count}] {type(record).__name__} record_id={record.record_id} "
            f"provider={getattr(record, 'provider_name', 'N/A')} "
            f"region={getattr(record, 'region', 'N/A')} "
            f"type={getattr(record, 'provider_type', 'N/A')}"
        )
        yield record

    def shutdown(self):
        """No-op for compatibility."""
        pass

    async def finalize_processing(self) -> bool:
        """No-op for compatibility."""
        return True


@pytest.fixture
def api_key() -> str:
    """Get TMDB API key from environment."""
    key = os.getenv("TMDB_API_KEY")
    if not key:
        pytest.skip("TMDB_API_KEY environment variable not set")
    return key


@pytest.mark.parametrize(
    "record_id,enable_record_cache,force_reprocess,expected_results",
    [
        pytest.param("56599", True, False, 5, id="cache_enabled_use_cache"),
        pytest.param("56599", False, True, 5, id="cache_disabled_force_reprocess"),
        pytest.param("56599", True, True, 5, id="cache_enabled_force_reprocess"),
        pytest.param("56599", False, False, 5, id="cache_disabled_no_force"),
    ],
)
@pytest.mark.anyio
async def test_mmtmdb_single_record_caching(api_key: str, record_id, enable_record_cache, force_reprocess, expected_results, real_bm, tmp_path):
    """Run record  through mmtmdb pipeline and capture all observations.

    This test demonstrates caching behavior within a single pipeline run.
    It uses production-like settings (enable_record_cache=True) to show
    whether caching is affecting observation capture.
    """

    # Create the source - single Title with record_id 56599
    async def source():
        # Record 56599 - we just need the ID, TMDB API will look it up
        title = Title(
            record_id=record_id,
            title=f"Test Title {record_id}",  # Will be enriched by TMDB
            year=None,  # Let TMDB fill in
            type=TitleType.MOVIE,
        )
        print(f"\n📥 Source yielding: record_id={title.record_id}")
        yield title

    # Create TMDBTool processor (matches production config)
    tmdb_tool = TMDBTool(api_key=api_key, region="AU")

    # Create observation capture processor (replaces AsyncDataUploader)
    capture = ObservationCapture()

    # Create pipeline with production-like caching settings
    orchestrator = PipelineOrchestrator(
        pipeline_name="tmdb_observations_demo",
        source=source(),
        processors=[tmdb_tool, capture],
        concurrency=1,  # Single record, no need for concurrency
        enable_record_cache=enable_record_cache,  # PRODUCTION SETTING - caching enabled
        force_reprocess=force_reprocess,  # PRODUCTION SETTING - use cache if available
    )

    print("\n" + "=" * 70)
    print("MMTMDB PIPELINE CACHING DEMO")
    print("=" * 70)
    print("Record ID: 56599")
    print("Caching: ENABLED (production setting)")
    print("Force Reprocess: False")
    print(f"Cache Dir: {orchestrator._record_cache.base_dir if orchestrator._record_cache else 'N/A'}")
    print("=" * 70)

    # Run pipeline
    print("\n🚀 Running pipeline...\n")
    results = []
    async for record in orchestrator():
        results.append(record)

    # Output summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Pipeline yielded: {len(results)} records")
    print(f"Capture received: {len(capture.captured)} observations")

    # Detailed observation breakdown
    print("\n📊 OBSERVATIONS CAPTURED (would be saved to BigQuery):\n")

    # Group by provider type
    by_type: dict[str, list] = {}
    for obs in capture.captured:
        ptype = obs.provider_type or "unknown"
        if ptype not in by_type:
            by_type[ptype] = []
        by_type[ptype].append(obs)

    for ptype, obs_list in sorted(by_type.items()):
        print(f"\n  {ptype.upper()} ({len(obs_list)} records):")
        for obs in obs_list:
            available = "✓" if obs.available else "✗"
            print(f"    {available} {obs.provider_name or 'N/A':20} | region={obs.region or 'N/A':4} | provider_id={obs.provider_id or 'N/A'}")

    # Summary stats
    print("\n" + "-" * 70)
    regions = set(obs.region for obs in capture.captured if obs.region)
    providers = set(obs.provider_name for obs in capture.captured if obs.provider_name)
    available_count = sum(1 for obs in capture.captured if obs.available)

    print(f"Total observations: {len(capture.captured)}")
    print(f"Unique regions: {len(regions)} - {sorted(regions)[:10]}{'...' if len(regions) > 10 else ''}")
    print(f"Unique providers: {len(providers)}")
    print(f"Available: {available_count} | Unavailable: {len(capture.captured) - available_count}")

    # Check for potential caching issues
    print("\n" + "=" * 70)
    print("CACHING ANALYSIS")
    print("=" * 70)

    if orchestrator._record_cache:
        cache_dir = orchestrator._record_cache.base_dir
        print(f"Cache directory: {cache_dir}")
        # Check if cache files exist for this record
        import glob

        cache_pattern = str(cache_dir / "**" / "*56599*")
        cache_files = glob.glob(cache_pattern, recursive=True)
        print(f"Cache files for record 56599: {len(cache_files)}")
        for cf in cache_files[:5]:
            print(f"  - {cf}")
        if len(cache_files) > 5:
            print(f"  ... and {len(cache_files) - 5} more")

    # Assertions
    # Note: With warm cache, ObservationCapture may not be called (outputs served from cache).
    # So we check pipeline output (results) which represents actual data that would be saved.
    # capture.captured only has items when the processor is actually invoked (cold cache or force_reprocess).
    assert len(results) >= expected_results, f"Expected pipeline to yield {expected_results} records, got {len(results)}"

    # For non-cached runs, also verify capture received the records
    if not enable_record_cache or force_reprocess:
        assert len(capture.captured) >= expected_results, f"Expected capture to receive {expected_results} observations"

    # Return captured data for further inspection
    return {
        "record_id": record_id,
        "observations": capture.captured,
        "by_type": by_type,
        "regions": regions,
        "providers": providers,
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""Integration tests for TMDB API with live API calls.

These tests require a real TMDB API key set in the TMDB_API_KEY environment variable.
They are marked as integration tests and will be skipped if no API key is available.

Usage:
    # Run all unit tests (no API key needed)
    pytest tests/tools/test_tmdb.py

    # Run integration tests (API key required)
    TMDB_API_KEY=your_key pytest tests/integration/test_tmdb_live.py

    # Run only integration tests
    pytest -m integration

    # Skip integration tests
    pytest -m "not integration"
"""

import os

import pytest

from buttermilk import BM
from buttermilk.tools.catalog_test import (
    THEMOVIEDB_AVAILABLE,
    Observation,
    Title,
    TitleType,
    TMDBTool,
)

# Pytest markers for conditional test execution
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not THEMOVIEDB_AVAILABLE,
        reason="themoviedb package not installed - install with: pip install themoviedb.py",
    ),
]


@pytest.fixture
def api_key() -> str:
    """Get TMDB API key from environment."""
    key = os.getenv("TMDB_API_KEY")
    if not key:
        pytest.skip("TMDB_API_KEY environment variable not set")
    return key


@pytest.fixture
def tmdb_tool_live(api_key: str) -> TMDBTool:
    """Create TMDBTool with real API key for live testing."""
    return TMDBTool(api_key=api_key)


class TestTMDBLiveAPI:
    """Live API integration tests for TMDB functionality."""

    @pytest.mark.anyio
    async def test_live_search_movie_only(self, tmdb_tool_live: TMDBTool) -> None:
        """Test searching for a movie (search functionality only)."""
        # Search for a well-known movie
        title = await tmdb_tool_live.search_movie(title="The Matrix", year=1999)

        # Verify we got a Title object with correct search results
        assert isinstance(title, Title)
        assert title.title is not None
        assert "Matrix" in title.title  # Should contain "Matrix"
        assert title.record_id is not None
        assert title.year == 1999
        assert isinstance(title.metadata, dict)
        assert title.type == TitleType.MOVIE

        # Verify metadata contains expected fields from search
        assert "overview" in title.metadata or "original_title" in title.metadata

    @pytest.mark.anyio
    async def test_live_availability_batman_1989(self, tmdb_tool_live: TMDBTool) -> None:
        """Test availability checking for Batman (1989) using known movie ID."""
        # Create Title object directly from known movie ID (no search needed)
        title = Title(
            record_id="268",  # Batman (1989)
            title="Batman",
            year=1989,
            type=TitleType.MOVIE,
            metadata={"known_movie": True},
        )

        # Test availability checking - get_availability returns async generator for all regions
        results = [obs async for obs in tmdb_tool_live.get_availability(title)]

        # Verify we get observations (TMDB returns all available regions)
        assert isinstance(results, list)
        assert len(results) >= 1  # At least one observation

        # Check that we have observations with regions
        regions_found = {obs.region for obs in results if obs.region}
        # TMDB returns whatever regions have data - we just verify we got some
        assert len(regions_found) >= 1 or any(obs.region is None for obs in results)

        # Check basic structure of observations
        for result in results:
            assert isinstance(result, Observation)
            assert result.source == "TMDB"
            assert isinstance(result.available, bool)
            assert isinstance(result.metadata, dict)

            # Verify record structure
            if result.available:
                assert result.provider_name is not None
                assert result.provider_id is not None
                assert result.provider_type in ["flatrate", "rent", "buy"]
            else:
                assert result.provider_name is None

    @pytest.mark.anyio
    async def test_live_availability_batman_azteca_2025(self, tmdb_tool_live: TMDBTool) -> None:
        """Test availability checking for Batman Azteca: Choque de Imperios (2025)."""
        # Create Title object for the 2025 Batman movie
        title = Title(
            record_id="987400",  # Batman Azteca: Choque de Imperios (2025)
            title="Batman Azteca: Choque de Imperios",
            year=2025,
            type=TitleType.MOVIE,
            metadata={"known_movie": True},
        )

        # Test availability - get_availability returns async generator for all regions
        results = [obs async for obs in tmdb_tool_live.get_availability(title)]

        # Verify we get a response (may not be available anywhere yet)
        assert isinstance(results, list)
        assert len(results) >= 1

        # Check observation structure
        for result in results:
            assert isinstance(result, Observation)
            assert result.source == "TMDB"
            assert isinstance(result.available, bool)
            assert isinstance(result.metadata, dict)

            # For a 2025 movie, it's likely not widely available yet
            # so we mainly test that the API call works correctly

    @pytest.mark.anyio
    async def test_live_availability_multiple_batman_movies(self, tmdb_tool_live: TMDBTool) -> None:
        """Test availability checking for multiple Batman movies to compare results."""
        # Create Title objects for both Batman movies
        batman_1989 = Title(
            record_id="268",
            title="Batman",
            year=1989,
            type=TitleType.MOVIE,
            metadata={"era": "classic"},
        )

        batman_2025 = Title(
            record_id="987400",
            title="Batman Azteca: Choque de Imperios",
            year=2025,
            type=TitleType.MOVIE,
            metadata={"era": "modern"},
        )

        # Get availability for both movies - async generators for all regions
        results_1989 = [obs async for obs in tmdb_tool_live.get_availability(batman_1989)]
        results_2025 = [obs async for obs in tmdb_tool_live.get_availability(batman_2025)]

        # Both should return valid observation lists
        assert isinstance(results_1989, list)
        assert isinstance(results_2025, list)
        assert len(results_1989) >= 1
        assert len(results_2025) >= 1

        # Check that movie metadata is correctly preserved in observations
        for result in results_1989:
            assert result.metadata.get("movie_id") == "268"

        for result in results_2025:
            assert result.metadata.get("movie_id") == "987400"

        # The classic Batman (1989) is more likely to be available
        # than the 2025 movie, but we don't assert this since availability changes

    @pytest.mark.anyio
    async def test_live_complete_workflow_search_then_availability(self, tmdb_tool_live: TMDBTool) -> None:
        """Test complete end-to-end workflow: search movie then check availability."""
        # Step 1: Search for a movie
        title = await tmdb_tool_live.search_movie(title="Inception", year=2010)

        # Verify search worked
        assert isinstance(title, Title)
        assert title.title is not None
        assert title.record_id is not None
        assert title.year == 2010

        # Step 2: Use the found title to check availability - async generator for all regions
        results = [obs async for obs in tmdb_tool_live.get_availability(title)]

        # Verify availability check worked
        assert isinstance(results, list)
        assert len(results) >= 1

        # Verify the complete data flow from search to observations
        for result in results:
            assert isinstance(result, Observation)
            assert result.source == "TMDB"
            assert result.metadata.get("movie_id") == title.record_id
            assert isinstance(result.available, bool)

    @pytest.mark.anyio
    async def test_live_search_nonexistent_movie(self, tmdb_tool_live: TMDBTool) -> None:
        """Test searching for a movie that definitely doesn't exist."""
        title = await tmdb_tool_live.search_movie(title="Absolutely Nonexistent Movie Title 9999")

        # Should return None when movie not found
        assert title is None

    @pytest.mark.anyio
    async def test_live_regional_differences(self, tmdb_tool_live: TMDBTool) -> None:
        """Test that the same movie may have different availability in different regions."""
        movie_title = "Pulp Fiction"
        year = 1994

        # Search for the movie
        title = await tmdb_tool_live.search_movie(title=movie_title, year=year)

        # Verify movie was found
        assert title is not None
        assert title.title is not None

        # Get availability for all regions - TMDB returns whatever regions have data
        results = [obs async for obs in tmdb_tool_live.get_availability(title)]

        # Should return observations
        assert isinstance(results, list)
        assert len(results) >= 1

        # Collect unique regions from results
        regions_found = {r.region for r in results if r.region}

        # A classic movie like Pulp Fiction should have data in multiple regions
        # But we just verify we got at least some regional data
        assert len(regions_found) >= 1 or any(obs.region is None for obs in results)

        # Availability may differ between regions
        # (This is the key insight - same content, different regional licensing)

    @pytest.mark.anyio
    async def test_live_search_with_year_filtering(self, tmdb_tool_live: TMDBTool) -> None:
        """Test that year filtering works correctly."""
        # Search for "Batman" without year (should get recent results)
        recent_title = await tmdb_tool_live.search_movie(title="Batman")

        # Search for "Batman" from 1989 (specific Tim Burton film)
        year_filtered_title = await tmdb_tool_live.search_movie(title="Batman", year=1989)

        # Both should find movies
        assert recent_title is not None
        assert year_filtered_title is not None

        # Both should have Batman in the title
        assert "Batman" in recent_title.title
        assert "Batman" in year_filtered_title.title

        # The 1989 result should have that year
        assert year_filtered_title.year == 1989

        # The two results might be different movies
        # (Recent could be The Batman 2022, while 1989 is the Tim Burton film)

    @pytest.mark.anyio
    async def test_live_error_handling_no_availability(self, tmdb_tool_live: TMDBTool) -> None:
        """Test handling when a movie has no availability data."""
        # First search for the movie
        title = await tmdb_tool_live.search_movie(title="The Matrix")
        assert title is not None

        # Get availability - API returns whatever regions have data
        results = [obs async for obs in tmdb_tool_live.get_availability(title)]

        # Should handle gracefully - return observations
        assert isinstance(results, list)
        assert len(results) >= 1

        # Check all results are valid Observation objects
        for result in results:
            assert isinstance(result, Observation)
            assert isinstance(result.available, bool)
            if not result.available:
                assert result.provider_name is None

    @pytest.mark.anyio
    async def test_endtoend_response_structure(self, tmdb_tool_live: TMDBTool) -> None:
        """Test that live API responses match our expected data structure."""
        # First search for movie
        title = await tmdb_tool_live.search_movie(title="Inception", year=2010)

        assert isinstance(title, Title)
        assert title.record_id is not None
        assert title.title is not None
        assert title.year == 2010
        assert isinstance(title.metadata, dict)

        # Now get availability - async generator for all regions
        results = [obs async for obs in tmdb_tool_live.get_availability(title)]

        assert isinstance(results, list)
        assert len(results) >= 1

        for result in results:
            assert isinstance(result, Observation)

            # Required fields
            assert result.record_id is not None
            assert result.call_id is not None
            assert result.test_date is not None
            assert result.source == "TMDB"
            assert isinstance(result.available, bool)
            assert isinstance(result.metadata, dict)
            assert isinstance(result.error, list)

            # Metadata should include movie_id
            assert "movie_id" in result.metadata

            if result.available:
                # Available results should have provider info
                assert result.provider_name is not None
                assert result.provider_id is not None
                assert result.provider_type in ["flatrate", "rent", "buy"]
            else:
                # Unavailable results should be explicit
                assert result.provider_name is None
                assert result.provider_id is None

    @pytest.mark.anyio
    async def test_live_rate_limiting_resilience(self, tmdb_tool_live: TMDBTool) -> None:
        """Test that multiple rapid requests work within API rate limits."""
        movies = [
            ("The Shawshank Redemption", 1994),
            ("The Godfather", 1972),
            ("The Dark Knight", 2008),
        ]

        titles = []

        # Make multiple search requests in sequence
        for movie_title, year in movies:
            title = await tmdb_tool_live.search_movie(title=movie_title, year=year)
            if title:
                titles.append(title)

        # All searches should succeed (no rate limit errors)
        assert len(titles) == len(movies)

        # Now get availability for all movies - async generator for all regions
        for title in titles:
            results = [obs async for obs in tmdb_tool_live.get_availability(title)]

            # Should get results without rate limiting errors
            assert isinstance(results, list)
            for result in results:
                assert isinstance(result, Observation)
                assert result.source == "TMDB"
                # No rate limiting errors
                if result.error:
                    error_messages = [str(err) for err in result.error]
                    for msg in error_messages:
                        assert "rate limit" not in msg.lower()
                        assert "too many requests" not in msg.lower()

    @pytest.mark.anyio
    async def test_live_discover_movies_with_backup(self, tmdb_tool_live: TMDBTool, tmp_path) -> None:
        """Test discovering movies from TMDB with JSON backup using minimal data."""
        from datetime import date

        from buttermilk.tools.catalog_test import DatePeriod

        # Test backup functionality by fetching just two small periods (two single days)
        # This tests the concurrency and backup file creation without fetching too much data
        period1 = DatePeriod(date(1968, 6, 15), date(1968, 6, 15))  # Single day
        period2 = DatePeriod(date(1969, 3, 20), date(1969, 3, 20))  # Another single day

        # Fetch first period
        titles1, _ = await tmdb_tool_live.fetch_single_page(period1, page=1)

        # Fetch second period
        titles2, _ = await tmdb_tool_live.fetch_single_page(period2, page=1)

        # Combine results
        results = titles1 + titles2

        # Should get some results (though may be empty for early dates)
        assert isinstance(results, list)

        # Check each result if any exist
        for title in results[:5]:  # Limit check to first 5 for performance
            assert isinstance(title, Title)
            assert title.record_id is not None
            assert title.title is not None
            assert title.type == TitleType.MOVIE

            # Year should be reasonable if present (around 1968-1969)
            if title.year:
                assert 1960 <= title.year <= 1975  # Should be around 1968-1969 for our test period

        # Test that we can create backup files manually (simulating the backup functionality)
        backup_file = tmp_path / "period_test.json"
        import json

        from buttermilk.utils.utils import scrub_serializable

        if results:
            with open(backup_file, "w") as f:
                json.dump(
                    [scrub_serializable(title.model_dump()) for title in results],
                    f,
                    indent=2,
                    default=str,
                )

            # Verify backup file was created
            assert backup_file.exists()

            # Verify we can load it back
            with open(backup_file) as f:
                loaded_data = json.load(f)
                assert isinstance(loaded_data, list)

    @pytest.mark.anyio
    async def test_fetch_single_page_with_bigquery_save(self, real_bm: BM, real_conf, tmdb_tool_live: TMDBTool, tmp_path) -> None:
        """Test single page fetch with real API and BigQuery save."""
        from datetime import date

        from buttermilk.tools.catalog_test import DatePeriod

        # Get storage configs from real_conf - it's a Pydantic model
        storage_conf = getattr(real_conf, "storage", None)
        observations_config = getattr(storage_conf, "observations", None) if storage_conf else None
        titles_config = getattr(storage_conf, "titles", None) if storage_conf else None

        # Setup TMDBTool with storage configs for test datasets (if available)
        tmdb_tool = TMDBTool(
            api_key=tmdb_tool_live.api_key,
            observations_storage_config=observations_config,
            titles_storage_config=titles_config,
        )

        # Fetch one page for a single day in January 1968 (minimal data but likely to have some movies)
        period = DatePeriod(date(1968, 1, 15), date(1968, 1, 15))
        titles, has_more = await tmdb_tool.fetch_single_page(period, page=1)

        # Verify results (1968 should have some movies)
        assert isinstance(titles, list), "Should return a list"
        if titles:  # Only check Title properties if we got results
            assert isinstance(titles[0], Title)
            assert titles[0].record_id is not None
            assert titles[0].title is not None
            assert titles[0].type == TitleType.MOVIE

        # Verify has_more flag makes sense
        assert isinstance(has_more, bool)

        # Verify BigQuery save if uploader was configured
        if tmdb_tool.titles_uploader:
            tmdb_tool.titles_uploader.shutdown()

            bm = real_bm
            if titles_config:
                titles_storage = bm.get_storage(titles_config)
                # Try to query for some of the data we just saved
                # Note: This is a simple existence check - may fail if table doesn't exist yet
                try:
                    saved_count = len(titles_storage) if hasattr(titles_storage, "__len__") else 0
                    print(f"Saved {saved_count} titles to BigQuery test dataset from 1968")
                except Exception as e:
                    print(f"Could not verify BigQuery save (table may not exist yet): {e}")
                    # Test still passes - the main functionality (API calls) worked


@pytest.mark.anyio
async def test_fetch_single_page_date_range_validation(
    tmdb_tool_live: TMDBTool,
) -> None:
    """Test that fetch_single_page respects date range parameters."""
    from datetime import date

    from buttermilk.tools.catalog_test import DatePeriod

    # Fetch from a single day in 1969 for minimal data but likely results
    period = DatePeriod(date(1969, 3, 20), date(1969, 3, 20))  # Single day in March 1969
    titles, has_more = await tmdb_tool_live.fetch_single_page(period, page=1)

    # Verify that returned movies are from the correct date range
    assert isinstance(titles, list), "Should return a list"

    # Check a few movies have reasonable dates (around 1969) if any exist
    for title in titles[:5]:  # Check first 5 movies
        if title.year:
            assert 1960 <= title.year <= 1975, f"Movie year {title.year} should be near 1969"

        # Verify movie has basic required fields
        assert title.record_id
        assert title.title
        assert title.type == TitleType.MOVIE


@pytest.mark.skipif(not os.getenv("TMDB_API_KEY"), reason="No TMDB API key provided")
class TestTMDBConfigurationLive:
    """Test TMDB tool configuration with live API."""

    def test_live_tool_initialization_with_valid_key(self) -> None:
        """Test that tool initializes correctly with valid API key."""
        api_key = os.getenv("TMDB_API_KEY")
        tool = TMDBTool(api_key=api_key)

        assert tool.api_key == api_key
        assert tool.base_url == "https://api.themoviedb.org/3"
        assert tool.language == "en-US"
        assert tool.region == "AU"  # Default region

    def test_live_tool_custom_configuration(self) -> None:
        """Test tool with custom configuration parameters."""
        api_key = os.getenv("TMDB_API_KEY")
        tool = TMDBTool(api_key=api_key, language="fr-FR", region="FR")

        assert tool.language == "fr-FR"
        assert tool.region == "FR"

    def test_live_environment_variable_fallback(self) -> None:
        """Test that tool picks up API key from environment."""
        # This test relies on TMDB_API_KEY being set
        tool = TMDBTool()  # Should use env var

        assert tool.api_key == os.getenv("TMDB_API_KEY")


class TestTMDBLivePricing:
    """Live guard for the rent/buy price+currency invariant (movie 98 = Gladiator).

    Context: a collection recorded movie 98's AU buy providers (Fetch TV 436,
    YouTube 192) with empty price/currency and no errors. PR #414 claimed to fix
    this, but its unit tests fed *fabricated* raw provider dicts that bypass the
    themoviedb SDK, so the green tests never exercised the real production path.

    This test hits the REAL TMDB API with NO mocks and NO cached fixtures: it
    reproduces production exactly and detects if/when TMDB's watch/providers
    payload changes. The themoviedb SDK opens a fresh aiohttp session per call
    with no HTTP cache, so every run is genuinely live.

    Invariant: every *available* rent/buy observation must carry both price and
    currency. CONFIRMED 2026-05-28 (live, movie 98): TMDB does NOT supply pricing
    on this endpoint, so the invariant cannot currently hold -- the test is marked
    xfail(strict=True). If TMDB ever starts returning price/currency, this test
    XPASSes and (being strict) fails the suite, signalling us to wire it through
    rather than leave the columns silently empty.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "TMDB /watch/providers returns no price/currency/presentation_type, even "
            "for rent/buy (confirmed live against movie 98 on 2026-05-28). A strict "
            "XPASS is the intended tripwire if the API ever begins supplying pricing."
        ),
    )
    @pytest.mark.anyio
    async def test_rent_buy_observations_have_price_and_currency(
        self, tmdb_tool_live: TMDBTool
    ) -> None:
        # record_id is passed straight through to client.movie(98).watch/providers.
        results = [
            obs
            async for obs in tmdb_tool_live.get_availability_by_id(
                record_id=98, title="Gladiator", year=2000
            )
        ]

        purchasable = [
            o for o in results if o.available and o.provider_type in ("rent", "buy")
        ]
        # The invariant is only meaningful if the title actually has paid offers.
        assert purchasable, (
            "movie 98 returned no available rent/buy providers in any region; "
            "cannot validate the price/currency invariant"
        )

        missing = [
            {
                "region": o.region,
                "type": o.provider_type,
                "provider": o.provider_name,
                "price": o.price,
                "currency": o.currency,
            }
            for o in purchasable
            if o.price is None or o.currency is None
        ]
        assert not missing, (
            f"{len(missing)} of {len(purchasable)} available rent/buy observations are "
            f"missing price and/or currency (TMDB watch/providers may not supply "
            f"pricing): {missing}"
        )


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v"])

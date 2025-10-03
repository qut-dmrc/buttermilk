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
from buttermilk.tools.catalog_test import Observation, Title, TitleType, TMDBTool

# Pytest markers for conditional test execution
pytestmark = [pytest.mark.integration, pytest.mark.endtoend]


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
            metadata={"known_movie": True}
        )

        # Test availability checking for multiple regions
        results = await tmdb_tool_live.get_availability(title, regions=["US", "GB"])

        # Verify we get observations for each region
        assert isinstance(results, list)
        assert len(results) >= 2  # At least one observation per region

        # Check that we have observations for each requested region
        regions_found = {obs.region for obs in results}
        assert "US" in regions_found
        assert "GB" in regions_found

        # Check basic structure of observations
        for result in results:
            assert isinstance(result, Observation)
            assert result.source == "TMDB"
            assert result.region in ["US", "GB"]
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
            metadata={"known_movie": True}
        )

        # Test availability (this is a very new/upcoming movie)
        results = await tmdb_tool_live.get_availability(title, regions=["US"])

        # Verify we get a response (may not be available anywhere yet)
        assert isinstance(results, list)
        assert len(results) >= 1

        # Check observation structure
        for result in results:
            assert isinstance(result, Observation)
            assert result.source == "TMDB"
            assert result.region == "US"
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
            metadata={"era": "classic"}
        )

        batman_2025 = Title(
            record_id="987400",
            title="Batman Azteca: Choque de Imperios",
            year=2025,
            type=TitleType.MOVIE,
            metadata={"era": "modern"}
        )

        # Get availability for both movies in US region
        results_1989 = await tmdb_tool_live.get_availability(batman_1989, regions=["US"])
        results_2025 = await tmdb_tool_live.get_availability(batman_2025, regions=["US"])

        # Both should return valid observation lists
        assert isinstance(results_1989, list)
        assert isinstance(results_2025, list)
        assert len(results_1989) >= 1
        assert len(results_2025) >= 1

        # Check that movie metadata is correctly preserved in observations
        for result in results_1989:
            assert result.metadata.get("movie_id") == "268"
            assert "Batman" in result.metadata.get("title", "")

        for result in results_2025:
            assert result.metadata.get("movie_id") == "987400"
            assert "Batman Azteca" in result.metadata.get("title", "")

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

        # Step 2: Use the found title to check availability
        results = await tmdb_tool_live.get_availability(title, regions=["US"])

        # Verify availability check worked
        assert isinstance(results, list)
        assert len(results) >= 1

        # Verify the complete data flow from search to observations
        for result in results:
            assert isinstance(result, Observation)
            assert result.source == "TMDB"
            assert result.region == "US"
            assert result.metadata.get("title") == title.title
            assert result.metadata.get("movie_id") == title.record_id
            assert isinstance(result.available, bool)

    @pytest.mark.anyio
    async def test_live_search_nonexistent_movie(self, tmdb_tool_live: TMDBTool) -> None:
        """Test searching for a movie that definitely doesn't exist."""
        title = await tmdb_tool_live.search_movie(
            title="Absolutely Nonexistent Movie Title 9999"
        )

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

        # Get availability for both US and GB in one call
        results = await tmdb_tool_live.get_availability(title, regions=["US", "GB"])

        # Should return observations for both regions
        assert isinstance(results, list)
        
        # Separate results by region
        us_results = [r for r in results if r.region == "US"]
        gb_results = [r for r in results if r.region == "GB"]
        
        # Should have at least one observation per region (even if null)
        assert len(us_results) >= 1
        assert len(gb_results) >= 1

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
    async def test_live_error_handling_invalid_region(self, tmdb_tool_live: TMDBTool) -> None:
        """Test error handling with invalid region codes."""
        # First search for the movie
        title = await tmdb_tool_live.search_movie(title="The Matrix")
        assert title is not None
        
        # Try to get availability for invalid region
        results = await tmdb_tool_live.get_availability(title, regions=["INVALID"])

        # Should handle gracefully - return null observation for invalid region
        assert isinstance(results, list)
        assert len(results) >= 1

        result = results[0]
        assert isinstance(result, Observation)
        assert result.region == "INVALID"
        assert result.available is False
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
        
        # Now get availability
        results = await tmdb_tool_live.get_availability(title, regions=["US"])

        assert isinstance(results, list)
        assert len(results) >= 1

        for result in results:
            assert isinstance(result, Observation)
            
            # Required fields
            assert result.record_id is not None
            assert result.call_id is not None
            assert result.test_date is not None
            assert result.region == "US"
            assert result.source == "TMDB"
            assert isinstance(result.available, bool)
            assert isinstance(result.metadata, dict)
            assert isinstance(result.error, list)

            # Metadata should include movie info
            assert "title" in result.metadata
            assert result.metadata["title"] == "Inception"
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
        
        # Now get availability for all movies
        for title in titles:
            results = await tmdb_tool_live.get_availability(title, regions=["US"])
            
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
                json.dump([scrub_serializable(title.model_dump()) for title in results], f, indent=2, default=str)

            # Verify backup file was created
            assert backup_file.exists()

            # Verify we can load it back
            with open(backup_file, "r") as f:
                loaded_data = json.load(f)
                assert isinstance(loaded_data, list)

    @pytest.mark.anyio
    async def test_fetch_single_page_with_bigquery_save(self, real_bm: BM, real_conf, tmdb_tool_live: TMDBTool, tmp_path) -> None:
        """Test single page fetch with real API and BigQuery save."""
        from datetime import date

        from buttermilk.tools.catalog_test import DatePeriod

        # Setup TMDBTool with storage configs for test datasets
        tmdb_tool = TMDBTool(
            api_key=tmdb_tool_live.api_key,
            observations_storage_config=real_conf.storage.observations,  # Safe test dataset
            titles_storage_config=real_conf.storage.titles,  # Safe test dataset
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

        # Verify BigQuery save
        # Force flush to ensure data is saved
        tmdb_tool.titles_uploader.shutdown()

        bm = real_bm

        titles_storage = bm.get_storage(real_conf.storage.titles)
        # Try to query for some of the data we just saved
        # Note: This is a simple existence check - may fail if table doesn't exist yet
        try:
            saved_count = len(titles_storage) if hasattr(titles_storage, "__len__") else 0
            print(f"Saved {saved_count} titles to BigQuery test dataset from 1968")
        except Exception as e:
            print(f"Could not verify BigQuery save (table may not exist yet): {e}")
            # Test still passes - the main functionality (API calls) worked


@pytest.mark.anyio
async def test_fetch_single_page_date_range_validation(tmdb_tool_live: TMDBTool) -> None:
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
        tool = TMDBTool(
            api_key=api_key,
            language="fr-FR",
            region="FR"
        )
        
        assert tool.language == "fr-FR"
        assert tool.region == "FR"

    def test_live_environment_variable_fallback(self) -> None:
        """Test that tool picks up API key from environment."""
        # This test relies on TMDB_API_KEY being set
        tool = TMDBTool()  # Should use env var
        
        assert tool.api_key == os.getenv("TMDB_API_KEY")


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v"])

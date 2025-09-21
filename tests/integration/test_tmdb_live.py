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

from buttermilk.tools.catalog_test import Observation, TMDBTool

# Pytest markers for conditional test execution
pytestmark = [pytest.mark.integration, pytest.mark.live_api]


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
    async def test_live_search_popular_movie(self, tmdb_tool_live: TMDBTool) -> None:
        """Test searching for a popular movie that should have availability."""
        results = await tmdb_tool_live.search_movie_availability(
            title="The Matrix", year=1999, region="US"
        )

        # Verify we get results
        assert isinstance(results, list)
        assert len(results) >= 1

        # Check basic structure of first result
        result = results[0]
        assert isinstance(result, Observation)
        assert result.source == "TMDB"
        assert result.region == "US"
        assert result.match_title is not None
        
        # Should either have availability or explicit unavailability
        if result.available:
            assert result.provider_name is not None
            assert result.provider_id is not None
            assert result.provider_type in ["flatrate", "rent", "buy"]
        else:
            # If not available, should have clear reason in metadata
            assert result.provider_name is None

    @pytest.mark.anyio
    async def test_live_search_nonexistent_movie(self, tmdb_tool_live: TMDBTool) -> None:
        """Test searching for a movie that definitely doesn't exist."""
        results = await tmdb_tool_live.search_movie_availability(
            title="Absolutely Nonexistent Movie Title 9999", region="US"
        )

        # Should return one null observation
        assert isinstance(results, list)
        assert len(results) == 1

        result = results[0]
        assert isinstance(result, Observation)
        assert result.available is False
        assert result.provider_name is None
        assert result.source == "TMDB"
        assert len(result.error) == 0

    @pytest.mark.anyio
    async def test_live_regional_differences(self, tmdb_tool_live: TMDBTool) -> None:
        """Test that the same movie may have different availability in different regions."""
        movie_title = "Pulp Fiction"
        year = 1994

        # Test US availability
        us_results = await tmdb_tool_live.search_movie_availability(
            title=movie_title, year=year, region="US"
        )

        # Test UK availability
        uk_results = await tmdb_tool_live.search_movie_availability(
            title=movie_title, year=year, region="GB"
        )

        # Both should return valid observations
        assert isinstance(us_results, list)
        assert isinstance(uk_results, list)
        assert len(us_results) >= 1
        assert len(uk_results) >= 1

        # Both should find the same movie
        us_result = us_results[0]
        uk_result = uk_results[0]
        
        assert us_result.match_title == uk_result.match_title
        assert us_result.region == "US"
        assert uk_result.region == "GB"

        # Availability may differ between regions
        # (This is the key insight - same content, different regional licensing)

    @pytest.mark.anyio
    async def test_live_search_with_year_filtering(self, tmdb_tool_live: TMDBTool) -> None:
        """Test that year filtering works correctly."""
        # Search for "Batman" without year (should get recent results)
        recent_results = await tmdb_tool_live.search_movie_availability(
            title="Batman", region="US"
        )

        # Search for "Batman" from 1989 (specific Tim Burton film)
        year_filtered_results = await tmdb_tool_live.search_movie_availability(
            title="Batman", year=1989, region="US"
        )

        assert isinstance(recent_results, list)
        assert isinstance(year_filtered_results, list)
        assert len(recent_results) >= 1
        assert len(year_filtered_results) >= 1

        # The 1989 result should specifically match that year's Batman
        if year_filtered_results[0].available or len(year_filtered_results[0].error) == 0:
            # Should have found the 1989 Batman
            result = year_filtered_results[0]
            assert "Batman" in result.match_title
            # Metadata should include the search year
            assert result.metadata.get("search_year") == 1989

    @pytest.mark.anyio
    async def test_live_error_handling_invalid_region(self, tmdb_tool_live: TMDBTool) -> None:
        """Test error handling with invalid region codes."""
        results = await tmdb_tool_live.search_movie_availability(
            title="The Matrix", region="INVALID"
        )

        # Should handle gracefully - either return no availability or proper error
        assert isinstance(results, list)
        assert len(results) >= 1

        result = results[0]
        assert isinstance(result, Observation)
        assert result.region == "INVALID"
        
        # Either no availability (expected) or a handled error
        if not result.available:
            # Should have clear indication why not available
            assert result.provider_name is None

    @pytest.mark.anyio
    async def test_live_api_response_structure(self, tmdb_tool_live: TMDBTool) -> None:
        """Test that live API responses match our expected data structure."""
        results = await tmdb_tool_live.search_movie_availability(
            title="Inception", year=2010, region="US"
        )

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

            # Metadata should include search info
            assert "search_title" in result.metadata
            assert result.metadata["search_title"] == "Inception"

            if result.available:
                # Available results should have provider info
                assert result.provider_name is not None
                assert result.provider_id is not None
                assert result.provider_type in ["flatrate", "rent", "buy"]
                assert result.match_title is not None
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

        all_results = []
        
        # Make multiple requests in sequence
        for title, year in movies:
            results = await tmdb_tool_live.search_movie_availability(
                title=title, year=year, region="US"
            )
            all_results.extend(results)

        # All requests should succeed (no rate limit errors)
        assert len(all_results) >= len(movies)
        
        for result in all_results:
            assert isinstance(result, Observation)
            assert result.source == "TMDB"
            # No rate limiting errors
            if result.error:
                error_messages = [str(err) for err in result.error]
                for msg in error_messages:
                    assert "rate limit" not in msg.lower()
                    assert "too many requests" not in msg.lower()


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

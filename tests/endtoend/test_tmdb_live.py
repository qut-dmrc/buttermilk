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
    async def test_live_search_popular_movie(self, tmdb_tool_live: TMDBTool) -> None:
        """Test searching for a popular movie and checking availability."""
        # First search for the movie
        title = await tmdb_tool_live.search_movie(title="The Matrix", year=1999)
        
        # Verify we got a Title object
        assert isinstance(title, Title)
        assert title.title is not None
        assert title.record_id is not None
        assert title.year == 1999
        
        # Now get availability for US region
        results = await tmdb_tool_live.get_availability(title, regions=["US"])
        
        # Verify we get results
        assert isinstance(results, list)
        assert len(results) >= 1

        # Check basic structure of observations
        for result in results:
            assert isinstance(result, Observation)
            assert result.source == "TMDB"
            assert result.region == "US"
            assert not result.error  # No errors expected

            # Should either have availability or explicit unavailability
            if result.available:
                assert result.provider_name is not None
                assert result.provider_id is not None
                assert result.provider_type in ["flatrate", "rent", "buy"]
            else:
                # If not available, should have clear reason
                assert result.provider_name is None

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
        """Test discovering movies from TMDB with JSON backup."""
        # Get a small batch of popular movies
        results = await tmdb_tool_live.get_all_movies(
            backup_dir=tmp_path,
            max_results=5,
            sort_by="popularity.desc"
        )
        
        # Should get some results
        assert isinstance(results, list)
        assert len(results) <= 5
        
        # Check each result
        for title in results:
            assert isinstance(title, Title)
            assert title.record_id is not None
            assert title.title is not None
            assert title.type == TitleType.MOVIE
            
            # Year should be reasonable if present
            if title.year:
                assert 1900 <= title.year <= 2030  # Reasonable movie year range
            
            # Check backup file exists
            backup_file = tmp_path / f"movie_{title.record_id}.json"
            assert backup_file.exists()
            
            # Verify backup content
            import json
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            assert str(backup_data["id"]) == title.record_id
            assert backup_data["title"] == title.title


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

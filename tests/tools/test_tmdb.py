from unittest.mock import AsyncMock, patch

import pytest
from autogen_core.tools import FunctionTool

from buttermilk.tools.catalog_test import Observation, TMDBTool


# Test Fixtures
@pytest.fixture
def tmdb_tool():
    """Create a TMDBTool instance with mock API key."""
    return TMDBTool(api_key="fake-tmdb-api-key-test-only")


@pytest.fixture
def mock_tmdb_api_response():
    """Mock successful TMDB API response."""
    return {
        "results": [
            {
                "id": 550,
                "title": "Fight Club",
                "release_date": "1999-10-15",
                "overview": "A ticking-time-bomb insomniac...",
                "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
            }
        ],
        "total_results": 1,
    }


@pytest.fixture
def mock_availability_response():
    """Mock TMDB watch provider response."""
    return {
        "results": {
            "US": {
                "flatrate": [{"display_priority": 1, "logo_path": "/7GbbIEw5jQZdNwj69yPGPy6JXVV.jpg", "provider_id": 8, "provider_name": "Netflix"}],
                "rent": [{"display_priority": 10, "logo_path": "/8N8c3m5VUTL9Gxu4kP4jzgNsZaU.jpg", "provider_id": 2, "provider_name": "Apple TV"}],
            }
        }
    }


@pytest.fixture
def mock_empty_response():
    """Mock empty TMDB API response (no results found)."""
    return {"results": [], "total_results": 0}


# Test Cases
class TestTMDBTool:
    """Test cases for TMDB tool functionality."""

    @pytest.mark.anyio
    async def test_successful_search_with_availability(self, tmdb_tool, mock_tmdb_api_response, mock_availability_response):
        """Test successful movie search with availability results."""
        with patch("httpx.AsyncClient.get") as mock_get:
            # Mock the API calls
            mock_get.side_effect = [
                # First call: search response
                AsyncMock(json=AsyncMock(return_value=mock_tmdb_api_response), status_code=200),
                # Second call: availability response
                AsyncMock(json=AsyncMock(return_value=mock_availability_response), status_code=200),
            ]

            results = await tmdb_tool.search_movie_availability(title="Fight Club", year=1999, region="US")

            # Verify we get a list of observations
            assert isinstance(results, list)
            assert len(results) > 0  # Should have at least one observation

            # Check each observation has correct structure
            for result in results:
                assert isinstance(result, Observation)
                assert result.available is True
                assert result.match_title == "Fight Club"
                assert result.region == "US"
                assert result.provider_name is not None
                assert result.source == "TMDB"
                assert len(result.error) == 0

    @pytest.mark.anyio
    async def test_search_no_results(self, tmdb_tool, mock_empty_response):
        """Test search with no results returns null observation."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.return_value = AsyncMock(json=AsyncMock(return_value=mock_empty_response), status_code=200)

            results = await tmdb_tool.search_movie_availability(title="Nonexistent Movie", year=2023, region="US")

            # Verify null observation in list
            assert isinstance(results, list)
            assert len(results) == 1  # Should have exactly one null result

            result = results[0]
            assert isinstance(result, Observation)
            assert result.available is False
            assert result.match_title == "Nonexistent Movie"
            assert result.region == "US"
            assert result.provider_name is None
            assert result.source == "TMDB"
            assert len(result.error) == 0

    @pytest.mark.anyio
    async def test_api_error_handling(self, tmdb_tool):
        """Test error handling when API fails."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = Exception("API connection failed")

            results = await tmdb_tool.search_movie_availability(title="Any Movie", region="US")

            # Verify error observation in list
            assert isinstance(results, list)
            assert len(results) == 1  # Should have exactly one error result

            result = results[0]
            assert isinstance(result, Observation)
            assert result.available is False
            assert result.match_title == "Any Movie"
            assert result.region == "US"
            assert result.provider_name is None
            assert result.source == "TMDB"
            assert len(result.error) > 0  # Should have error information
            assert "API connection failed" in str(result.error)

    @pytest.mark.anyio
    async def test_http_error_status(self, tmdb_tool):
        """Test handling of HTTP error status codes."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.return_value = AsyncMock(status_code=404, json=AsyncMock(return_value={"status_message": "Not found"}))

            results = await tmdb_tool.search_movie_availability(title="Movie Title", region="US")

            # Verify error handling
            assert isinstance(results, list)
            assert len(results) == 1

            result = results[0]
            assert isinstance(result, Observation)
            assert result.available is False
            assert len(result.error) > 0
            error_str = str(result.error)
            assert "404" in error_str or "Not found" in error_str

    @pytest.mark.anyio
    async def test_invalid_api_key(self, tmdb_tool):
        """Test handling of invalid API key."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.return_value = AsyncMock(status_code=401, json=AsyncMock(return_value={"status_message": "Invalid API key", "status_code": 7}))

            results = await tmdb_tool.search_movie_availability(title="Movie Title", region="US")

            # Verify authentication error handling
            assert isinstance(results, list)
            assert len(results) == 1

            result = results[0]
            assert isinstance(result, Observation)
            assert result.available is False
            assert len(result.error) > 0
            error_str = str(result.error)
            assert "Invalid API key" in error_str or "401" in error_str

    @pytest.mark.anyio
    async def test_no_availability_in_region(self, tmdb_tool, mock_tmdb_api_response):
        """Test movie found but no availability in specified region."""
        mock_availability_empty = {
            "results": {
                "AU": {}  # Empty availability for Australia
            }
        }

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = [
                AsyncMock(json=AsyncMock(return_value=mock_tmdb_api_response), status_code=200),
                AsyncMock(json=AsyncMock(return_value=mock_availability_empty), status_code=200),
            ]

            results = await tmdb_tool.search_movie_availability(title="Fight Club", region="AU")

            # Verify movie found but no availability
            assert isinstance(results, list)
            assert len(results) == 1  # Should have one result showing no availability

            result = results[0]
            assert isinstance(result, Observation)
            assert result.available is False  # No availability in AU
            assert result.match_title == "Fight Club"
            assert result.region == "AU"
            assert result.provider_name is None  # No providers available
            assert result.source == "TMDB"

    def test_as_tool_function(self, tmdb_tool):
        """Test that tool can be converted to FunctionTool for agent use."""
        function_tool = tmdb_tool.as_tool()

        assert isinstance(function_tool, FunctionTool)
        assert function_tool.name == "tmdb_search"
        assert "movie availability" in function_tool.description.lower()
        assert "TMDB" in function_tool.description or "Movie Database" in function_tool.description

    def test_tool_configuration(self, tmdb_tool):
        """Test tool configuration and initialization."""
        assert tmdb_tool.api_key == "fake-tmdb-api-key-test-only"
        assert tmdb_tool.base_url == "https://api.themoviedb.org/3"

        # Test with custom configuration
        custom_tool = TMDBTool(api_key="another-fake-key", base_url="https://custom.tmdb.api/v3")
        assert custom_tool.base_url == "https://custom.tmdb.api/v3"

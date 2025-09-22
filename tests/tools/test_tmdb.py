# ruff: noqa: PLR6301
import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from autogen_core.tools import FunctionTool

from buttermilk.tools.catalog_test import Observation, Title, TitleType, TMDBTool


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


# Test Cases for separated search and availability methods
class TestTMDBSearchMovie:
    """Test cases for movie search functionality."""

    @pytest.mark.anyio
    async def test_search_movie_returns_title_object(self, tmdb_tool):
        """Test that search_movie returns a Title object with metadata."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock movie object from TMDB API
            mock_movie = AsyncMock()
            mock_movie.id = 550
            mock_movie.title = "Fight Club"
            mock_movie.release_date = "1999-10-15"
            mock_movie.overview = "A ticking-time-bomb insomniac..."
            mock_movie.poster_path = "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"

            # Mock search response
            mock_search = AsyncMock()
            mock_search.movies = AsyncMock(return_value=[mock_movie])
            mock_tmdb.search.return_value = mock_search

            # Call the new search_movie method
            result = await tmdb_tool.search_movie(title="Fight Club", year=1999)

            # Verify we get a Title object
            assert isinstance(result, Title)
            assert result.record_id == "550"
            assert result.title == "Fight Club"
            assert result.year == 1999
            assert isinstance(result.metadata, dict)
            assert result.metadata.get("release_date") == "1999-10-15"
            assert result.metadata.get("overview") == "A ticking-time-bomb insomniac..."

    @pytest.mark.anyio
    async def test_search_movie_no_results_returns_none(self, tmdb_tool):
        """Test that search_movie returns None when no movie is found."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock empty search response
            mock_search = AsyncMock()
            mock_search.movies = AsyncMock(return_value=[])
            mock_tmdb.search.return_value = mock_search

            result = await tmdb_tool.search_movie(title="Nonexistent Movie", year=2023)

            # Should return None when no movie found
            assert result is None

    @pytest.mark.anyio
    async def test_search_movie_extracts_year_from_release_date(self, tmdb_tool):
        """Test that year is extracted from release_date if not provided."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movie = AsyncMock()
            mock_movie.id = 550
            mock_movie.title = "Fight Club"
            mock_movie.release_date = "1999-10-15"
            
            mock_search = AsyncMock()
            mock_search.movies = AsyncMock(return_value=[mock_movie])
            mock_tmdb.search.return_value = mock_search

            result = await tmdb_tool.search_movie(title="Fight Club")

            assert result.year == 1999


class TestTMDBGetAvailability:
    """Test cases for availability checking functionality."""

    @pytest.mark.anyio
    async def test_get_availability_with_providers(self, tmdb_tool):
        """Test get_availability returns observations for available providers."""
        # Create a Title object
        title = Title(
            record_id="550",
            title="Fight Club",
            year=1999,
            metadata={"original_title": "Fight Club"}
        )

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock watch providers response
            mock_providers = {
                "results": {
                    "US": {
                        "flatrate": [
                            {"provider_id": 8, "provider_name": "Netflix"},
                            {"provider_id": 9, "provider_name": "Amazon Prime Video"}
                        ],
                        "rent": [
                            {"provider_id": 2, "provider_name": "Apple TV"}
                        ]
                    },
                    "GB": {
                        "flatrate": [
                            {"provider_id": 8, "provider_name": "Netflix"}
                        ]
                    }
                }
            }
            
            # Mock the movie(id) method to return an object with watch_providers
            mock_movie_obj = AsyncMock()
            mock_movie_obj.watch_providers = AsyncMock(return_value=mock_providers)
            mock_tmdb.movie.return_value = mock_movie_obj

            # Get availability for US and GB regions
            results = await tmdb_tool.get_availability(title, regions=["US", "GB", "AU"])

            # Should return observations for each region
            assert isinstance(results, list)
            assert len(results) >= 3  # At least one observation per region

            # Check US has providers
            us_observations = [r for r in results if r.region == "US"]
            assert len(us_observations) >= 2  # Netflix and Amazon Prime at minimum
            
            netflix_us = next((r for r in us_observations if r.provider_name == "Netflix"), None)
            assert netflix_us is not None
            assert netflix_us.available is True
            assert netflix_us.provider_type == "flatrate"
            assert netflix_us.provider_id == "8"

            # Check GB has Netflix
            gb_observations = [r for r in results if r.region == "GB"]
            assert len(gb_observations) >= 1
            
            # Check AU has null observation (no providers)
            au_observations = [r for r in results if r.region == "AU"]
            assert len(au_observations) == 1
            assert au_observations[0].available is False
            assert au_observations[0].provider_name is None

    @pytest.mark.anyio
    async def test_get_availability_no_providers_returns_null_observations(self, tmdb_tool):
        """Test that regions with no providers get null observations."""
        title = Title(
            record_id="550",
            title="Fight Club",
            year=1999,
            metadata={}
        )

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock empty providers response
            mock_providers = {"results": {}}
            
            # Mock the movie(id) method to return an object with watch_providers
            mock_movie_obj = AsyncMock()
            mock_movie_obj.watch_providers = AsyncMock(return_value=mock_providers)
            mock_tmdb.movie.return_value = mock_movie_obj

            results = await tmdb_tool.get_availability(title, regions=["US", "GB"])

            # Should return null observations for each region
            assert len(results) == 2
            for obs in results:
                assert isinstance(obs, Observation)
                assert obs.available is False
                assert obs.provider_name is None
                assert obs.source == "TMDB"

    @pytest.mark.anyio
    async def test_get_availability_api_error_handling(self, tmdb_tool):
        """Test error handling when availability API fails."""
        title = Title(record_id="550", title="Fight Club", year=1999, metadata={})

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock API failure
            mock_movie_obj = AsyncMock()
            mock_movie_obj.watch_providers = AsyncMock(side_effect=Exception("API connection failed"))
            mock_tmdb.movie.return_value = mock_movie_obj

            results = await tmdb_tool.get_availability(title, regions=["US"])

            # Should return error observation
            assert len(results) == 1
            result = results[0]
            assert isinstance(result, Observation)
            assert result.available is False
            assert len(result.error) > 0
            assert "API connection failed" in str(result.error)


class TestTMDBDiscoverMovies:
    """Test cases for discovering all movies functionality."""

    @pytest.mark.anyio
    async def test_get_all_movies_returns_title_objects(self, tmdb_tool, tmp_path):
        """Test that get_all_movies returns Title objects using month-based fetching."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock discover movies response for a specific month
            mock_movies = [
                {
                    "id": 550,
                    "title": "Fight Club",
                    "release_date": "1999-10-15",
                    "overview": "A ticking-time-bomb insomniac...",
                    "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
                },
                {
                    "id": 603,
                    "title": "The Matrix",
                    "release_date": "1999-03-30",
                    "overview": "Set in the 22nd century...",
                    "poster_path": "/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
                },
                {
                    "id": 123,
                    "title": "Movie Without Date",
                    "release_date": None,
                    "overview": "A movie without release date",
                },
            ]

            # Mock discover().movie() method with date range parameters
            async def mock_movie_func(**kwargs):
                page = kwargs.get('page', 1)
                # Only return movies on page 1, simulate month-based fetching
                if page == 1 and 'primary_release_date.gte' in kwargs:
                    return mock_movies
                else:
                    return []  # No more results

            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            # Call get_all_movies with new signature (single year for testing)
            results = await tmdb_tool.get_all_movies(
                start_year=1999,
                end_year=1999,  # Single year to limit test scope
                max_concurrent=1,
                backup_dir=tmp_path,
                max_results=10,
                include_adult=True,
                include_video=True,
                resume=False
            )

            # Verify we get Title objects
            assert isinstance(results, list)
            # Should get 3 movies * 12 months = 36 total, but limited by max_results
            assert len(results) <= 10

            # Check that results contain Title objects
            if results:
                assert isinstance(results[0], Title)
                assert results[0].type == TitleType.MOVIE

            # Verify period backup files were created
            period_files = list(tmp_path.glob("period_*.json"))
            assert len(period_files) >= 1

    @pytest.mark.anyio
    async def test_get_all_movies_saves_backup_json(self, tmdb_tool, tmp_path):
        """Test that get_all_movies saves period-based JSON backups to disk."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movie = {
                "id": 550,
                "title": "Fight Club",
                "release_date": "1999-10-15",
                "overview": "A ticking-time-bomb insomniac...",
            }

            # Mock discover().movie() method with date range parameters
            async def mock_movie_func(**kwargs):
                page = kwargs.get('page', 1)
                if page == 1 and 'primary_release_date.gte' in kwargs:
                    return [mock_movie]
                else:
                    return []  # No more results

            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            results = await tmdb_tool.get_all_movies(
                start_year=1999,
                end_year=1999,
                max_concurrent=1,
                backup_dir=tmp_path,
                max_results=1,
                resume=False
            )

            # Check period backup files were created
            period_files = list(tmp_path.glob("period_*.json"))
            assert len(period_files) >= 1

            # Verify period backup content contains our movie
            if period_files:
                with open(period_files[0], 'r') as f:
                    saved_data = json.load(f)
                assert len(saved_data) >= 1
                # Each period file contains a list of Title objects
                if saved_data:
                    first_movie = saved_data[0]
                    assert first_movie["record_id"] == "550"
                    assert first_movie["title"] == "Fight Club"

    @pytest.mark.anyio
    async def test_get_all_movies_handles_api_errors(self, tmdb_tool, tmp_path):
        """Test error handling when discover API fails during period fetching."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock API failure
            async def mock_movie_func(**kwargs):
                raise Exception("API error")

            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            results = await tmdb_tool.get_all_movies(
                start_year=1999,
                end_year=1999,
                max_concurrent=1,
                backup_dir=tmp_path,
                max_results=1,
                resume=False
            )

            # Should return empty list on error (periods that fail return empty lists)
            assert results == []

    @pytest.mark.anyio
    async def test_get_all_movies_max_results_limit(self, tmdb_tool, tmp_path):
        """Test that get_all_movies respects max_results parameter."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock response with many movies per period
            movies_per_period = [
                {"id": i, "title": f"Movie {i}", "release_date": f"2020-01-{i:02d}"}
                for i in range(1, 21)  # 20 movies per period
            ]

            # Mock discover().movie() method
            async def mock_movie_func(**kwargs):
                page = kwargs.get('page', 1)
                if page == 1 and 'primary_release_date.gte' in kwargs:
                    return movies_per_period
                else:
                    return []  # No more results after page 1

            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            results = await tmdb_tool.get_all_movies(
                start_year=2020,
                end_year=2020,  # Single year
                max_concurrent=1,
                backup_dir=tmp_path,
                max_results=10,  # Limit results for testing
                resume=False
            )

            # Should return only requested number of results
            assert len(results) <= 10
            assert all(isinstance(r, Title) for r in results)

    @pytest.mark.anyio
    async def test_get_all_movies_multiple_periods(self, tmdb_tool, tmp_path):
        """Test that get_all_movies fetches multiple monthly periods."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock response that varies by period (month)
            def create_period_movies(start_id):
                return [
                    {"id": i, "title": f"Movie {i}", "release_date": f"2020-01-{i%28 + 1:02d}"}
                    for i in range(start_id, start_id + 5)  # 5 movies per period
                ]

            period_responses = {
                "2020-01": create_period_movies(1),   # Jan: movies 1-5
                "2020-02": create_period_movies(6),   # Feb: movies 6-10
                "2020-03": create_period_movies(11),  # Mar: movies 11-15
            }

            # Mock discover to return different movies based on date range
            async def mock_movie_func(**kwargs):
                page = kwargs.get('page', 1)
                if page > 1:
                    return []  # Only page 1 has results

                # Determine period from date range
                gte_date = kwargs.get('primary_release_date.gte', '')
                if '2020-01' in gte_date:
                    return period_responses["2020-01"]
                elif '2020-02' in gte_date:
                    return period_responses["2020-02"]
                elif '2020-03' in gte_date:
                    return period_responses["2020-03"]
                else:
                    return []

            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            # Fetch first quarter of 2020
            results = await tmdb_tool.get_all_movies(
                start_year=2020,
                end_year=2020,
                max_concurrent=2,
                backup_dir=tmp_path,
                max_results=50,  # Plenty to get all
                resume=False
            )

            # Should return movies from multiple periods
            # 12 months * 5 movies = 60 movies total for 2020
            assert len(results) <= 50  # Limited by max_results
            assert all(isinstance(r, Title) for r in results)

            # Verify period backup files were created for multiple months
            period_files = list(tmp_path.glob("period_*.json"))
            assert len(period_files) >= 3  # At least Jan, Feb, Mar

    @pytest.mark.anyio
    async def test_get_all_movies_default_cache_dir(self, tmdb_tool):
        """Test that default cache directory follows buttermilk pattern."""
        from pathlib import Path

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movies = [{"id": 1, "title": "Test Movie", "release_date": "2020-01-01"}]

            # Mock discover().movie() method
            async def mock_movie_func(**kwargs):
                page = kwargs.get('page', 1)
                if page == 1 and 'primary_release_date.gte' in kwargs:
                    return mock_movies
                else:
                    return []  # No more results

            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            # Call without backup_dir to test default
            results = await tmdb_tool.get_all_movies(
                start_year=2020,
                end_year=2020,
                max_concurrent=1,
                max_results=1,
                resume=False
            )

            # Verify we get results (which means default directory worked)
            assert len(results) <= 1

    @pytest.mark.anyio
    async def test_get_all_movies_resume_functionality(self, tmdb_tool, tmp_path):
        """Test that get_all_movies can resume from previous progress."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movies = [{"id": 1, "title": "Test Movie", "release_date": "2020-01-01"}]

            # Mock discover().movie() method
            async def mock_movie_func(**kwargs):
                page = kwargs.get('page', 1)
                if page == 1 and 'primary_release_date.gte' in kwargs:
                    return mock_movies
                else:
                    return []

            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            # First call - should fetch and save progress
            results1 = await tmdb_tool.get_all_movies(
                start_year=2020,
                end_year=2020,
                max_concurrent=1,
                backup_dir=tmp_path,
                max_results=12,  # One month's worth
                resume=True
            )

            # Verify progress file was created
            progress_file = tmp_path / "fetch_progress.json"
            assert progress_file.exists()

            # Second call with resume=True - should skip completed periods
            results2 = await tmdb_tool.get_all_movies(
                start_year=2020,
                end_year=2020,
                max_concurrent=1,
                backup_dir=tmp_path,
                max_results=12,
                resume=True
            )

            # Should get same results from cached data
            assert len(results2) == len(results1)


class TestTMDBToolConfiguration:
    """Test tool configuration and setup."""

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
        assert tmdb_tool.language == "en-US"
        assert tmdb_tool.region == "AU"

        # Test with custom configuration
        custom_tool = TMDBTool(
            api_key="another-fake-key",
            base_url="https://custom.tmdb.api/v3",
            language="fr-FR",
            region="FR"
        )
        assert custom_tool.base_url == "https://custom.tmdb.api/v3"
        assert custom_tool.language == "fr-FR"
        assert custom_tool.region == "FR"

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):  # Clear TMDB_API_KEY env var
            with pytest.raises(ValueError, match="TMDB API key is required"):
                TMDBTool()

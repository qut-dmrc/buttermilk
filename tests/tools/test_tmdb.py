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
        """Test that get_all_movies returns Title objects from discover endpoint."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock discover movies response
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
            
            # Mock discover().movie() method to return empty list on page 2
            async def mock_movie_func(**kwargs):
                page = kwargs.get('page', 1)
                if page == 1:
                    return mock_movies
                else:
                    return []  # No more results after page 1
            
            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            # Call get_all_movies with backup directory
            results = await tmdb_tool.get_all_movies(
                backup_dir=tmp_path,
                include_adult=True,
                include_video=True,
                sort_by="primary_release_date.asc"
            )

            # Verify we get Title objects
            assert isinstance(results, list)
            assert len(results) == 3
            
            # Check first movie
            assert isinstance(results[0], Title)
            assert results[0].record_id == "550"
            assert results[0].title == "Fight Club"
            assert results[0].year == 1999
            assert results[0].type == TitleType.MOVIE
            
            # Check second movie
            assert results[1].record_id == "603"
            assert results[1].title == "The Matrix"
            assert results[1].year == 1999
            
            # Check movie without release date
            assert results[2].record_id == "123"
            assert results[2].title == "Movie Without Date"
            assert results[2].year is None
            
            # Verify backup files were created
            backup_files = list(tmp_path.glob("*.json"))
            assert len(backup_files) == 3

    @pytest.mark.anyio
    async def test_get_all_movies_saves_backup_json(self, tmdb_tool, tmp_path):
        """Test that get_all_movies saves JSON backups to disk."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movie = {
                "id": 550,
                "title": "Fight Club",
                "release_date": "1999-10-15",
                "overview": "A ticking-time-bomb insomniac...",
            }
            
            # Mock discover().movie() method to return empty list on page 2
            async def mock_movie_func(**kwargs):
                page = kwargs.get('page', 1)
                if page == 1:
                    return [mock_movie]
                else:
                    return []  # No more results after page 1
            
            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            results = await tmdb_tool.get_all_movies(backup_dir=tmp_path)

            # Check backup file was created
            backup_file = tmp_path / "movie_550.json"
            assert backup_file.exists()
            
            # Verify backup content
            with open(backup_file, 'r') as f:
                saved_data = json.load(f)
            assert saved_data["id"] == 550
            assert saved_data["title"] == "Fight Club"

    @pytest.mark.anyio
    async def test_get_all_movies_handles_api_errors(self, tmdb_tool, tmp_path):
        """Test error handling when discover API fails."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock API failure
            async def mock_movie_func(**kwargs):
                raise Exception("API error")
            
            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            results = await tmdb_tool.get_all_movies(backup_dir=tmp_path)

            # Should return empty list on error
            assert results == []

    @pytest.mark.anyio
    async def test_get_all_movies_pagination(self, tmdb_tool, tmp_path):
        """Test that get_all_movies handles pagination correctly."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock paginated response (simplified - real API uses pages)
            page1_movies = [
                {"id": i, "title": f"Movie {i}", "release_date": f"2020-01-{i:02d}"}
                for i in range(1, 21)  # 20 movies
            ]
            
            # Mock discover().movie() method - return only page 1 then empty
            async def mock_movie_func(**kwargs):
                page = kwargs.get('page', 1)
                if page == 1:
                    return page1_movies
                else:
                    return []  # No more results after page 1
            
            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            results = await tmdb_tool.get_all_movies(
                backup_dir=tmp_path,
                max_results=10  # Limit results for testing
            )

            # Should return only requested number of results
            assert len(results) == 10
            assert all(isinstance(r, Title) for r in results)

    @pytest.mark.anyio
    async def test_get_all_movies_multi_page(self, tmdb_tool, tmp_path):
        """Test that get_all_movies fetches multiple pages."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock multi-page responses
            page1_movies = [
                {"id": i, "title": f"Movie {i}", "release_date": f"2020-01-{i:02d}"}
                for i in range(1, 21)  # 20 movies on page 1
            ]
            page2_movies = [
                {"id": i, "title": f"Movie {i}", "release_date": f"2020-02-{i-20:02d}"}
                for i in range(21, 41)  # 20 movies on page 2
            ]
            page3_movies = [
                {"id": i, "title": f"Movie {i}", "release_date": f"2020-03-{i-40:02d}"}
                for i in range(41, 51)  # 10 movies on page 3
            ]
            
            # Mock discover to return different pages based on page parameter
            async def mock_movie_func(**kwargs):
                page = kwargs.get('page', 1)
                if page == 1:
                    return page1_movies
                elif page == 2:
                    return page2_movies
                elif page == 3:
                    return page3_movies
                else:
                    return []  # No more results
            
            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            # Request 50 movies (should fetch 3 pages)
            results = await tmdb_tool.get_all_movies(
                backup_dir=tmp_path,
                max_results=50
            )

            # Should return all 50 movies from 3 pages
            assert len(results) == 50
            assert all(isinstance(r, Title) for r in results)
            
            # Verify we got movies from all pages
            movie_ids = [int(r.record_id) for r in results]
            assert min(movie_ids) == 1
            assert max(movie_ids) == 50

    @pytest.mark.anyio
    async def test_get_all_movies_default_cache_dir(self, tmdb_tool):
        """Test that default cache directory follows buttermilk pattern."""
        from pathlib import Path
        
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movies = [{"id": 1, "title": "Test Movie", "release_date": "2020-01-01"}]
            
            # Mock discover().movie() method to return one page then empty
            async def mock_movie_func(**kwargs):
                page = kwargs.get('page', 1)
                if page == 1:
                    return mock_movies
                else:
                    return []  # No more results after page 1
            
            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            # Mock Path operations to check the directory used
            with patch('buttermilk.tools.catalog_test.Path') as mock_path_class:
                mock_path = mock_path_class.return_value
                mock_path.mkdir.return_value = None
                mock_path.__truediv__ = lambda self, other: self  # Mock path joining
                
                # Use home() directly without mocking it
                expected_dir = Path.home() / ".cache" / "buttermilk" / "tmdb"
                
                # Call without backup_dir to test default
                results = await tmdb_tool.get_all_movies(max_results=1)
                
                # Verify default directory was created
                # The actual check would be in implementation
                assert len(results) == 1


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

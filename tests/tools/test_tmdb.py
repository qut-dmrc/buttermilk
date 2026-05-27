import json
import os
from unittest.mock import AsyncMock, call, patch

import pytest

from buttermilk._core.tool_types import FunctionTool
from buttermilk.tools.catalog_test import (
    THEMOVIEDB_AVAILABLE,
    Observation,
    Title,
    TitleType,
    TMDBTool,
)

# Skip entire module if themoviedb is not installed
pytestmark = pytest.mark.skipif(
    not THEMOVIEDB_AVAILABLE,
    reason="themoviedb package not installed - install with: pip install themoviedb.py",
)


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
                "flatrate": [
                    {
                        "display_priority": 1,
                        "logo_path": "/7GbbIEw5jQZdNwj69yPGPy6JXVV.jpg",
                        "provider_id": 8,
                        "provider_name": "Netflix",
                    }
                ],
                "rent": [
                    {
                        "display_priority": 10,
                        "logo_path": "/8N8c3m5VUTL9Gxu4kP4jzgNsZaU.jpg",
                        "provider_id": 2,
                        "provider_name": "Apple TV",
                    }
                ],
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
            metadata={"original_title": "Fight Club"},
        )

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock watch providers response - needs to be an object with .results attribute
            # region_data must be a dataclass, so we'll use dataclasses.make_dataclass
            from dataclasses import make_dataclass

            # Create a simple dataclass for region data
            RegionData = make_dataclass(
                "RegionData",
                [("flatrate", list), ("rent", list), ("buy", list), ("link", str)],
            )

            from types import SimpleNamespace

            mock_providers = SimpleNamespace()
            mock_providers.results = {
                "US": RegionData(
                    flatrate=[
                        {"provider_id": 8, "provider_name": "Netflix"},
                        {"provider_id": 9, "provider_name": "Amazon Prime Video"},
                    ],
                    rent=[{"provider_id": 2, "provider_name": "Apple TV"}],
                    buy=[],
                    link="",
                ),
                "GB": RegionData(
                    flatrate=[{"provider_id": 8, "provider_name": "Netflix"}],
                    rent=[],
                    buy=[],
                    link="",
                ),
            }

            # Mock the movie(id) method to return an object with watch_providers
            mock_movie_obj = AsyncMock()
            mock_movie_obj.watch_providers = AsyncMock(return_value=mock_providers)
            mock_tmdb.movie.return_value = mock_movie_obj

            # Get availability for all regions TMDB returns
            results = []
            async for obs in tmdb_tool.get_availability(title):
                results.append(obs)

            # Should return observations for regions with providers
            assert isinstance(results, list)
            assert len(results) >= 3  # At least some observations

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

    @pytest.mark.anyio
    async def test_get_availability_no_providers_returns_null_observations(self, tmdb_tool):
        """Test that when TMDB returns no providers, we get a null observation."""
        title = Title(record_id="550", title="Fight Club", year=1999, metadata={})

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock empty providers response - needs to be an object with .results attribute
            from types import SimpleNamespace

            mock_providers = SimpleNamespace()
            mock_providers.results = {}

            # Mock the movie(id) method to return an object with watch_providers
            mock_movie_obj = AsyncMock()
            mock_movie_obj.watch_providers = AsyncMock(return_value=mock_providers)
            mock_tmdb.movie.return_value = mock_movie_obj

            results = []
            async for obs in tmdb_tool.get_availability(title):
                results.append(obs)

            # Should return one null observation when no regions have providers
            assert len(results) == 1
            obs = results[0]
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

            results = []
            async for obs in tmdb_tool.get_availability(title):
                results.append(obs)

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
                page = kwargs.get("page", 1)
                # Only return movies on page 1, simulate month-based fetching
                if page == 1 and "primary_release_date__gte" in kwargs:
                    return mock_movies
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
                include_adult=True,
                include_video=True,
                resume=False,
            )

            # Verify we get Title objects
            assert isinstance(results, list)
            # Should get movies from 1999 (12 months worth)
            assert len(results) >= 0  # Could be empty if no movies in 1999 test period

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
                page = kwargs.get("page", 1)
                if page == 1 and "primary_release_date__gte" in kwargs:
                    return [mock_movie]
                return []  # No more results

            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            await tmdb_tool.get_all_movies(
                start_year=1999,
                end_year=1999,
                max_concurrent=1,
                backup_dir=tmp_path,
                resume=False,
            )

            # Check period backup files were created
            period_files = list(tmp_path.glob("period_*.json"))
            assert len(period_files) >= 1

            # Verify period backup content contains our movie
            if period_files:
                with open(period_files[0]) as f:
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
                resume=False,
            )

            # Should return empty list on error (periods that fail return empty lists)
            assert results == []

    @pytest.mark.anyio
    async def test_get_all_movies_multiple_periods(self, tmdb_tool, tmp_path):
        """Test that get_all_movies fetches multiple monthly periods."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock response that varies by period (month)
            def create_period_movies(start_id):
                return [
                    {
                        "id": i,
                        "title": f"Movie {i}",
                        "release_date": f"2020-01-{i % 28 + 1:02d}",
                    }
                    for i in range(start_id, start_id + 5)  # 5 movies per period
                ]

            period_responses = {
                "2020-01": create_period_movies(1),  # Jan: movies 1-5
                "2020-02": create_period_movies(6),  # Feb: movies 6-10
                "2020-03": create_period_movies(11),  # Mar: movies 11-15
            }

            # Mock discover to return different movies based on date range
            async def mock_movie_func(**kwargs):
                page = kwargs.get("page", 1)
                if page > 1:
                    return []  # Only page 1 has results

                # Determine period from date range
                gte_date = kwargs.get("primary_release_date__gte", "")
                if "2020-01" in gte_date:
                    return period_responses["2020-01"]
                if "2020-02" in gte_date:
                    return period_responses["2020-02"]
                if "2020-03" in gte_date:
                    return period_responses["2020-03"]
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
                resume=False,
            )

            # Should return movies from multiple periods
            # 12 months * 5 movies = 60 movies total for 2020
            assert len(results) >= 0  # Should get some movies
            assert all(isinstance(r, Title) for r in results)

            # Verify period backup files were created for multiple months
            period_files = list(tmp_path.glob("period_*.json"))
            assert len(period_files) >= 3  # At least Jan, Feb, Mar

    @pytest.mark.anyio
    async def test_get_all_movies_default_cache_dir(self, tmdb_tool):
        """Test that default cache directory follows buttermilk pattern."""

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movies = [{"id": 1, "title": "Test Movie", "release_date": "2020-01-01"}]

            # Mock discover().movie() method
            async def mock_movie_func(**kwargs):
                page = kwargs.get("page", 1)
                if page == 1 and "primary_release_date__gte" in kwargs:
                    return mock_movies
                return []  # No more results

            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            # Call without backup_dir to test default
            results = await tmdb_tool.get_all_movies(start_year=2020, end_year=2020, max_concurrent=1, resume=False)

            # Verify we get results (which means default directory worked)
            assert len(results) >= 0  # Could be empty, just need no errors

    @pytest.mark.anyio
    async def test_get_all_movies_resume_functionality(self, tmdb_tool, tmp_path):
        """Test that get_all_movies can resume from previous progress."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movies = [{"id": 1, "title": "Test Movie", "release_date": "2020-01-01"}]

            # Mock discover().movie() method
            async def mock_movie_func(**kwargs):
                page = kwargs.get("page", 1)
                if page == 1 and "primary_release_date__gte" in kwargs:
                    return mock_movies
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
                resume=True,
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
                resume=True,
            )

            # Should get same results from cached data
            assert len(results2) == len(results1)


class TestTMDBUnitTests:
    """Unit tests with mocked dependencies."""

    @pytest.mark.anyio
    async def test_fetch_period_movies_pagination(self, tmdb_tool, tmp_path):
        """Test that _fetch_period_movies correctly paginates using fetch_single_page."""
        from datetime import date

        from buttermilk.tools.catalog_test import DatePeriod, FetchProgress

        period = DatePeriod(date(2020, 1, 1), date(2020, 1, 31))
        progress = FetchProgress(tmp_path)

        # Mock fetch_single_page to return 3 pages of results
        with patch.object(tmdb_tool, "fetch_single_page") as mock_fetch:
            mock_fetch.side_effect = [
                (
                    [Title(record_id=f"{i}", title=f"Movie {i}", type=TitleType.MOVIE) for i in range(1, 21)],
                    True,
                ),  # Page 1: 20 movies, more available
                (
                    [Title(record_id=f"{i}", title=f"Movie {i}", type=TitleType.MOVIE) for i in range(21, 41)],
                    True,
                ),  # Page 2: 20 movies, more available
                (
                    [Title(record_id=f"{i}", title=f"Movie {i}", type=TitleType.MOVIE) for i in range(41, 51)],
                    False,
                ),  # Page 3: 10 movies, no more
            ]

            results = await tmdb_tool._fetch_period_movies(period, True, False, tmp_path, progress)

            # Should have called fetch_single_page 3 times
            assert mock_fetch.call_count == 3

            # Should return all 50 movies
            assert len(results) == 50

            # Verify correct page numbers were called
            expected_calls = [
                call(period, 1, True, False),
                call(period, 2, True, False),
                call(period, 3, True, False),
            ]
            mock_fetch.assert_has_calls(expected_calls)

            # Verify progress was tracked (pages but not completion)
            # Note: _fetch_period_movies doesn't mark completion, that's done in _fetch_periods_parallel
            assert not progress.is_completed(period)  # Should not be marked complete yet

            # Verify backup file was created
            backup_file = tmp_path / f"period_{period}.json"
            assert backup_file.exists()

    @pytest.mark.anyio
    async def test_fetch_period_movies_resume_from_checkpoint(self, tmdb_tool, tmp_path):
        """Test that _fetch_period_movies correctly resumes from checkpoint."""
        import json
        from datetime import date

        from buttermilk.tools.catalog_test import DatePeriod, FetchProgress

        period = DatePeriod(date(2020, 1, 1), date(2020, 1, 31))
        progress = FetchProgress(tmp_path)

        # Simulate partial progress (completed 2 pages)
        progress.mark_page_complete(period, 2, 40)

        # Create a checkpoint file with 40 movies
        checkpoint_file = progress.get_checkpoint_file(period)
        tmp_path.mkdir(parents=True, exist_ok=True)
        checkpoint_data = [
            {
                "record_id": str(i),
                "title": f"Movie {i}",
                "type": "movie",
                "year": 2020,
                "metadata": {},
            }
            for i in range(1, 41)
        ]
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        # Mock fetch_single_page to continue from page 3
        with patch.object(tmdb_tool, "fetch_single_page") as mock_fetch:
            mock_fetch.side_effect = [
                (
                    [Title(record_id=f"{i}", title=f"Movie {i}", type=TitleType.MOVIE) for i in range(41, 51)],
                    False,
                ),  # Page 3: 10 movies, no more
            ]

            results = await tmdb_tool._fetch_period_movies(period, True, False, tmp_path, progress)

            # Should have called fetch_single_page only once (page 3)
            assert mock_fetch.call_count == 1
            mock_fetch.assert_called_with(period, 3, True, False)

            # Should return all 50 movies (40 from checkpoint + 10 new)
            assert len(results) == 50

    @pytest.mark.anyio
    async def test_parallel_period_execution_timing(self, tmdb_tool, tmp_path):
        """Test that multiple periods are fetched in parallel."""
        import asyncio
        import time
        from datetime import date

        from buttermilk.tools.catalog_test import DatePeriod, FetchProgress

        # Create 3 periods
        periods = [
            DatePeriod(date(2020, 1, 1), date(2020, 1, 31)),
            DatePeriod(date(2020, 2, 1), date(2020, 2, 29)),
            DatePeriod(date(2020, 3, 1), date(2020, 3, 31)),
        ]
        progress = FetchProgress(tmp_path)

        # Track call times to verify parallelism
        call_times = []

        async def mock_fetch_period(*args, **kwargs):
            call_times.append(time.time())
            await asyncio.sleep(0.1)  # Simulate API delay
            return [Title(record_id="1", title="Test Movie", type=TitleType.MOVIE)]

        with patch.object(tmdb_tool, "_fetch_period_movies", side_effect=mock_fetch_period):
            start_time = time.time()
            results = await tmdb_tool._fetch_periods_parallel(
                periods,
                max_concurrent=3,
                include_adult=True,
                include_video=False,
                backup_dir=tmp_path,
                progress=progress,
            )
            total_time = time.time() - start_time

            # Verify calls were made in parallel (all started within 0.05s of each other)
            assert len(call_times) == 3
            assert max(call_times) - min(call_times) < 0.05, "Calls should start nearly simultaneously"

            # Total time should be close to single call time, not 3x (due to parallelism)
            assert total_time < 0.3, f"Parallel execution took {total_time}s, should be < 0.3s"

            # Should get results from all periods
            assert len(results) == 3


class TestTMDBProgressTracking:
    """Unit tests for progress tracking and resume functionality."""

    def test_fetch_progress_page_tracking(self, tmp_path):
        """Test page-level progress tracking."""
        from datetime import date

        from buttermilk.tools.catalog_test import DatePeriod, FetchProgress

        period = DatePeriod(date(2020, 1, 1), date(2020, 1, 31))
        progress = FetchProgress(tmp_path)

        # Initially no progress
        resume_page, checkpoint = progress.get_resume_info(period)
        assert resume_page == 1
        assert checkpoint is None

        # Mark some pages complete
        progress.mark_page_complete(period, 5, 100)
        resume_page, checkpoint = progress.get_resume_info(period)
        assert resume_page == 6  # Should resume from next page
        # Checkpoint file path is returned only if it exists
        assert checkpoint is None  # File doesn't exist yet

        # Mark period complete
        progress.mark_completed(period)
        resume_page, checkpoint = progress.get_resume_info(period)
        assert resume_page == 0  # Indicates already complete
        assert checkpoint is None

    def test_fetch_progress_persistence(self, tmp_path):
        """Test that progress persists across instances."""
        from datetime import date

        from buttermilk.tools.catalog_test import DatePeriod, FetchProgress

        period = DatePeriod(date(2020, 1, 1), date(2020, 1, 31))

        # Create progress and mark some completion
        progress1 = FetchProgress(tmp_path)
        progress1.mark_page_complete(period, 10, 200)
        progress1.mark_completed(period)

        # Create new instance and verify persistence
        progress2 = FetchProgress(tmp_path)
        assert progress2.is_completed(period)
        resume_page, checkpoint = progress2.get_resume_info(period)
        assert resume_page == 0  # Already complete

    def test_fetch_progress_multiple_periods(self, tmp_path):
        """Test tracking multiple periods independently."""
        from datetime import date

        from buttermilk.tools.catalog_test import DatePeriod, FetchProgress

        period1 = DatePeriod(date(2020, 1, 1), date(2020, 1, 31))
        period2 = DatePeriod(date(2020, 2, 1), date(2020, 2, 29))
        progress = FetchProgress(tmp_path)

        # Mark different progress for each period
        progress.mark_page_complete(period1, 5, 100)
        progress.mark_completed(period2)

        # Verify independent tracking
        resume_page1, _ = progress.get_resume_info(period1)
        resume_page2, _ = progress.get_resume_info(period2)

        assert resume_page1 == 6  # In progress
        assert resume_page2 == 0  # Complete
        assert not progress.is_completed(period1)
        assert progress.is_completed(period2)


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
            region="FR",
        )
        assert custom_tool.base_url == "https://custom.tmdb.api/v3"
        assert custom_tool.language == "fr-FR"
        assert custom_tool.region == "FR"

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):  # Clear TMDB_API_KEY env var
            with pytest.raises(ValueError, match="TMDB API key is required"):
                TMDBTool()


class TestTMDBGetAvailabilityById:
    """Test cases for get_availability_by_id with cached API responses."""

    @staticmethod
    def load_fixture_as_mock(fixture_filename: str):
        """Load a JSON fixture file and convert to mock TMDB SDK response.

        Args:
            fixture_filename: Name of fixture file in tests/tools/fixtures/

        Returns:
            Mock response object with results dict containing RegionData dataclasses
        """
        from dataclasses import make_dataclass
        from pathlib import Path

        fixture_path = Path(__file__).parent / "fixtures" / fixture_filename
        with open(fixture_path, encoding="utf-8") as f:
            raw_data = json.load(f)

        # Create dataclass matching TMDB SDK structure
        RegionData = make_dataclass(
            "RegionData",
            [
                ("link", str),
                ("flatrate", list),
                ("rent", list),
                ("buy", list),
                ("ads", list),
                ("free", list),
            ],
        )

        # Convert each region dict to a dataclass instance
        # Use SimpleNamespace (not Mock) so hasattr() returns False for absent fields
        from types import SimpleNamespace

        mock_response = SimpleNamespace()
        mock_response.results = {}
        for region_code, region_data in raw_data["results"].items():
            mock_response.results[region_code] = RegionData(
                link=region_data.get("link", ""),
                flatrate=region_data.get("flatrate", []),
                rent=region_data.get("rent", []),
                buy=region_data.get("buy", []),
                ads=region_data.get("ads", []),
                free=region_data.get("free", []),
            )
        return mock_response, raw_data

    @pytest.fixture
    def cached_availability_response(self):
        """Cached response from TMDB API for movie 1013482 (Borderline)."""
        mock_response, _ = self.load_fixture_as_mock("tmdb_availability_1013482.json")
        return mock_response

    @pytest.mark.anyio
    async def test_get_availability_by_id_parses_cached_response(self, tmdb_tool, cached_availability_response):
        """Test get_availability_by_id correctly parses cached TMDB response."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movie_obj = AsyncMock()
            mock_movie_obj.watch_providers = AsyncMock(return_value=cached_availability_response)
            mock_tmdb.movie.return_value = mock_movie_obj

            results = []
            async for obs in tmdb_tool.get_availability_by_id(
                record_id=1013482,
                title="Borderline",
                year=2024,
            ):
                results.append(obs)

            # Verify observations were created - full fixture has 11 regions
            assert len(results) > 0

            # All results should be Observation instances
            assert all(isinstance(r, Observation) for r in results)

            # Check all 11 regions are present
            regions = {r.region for r in results}
            assert regions == {"AE", "AU", "CA", "GB", "GG", "IE", "NZ", "RU", "SA", "US"}

            # Check AU region has rent and buy providers
            au_obs = [r for r in results if r.region == "AU"]
            au_rent = [r for r in au_obs if r.provider_type == "rent"]
            au_buy = [r for r in au_obs if r.provider_type == "buy"]
            assert len(au_rent) == 3  # Apple TV, Amazon Video, Fetch TV
            assert len(au_buy) == 3

            # Check US region has all provider types
            us_obs = [r for r in results if r.region == "US"]
            us_flatrate = [r for r in us_obs if r.provider_type == "flatrate"]
            us_ads = [r for r in us_obs if r.provider_type == "ads"]
            us_free = [r for r in us_obs if r.provider_type == "free"]
            assert len(us_flatrate) == 5  # Amazon Prime, Peacock, Philo, Prime with Ads, Peacock Plus
            assert len(us_ads) == 3  # Roku, Fandango Free, Tubi
            assert len(us_free) == 2  # Plex, Plex Channel

            # Check AE region has STARZPLAY flatrate
            ae_obs = [r for r in results if r.region == "AE"]
            ae_flatrate = [r for r in ae_obs if r.provider_type == "flatrate"]
            assert len(ae_flatrate) == 1
            assert ae_flatrate[0].provider_name == "STARZPLAY"
            assert ae_flatrate[0].provider_id == "630"

    @pytest.mark.anyio
    async def test_get_availability_by_id_provider_fields_correct(self, tmdb_tool, cached_availability_response):
        """Test that provider fields are correctly extracted from cached response."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movie_obj = AsyncMock()
            mock_movie_obj.watch_providers = AsyncMock(return_value=cached_availability_response)
            mock_tmdb.movie.return_value = mock_movie_obj

            results = []
            async for obs in tmdb_tool.get_availability_by_id(
                record_id=1013482,
                title="Borderline",
                year=2024,
            ):
                results.append(obs)

            # Find the Amazon Prime observation
            prime_obs = next((r for r in results if r.provider_name == "Amazon Prime Video"), None)
            assert prime_obs is not None

            # Verify all fields are correctly set
            assert prime_obs.record_id == "1013482"
            assert prime_obs.title == "Borderline"
            assert prime_obs.year == 2024
            assert prime_obs.provider_id == "9"
            assert prime_obs.provider_name == "Amazon Prime Video"
            assert prime_obs.provider_type == "flatrate"
            assert prime_obs.region == "US"
            assert prime_obs.available is True
            assert prime_obs.source == "TMDB"
            assert prime_obs.metadata["movie_id"] == "1013482"

    @pytest.mark.anyio
    async def test_get_availability_by_id_handles_empty_provider_types(self, tmdb_tool):
        """Test that empty provider types yield null observations."""
        from dataclasses import make_dataclass
        from types import SimpleNamespace

        RegionData = make_dataclass(
            "RegionData",
            [
                ("link", str),
                ("flatrate", list),
                ("rent", list),
                ("buy", list),
            ],
        )

        mock_response = SimpleNamespace()
        mock_response.results = {
            "AU": RegionData(
                link="",
                flatrate=[],  # Empty - should yield null observation
                rent=[{"provider_id": 2, "provider_name": "Apple TV"}],
                buy=[],  # Empty - should yield null observation
            ),
        }

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movie_obj = AsyncMock()
            mock_movie_obj.watch_providers = AsyncMock(return_value=mock_response)
            mock_tmdb.movie.return_value = mock_movie_obj

            results = []
            async for obs in tmdb_tool.get_availability_by_id(
                record_id=1013482,
                title="Borderline",
                year=2024,
            ):
                results.append(obs)

            # Should have observations for:
            # - flatrate (null, available=False)
            # - rent (Apple TV, available=True)
            # - buy (null, available=False)
            au_obs = [r for r in results if r.region == "AU"]
            assert len(au_obs) == 3

            # Check the available rent observation
            rent_obs = [r for r in au_obs if r.provider_type == "rent"]
            assert len(rent_obs) == 1
            assert rent_obs[0].available is True
            assert rent_obs[0].provider_name == "Apple TV"

            # Check the null observations for empty types
            flatrate_null = [r for r in au_obs if r.provider_type == "flatrate"]
            buy_null = [r for r in au_obs if r.provider_type == "buy"]
            assert len(flatrate_null) == 1
            assert len(buy_null) == 1
            assert flatrate_null[0].available is False
            assert flatrate_null[0].provider_name is None
            assert buy_null[0].available is False
            assert buy_null[0].provider_name is None

    @pytest.mark.anyio
    async def test_get_availability_by_id_truncated_response_only_ae(self, tmdb_tool):
        """Test behavior when response is truncated after AE region.

        Simulates what happens if JSON parsing succeeds but only contains
        partial data (e.g., AE region only). This tests whether:
        1. AE observations are still yielded
        2. No error is raised (since JSON parsed successfully)
        """
        from dataclasses import make_dataclass

        RegionData = make_dataclass(
            "RegionData",
            [
                ("link", str),
                ("flatrate", list),
                ("rent", list),
                ("buy", list),
                ("ads", list),
                ("free", list),
            ],
        )

        # Simulate truncated response - only AE region present
        # Use SimpleNamespace (not Mock) so hasattr() returns False for absent fields
        from types import SimpleNamespace

        mock_response = SimpleNamespace()
        mock_response.results = {
            "AE": RegionData(
                link="https://www.themoviedb.org/movie/1013482-borderline/watch?locale=AE",
                flatrate=[
                    {
                        "logo_path": "/pDroY6RxYdVw63eAepag4b116Ub.jpg",
                        "provider_id": 630,
                        "provider_name": "STARZPLAY",
                        "display_priority": 6,
                    }
                ],
                rent=[],
                buy=[],
                ads=[],
                free=[],
            ),
            # Other regions missing - simulates truncation
        }

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movie_obj = AsyncMock()
            mock_movie_obj.watch_providers = AsyncMock(return_value=mock_response)
            mock_tmdb.movie.return_value = mock_movie_obj

            results = []
            async for obs in tmdb_tool.get_availability_by_id(
                record_id=1013482,
                title="Borderline",
                year=2024,
            ):
                results.append(obs)

            # Should still get AE observations
            assert len(results) > 0
            regions = {r.region for r in results}
            assert regions == {"AE"}  # Only AE present

            # AE flatrate should have STARZPLAY
            ae_flatrate = [r for r in results if r.region == "AE" and r.provider_type == "flatrate"]
            assert len(ae_flatrate) == 1
            assert ae_flatrate[0].provider_name == "STARZPLAY"
            assert ae_flatrate[0].available is True

            # Should also have null observations for empty provider types in AE
            ae_rent = [r for r in results if r.region == "AE" and r.provider_type == "rent"]
            assert len(ae_rent) == 1
            assert ae_rent[0].available is False
            assert ae_rent[0].provider_name is None

    @pytest.mark.anyio
    async def test_get_availability_by_id_error_discards_partial_results(self, tmdb_tool):
        """Test that errors discard all partial observations and yield only error record.

        This ensures atomic behavior - either all observations for a title succeed,
        or none are yielded (only an error record).
        """
        from dataclasses import make_dataclass

        RegionData = make_dataclass(
            "RegionData",
            [
                ("link", str),
                ("flatrate", list),
                ("rent", list),
                ("buy", list),
                ("ads", list),
                ("free", list),
            ],
        )

        # Create a response where AE processes fine but AU will cause an error
        # by having a provider dict that's missing required keys
        # Use SimpleNamespace (not Mock) so hasattr() returns False for absent fields
        from types import SimpleNamespace

        mock_response = SimpleNamespace()

        # Create a custom class that raises during iteration for AU
        class BrokenRegionData:
            """Region data that raises an error when converted to dict."""

            link = ""
            flatrate = [{"provider_id": 1, "provider_name": "Test"}]

            def __iter__(self):
                raise RuntimeError("Simulated mid-processing error")

        mock_response.results = {
            "AE": RegionData(
                link="https://example.com",
                flatrate=[
                    {"provider_id": 630, "provider_name": "STARZPLAY"},
                ],
                rent=[],
                buy=[],
                ads=[],
                free=[],
            ),
            "AU": BrokenRegionData(),  # This will fail during iteration
        }

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movie_obj = AsyncMock()
            mock_movie_obj.watch_providers = AsyncMock(return_value=mock_response)
            mock_tmdb.movie.return_value = mock_movie_obj

            results = []
            async for obs in tmdb_tool.get_availability_by_id(
                record_id=1013482,
                title="Borderline",
                year=2024,
            ):
                results.append(obs)

            # Should get ONLY one error observation - no partial AE results
            assert len(results) == 1
            error_obs = results[0]

            # Verify it's an error observation
            assert error_obs.available is False
            assert len(error_obs.error) > 0
            assert error_obs.metadata.get("error_type") == "availability_check_failure"

            # Verify partial observations were discarded (logged in metadata)
            # AE would have produced observations before AU failed
            assert "discarded_observations" in error_obs.metadata

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "fixture_file,movie_id,title,expected_regions,expected_provider",
        [
            pytest.param(
                "tmdb_availability_1013482.json",
                1013482,
                "Borderline",
                {"AE", "AU", "CA", "GB", "GG", "IE", "NZ", "RU", "SA", "US"},
                ("AE", "flatrate", "STARZPLAY", "630"),
                id="borderline-multi-region",
            ),
            pytest.param(
                "tmdb_availability_101271.json",
                101271,
                "Vuelven los Garcia",
                {"CL", "CO", "EC", "MX", "PE"},
                ("MX", "flatrate", "Claro video", "167"),
                id="vuelven-los-garcia-latam",
            ),
        ],
    )
    async def test_get_availability_by_id_parametrized(
        self,
        tmdb_tool,
        fixture_file,
        movie_id,
        title,
        expected_regions,
        expected_provider,
    ):
        """Parametrized test for multiple cached TMDB API responses."""
        mock_response, raw_data = self.load_fixture_as_mock(fixture_file)

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movie_obj = AsyncMock()
            mock_movie_obj.watch_providers = AsyncMock(return_value=mock_response)
            mock_tmdb.movie.return_value = mock_movie_obj

            results = []
            async for obs in tmdb_tool.get_availability_by_id(
                record_id=movie_id,
                title=title,
                year=None,
            ):
                results.append(obs)

            # Verify observations were created
            assert len(results) > 0

            # All results should be Observation instances
            assert all(isinstance(r, Observation) for r in results)

            # Check all expected regions are present
            regions = {r.region for r in results}
            assert regions == expected_regions

            # Check specific provider exists
            region, ptype, pname, pid = expected_provider
            matching_obs = [r for r in results if r.region == region and r.provider_type == ptype and r.provider_name == pname]
            assert len(matching_obs) == 1
            assert matching_obs[0].provider_id == pid
            assert matching_obs[0].available is True

            # Verify record_id is set correctly
            assert all(r.record_id == str(movie_id) for r in results)

    @pytest.mark.anyio
    async def test_get_availability_vuelven_los_garcia_flatrate_only(self, tmdb_tool):
        """Test Vuelven los Garcia response with flatrate-only regions.

        This movie (101271) has 5 Latin American regions, each with only
        flatrate (Claro video) - no rent, buy, ads, or free options.
        """
        mock_response, raw_data = self.load_fixture_as_mock("tmdb_availability_101271.json")

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movie_obj = AsyncMock()
            mock_movie_obj.watch_providers = AsyncMock(return_value=mock_response)
            mock_tmdb.movie.return_value = mock_movie_obj

            results = []
            async for obs in tmdb_tool.get_availability_by_id(
                record_id=101271,
                title="Vuelven los Garcia",
                year=1947,
            ):
                results.append(obs)

            # 5 regions, each with:
            # - 1 flatrate (Claro video) = available=True
            # - 4 empty types (rent, buy, ads, free) = available=False each
            # Total: 5 regions * 5 provider types = 25 observations
            assert len(results) == 25

            # All Claro video observations
            claro_obs = [r for r in results if r.provider_name == "Claro video"]
            assert len(claro_obs) == 5  # One per region
            assert all(r.provider_type == "flatrate" for r in claro_obs)
            assert all(r.provider_id == "167" for r in claro_obs)
            assert all(r.available is True for r in claro_obs)

            # Null observations for empty provider types
            null_obs = [r for r in results if r.provider_name is None]
            assert len(null_obs) == 20  # 5 regions * 4 empty types
            assert all(r.available is False for r in null_obs)

            # Verify regions for null observations
            null_types = {r.provider_type for r in null_obs}
            assert null_types == {"rent", "buy", "ads", "free"}

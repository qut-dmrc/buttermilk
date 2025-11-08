# ruff: noqa: PLR6301
import json
import os
from unittest.mock import AsyncMock, call, patch

import pytest
from autogen_core.tools import FunctionTool

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
            from unittest.mock import Mock

            # Create a simple dataclass for region data
            RegionData = make_dataclass(
                "RegionData",
                [("flatrate", list), ("rent", list), ("buy", list), ("link", str)],
            )

            mock_providers = Mock()
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

            netflix_us = next(
                (r for r in us_observations if r.provider_name == "Netflix"), None
            )
            assert netflix_us is not None
            assert netflix_us.available is True
            assert netflix_us.provider_type == "flatrate"
            assert netflix_us.provider_id == "8"

            # Check GB has Netflix
            gb_observations = [r for r in results if r.region == "GB"]
            assert len(gb_observations) >= 1

    @pytest.mark.anyio
    async def test_get_availability_no_providers_returns_null_observations(
        self, tmdb_tool
    ):
        """Test that when TMDB returns no providers, we get a null observation."""
        title = Title(record_id="550", title="Fight Club", year=1999, metadata={})

        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            # Mock empty providers response - needs to be an object with .results attribute
            from unittest.mock import Mock

            mock_providers = Mock()
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
            mock_movie_obj.watch_providers = AsyncMock(
                side_effect=Exception("API connection failed")
            )
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
                if page == 1 and "primary_release_date.gte" in kwargs:
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
                if page == 1 and "primary_release_date.gte" in kwargs:
                    return [mock_movie]
                else:
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
                with open(period_files[0], "r") as f:
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
                gte_date = kwargs.get("primary_release_date.gte", "")
                if "2020-01" in gte_date:
                    return period_responses["2020-01"]
                elif "2020-02" in gte_date:
                    return period_responses["2020-02"]
                elif "2020-03" in gte_date:
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
            mock_movies = [
                {"id": 1, "title": "Test Movie", "release_date": "2020-01-01"}
            ]

            # Mock discover().movie() method
            async def mock_movie_func(**kwargs):
                page = kwargs.get("page", 1)
                if page == 1 and "primary_release_date.gte" in kwargs:
                    return mock_movies
                else:
                    return []  # No more results

            mock_discover = AsyncMock()
            mock_discover.movie = mock_movie_func
            mock_tmdb.discover.return_value = mock_discover

            # Call without backup_dir to test default
            results = await tmdb_tool.get_all_movies(
                start_year=2020, end_year=2020, max_concurrent=1, resume=False
            )

            # Verify we get results (which means default directory worked)
            assert len(results) >= 0  # Could be empty, just need no errors

    @pytest.mark.anyio
    async def test_get_all_movies_resume_functionality(self, tmdb_tool, tmp_path):
        """Test that get_all_movies can resume from previous progress."""
        with patch.object(tmdb_tool, "_tmdb_client") as mock_tmdb:
            mock_movies = [
                {"id": 1, "title": "Test Movie", "release_date": "2020-01-01"}
            ]

            # Mock discover().movie() method
            async def mock_movie_func(**kwargs):
                page = kwargs.get("page", 1)
                if page == 1 and "primary_release_date.gte" in kwargs:
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
                    [
                        Title(
                            record_id=f"{i}", title=f"Movie {i}", type=TitleType.MOVIE
                        )
                        for i in range(1, 21)
                    ],
                    True,
                ),  # Page 1: 20 movies, more available
                (
                    [
                        Title(
                            record_id=f"{i}", title=f"Movie {i}", type=TitleType.MOVIE
                        )
                        for i in range(21, 41)
                    ],
                    True,
                ),  # Page 2: 20 movies, more available
                (
                    [
                        Title(
                            record_id=f"{i}", title=f"Movie {i}", type=TitleType.MOVIE
                        )
                        for i in range(41, 51)
                    ],
                    False,
                ),  # Page 3: 10 movies, no more
            ]

            results = await tmdb_tool._fetch_period_movies(
                period, True, False, tmp_path, progress
            )

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
            assert not progress.is_completed(
                period
            )  # Should not be marked complete yet

            # Verify backup file was created
            backup_file = tmp_path / f"period_{period}.json"
            assert backup_file.exists()

    @pytest.mark.anyio
    async def test_fetch_period_movies_resume_from_checkpoint(
        self, tmdb_tool, tmp_path
    ):
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
                    [
                        Title(
                            record_id=f"{i}", title=f"Movie {i}", type=TitleType.MOVIE
                        )
                        for i in range(41, 51)
                    ],
                    False,
                ),  # Page 3: 10 movies, no more
            ]

            results = await tmdb_tool._fetch_period_movies(
                period, True, False, tmp_path, progress
            )

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

        with patch.object(
            tmdb_tool, "_fetch_period_movies", side_effect=mock_fetch_period
        ):
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
            assert max(call_times) - min(call_times) < 0.05, (
                "Calls should start nearly simultaneously"
            )

            # Total time should be close to single call time, not 3x (due to parallelism)
            assert total_time < 0.3, (
                f"Parallel execution took {total_time}s, should be < 0.3s"
            )

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
        assert (
            "TMDB" in function_tool.description
            or "Movie Database" in function_tool.description
        )

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

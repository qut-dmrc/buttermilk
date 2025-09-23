import asyncio
import calendar
import datetime
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import shortuuid
from autogen_core.tools import FunctionTool
from pydantic import ConfigDict, Field, field_validator
from themoviedb import aioTMDb
from tqdm.asyncio import tqdm

from buttermilk import get_bm
from buttermilk._core.contract import ErrorEvent
from buttermilk._core.retry import RetryWrapper
from buttermilk._core.storage_config import StorageConfig
from buttermilk._core.types import BaseRecord
from buttermilk.utils.uploader import AsyncDataUploader
from buttermilk.utils.utils import scrub_serializable
from buttermilk.utils.validators import make_list_validator  # Pydantic validators


# Data Models
class TitleType(str, Enum):
    """Type of media title."""

    MOVIE = "movie"
    TV = "tv"


class Title(BaseRecord):
    """Represents a movie title with metadata from TMDB search."""

    record_id: str = Field(..., description="TMDB movie ID")
    title: str = Field(..., description="Movie title")
    year: int | None = Field(None, description="Release year if available")
    type: TitleType = Field(default=TitleType.MOVIE, description="Type of title (movie or tv)")
    metadata: dict = Field(default_factory=dict, description="Additional movie metadata from TMDB")

    # BaseRecord required fields
    dataset_name: str = Field(default="tmdb", description="Dataset this title belongs to")
    split_type: str = Field(default="default", description="Dataset split (e.g., train, test)")
    error: list[Any] = Field(default_factory=list, description="List of ErrorEvent objects")

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        use_enum_values=True,
        validate_assignment=True,
    )


class Observation(Title):
    """Represents availability information for a movie in a specific region."""

    record_id: str = Field(..., description="Record under test")
    call_id: str = Field(
        default_factory=lambda: str(shortuuid.uuid()),  # Use shortuuid for brevity
        description="A unique ID for this specific agent execution/response. Attempts to use Weave call ID if available.",
    )
    test_date: datetime.datetime | None = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="Timestamp when the observation was made",
    )

    provider_id: str | None = Field(default=None, description="Unique ID of the streaming/availability provider")
    provider_name: str | None = Field(default=None, description="Name of the streaming/availability provider")
    provider_type: str | None = Field(default=None, description="Type of availability (flatrate, rent, buy)")
    region: str = Field(..., description="Geographical region of the observation (e.g., US, UK)")
    price: float | None = Field(None, description="Price for renting or buying, if applicable")
    currency: str | None = Field(None, description="Currency of the price, if applicable")
    format: str | None = Field(None, description="Format of the content (e.g., HD, SD, 4K)")
    available: bool = Field(..., description="Whether the title is available")
    source: str = Field(..., description="Source of the observation data")

    # error field is inherited from Title (which inherits from BaseRecord)
    _ensure_error_list: classmethod = field_validator("error", mode="before")(make_list_validator())  # type: ignore

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        use_enum_values=True,
        validate_assignment=True,
    )


@dataclass
class DatePeriod:
    """Represents a date period for TMDB movie fetching."""

    start_date: datetime.date
    end_date: datetime.date

    @property
    def api_params(self) -> dict[str, str]:
        """Get API parameters for this date period."""
        return {
            "primary_release_date__gte": self.start_date.isoformat(),
            "primary_release_date__lte": self.end_date.isoformat(),
        }

    def __str__(self) -> str:
        """String representation as YYYY-MM format."""
        return f"{self.start_date.year}-{self.start_date.month:02d}"

    def __hash__(self) -> int:
        """Make hashable for use in sets."""
        return hash((self.start_date, self.end_date))


class FetchProgress:
    """Manages progress tracking and resume functionality for movie fetching."""

    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        self.progress_file = backup_dir / "fetch_progress.json"
        self.completed_periods: set[str] = set()
        self.in_progress_periods: dict[str, dict] = {}
        self._load_progress()

    def _load_progress(self):
        """Load progress from progress file."""
        if not self.progress_file.exists():
            return

        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.completed_periods = set(data.get("completed_periods", []))
                self.in_progress_periods = data.get("in_progress_periods", {})
        except (json.JSONDecodeError, OSError):
            self.completed_periods = set()
            self.in_progress_periods = {}

    def _save_progress(self):
        """Save current progress to file."""
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Save progress
        progress_data = {
            "completed_periods": list(self.completed_periods),
            "in_progress_periods": self.in_progress_periods,
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=2)

    def mark_completed(self, period: DatePeriod):
        """Mark period as completed and save progress."""
        period_str = str(period)
        self.completed_periods.add(period_str)

        # Remove from in-progress if it was there
        if period_str in self.in_progress_periods:
            del self.in_progress_periods[period_str]

        self._save_progress()

    def mark_page_complete(self, period: DatePeriod, page: int, movies_count: int):
        """Track completion of a specific page within a period."""
        period_str = str(period)
        if period_str not in self.in_progress_periods:
            self.in_progress_periods[period_str] = {
                "last_completed_page": 0,
                "total_movies": 0,
                "checkpoint_file": f"checkpoint_{period}.json",
                "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

        self.in_progress_periods[period_str].update(
            {"last_completed_page": page, "total_movies": movies_count, "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        )
        self._save_progress()

    def get_resume_info(self, period: DatePeriod) -> tuple[int, Path | None]:
        """Get resume page and checkpoint file for a period.

        Returns:
            Tuple of (next_page_to_fetch, checkpoint_file_path)
            If period is complete, returns (0, None)
            If starting fresh, returns (1, None)
        """
        period_str = str(period)

        # Check if already complete
        if period_str in self.completed_periods:
            return (0, None)  # Already complete

        # Check if in progress
        if period_str in self.in_progress_periods:
            info = self.in_progress_periods[period_str]
            checkpoint = self.backup_dir / info["checkpoint_file"]
            next_page = info["last_completed_page"] + 1
            return (next_page, checkpoint if checkpoint.exists() else None)

        # Starting fresh
        return (1, None)

    def is_completed(self, period: DatePeriod) -> bool:
        """Check if period was already fetched."""
        return str(period) in self.completed_periods

    def get_period_backup_file(self, period: DatePeriod) -> Path:
        """Get backup file path for a specific period."""
        return self.backup_dir / f"period_{period}.json"

    def get_checkpoint_file(self, period: DatePeriod) -> Path:
        """Get checkpoint file path for a specific period."""
        return self.backup_dir / f"checkpoint_{period}.json"


def _generate_monthly_periods(start_year: int, end_year: int) -> list[DatePeriod]:
    """Generate monthly date periods using proper month-end dates.

    Args:
        start_year: First year to include
        end_year: Last year to include (inclusive)

    Returns:
        List of DatePeriod objects, one for each month
    """
    periods = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Get the last day of the month
            _, last_day = calendar.monthrange(year, month)

            start_date = datetime.date(year, month, 1)
            end_date = datetime.date(year, month, last_day)

            periods.append(DatePeriod(start_date, end_date))

    return periods


class TMDBTool:
    """Tool for searching movie availability using TMDB API.

    This tool searches for movies and returns availability information for the specified region.
    Returns null observations when no results are found.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.themoviedb.org/3",
        language: str = "en-US",
        region: str = "AU",
        observations_storage_config: str | StorageConfig | None = None,
        titles_storage_config: str | StorageConfig | None = None,
        batch_size: int = 10,
    ):
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        self.base_url = base_url
        self.language = language
        self.region = (region or "").upper() or "AU"

        if not self.api_key:
            raise ValueError("TMDB API key is required. Please set TMDB_API_KEY environment variable or pass api_key parameter.")

        # Underlying client + retry wrapper
        self._tmdb_client = aioTMDb(key=self.api_key, language=language, region=self.region)
        self._retry = RetryWrapper(client=self._tmdb_client)

        # Initialize storage uploaders if configs provided
        self.observations_uploader = None
        self.titles_uploader = None

        if observations_storage_config:
            self._setup_observations_storage(observations_storage_config, batch_size)

        if titles_storage_config:
            self._setup_titles_storage(titles_storage_config, batch_size)

    def _setup_observations_storage(self, config: StorageConfig, batch_size: int):
        """Set up storage for observations data."""
        bm = get_bm()

        storage = bm.get_storage(config)
        self.observations_uploader = AsyncDataUploader(storage=storage, buffer_size=batch_size)

    def _setup_titles_storage(self, config: StorageConfig, batch_size: int):
        """Set up storage for titles data."""
        bm = get_bm()
        storage = bm.get_storage(config)
        self.titles_uploader = AsyncDataUploader(storage=storage, buffer_size=batch_size)

    async def cleanup(self):
        """Gracefully shutdown uploaders and ensure all data is flushed."""
        if self.observations_uploader:
            self.observations_uploader.shutdown()
        if self.titles_uploader:
            self.titles_uploader.shutdown()

    # -------------------------
    # Pipeline method: provides TMDBTool.get_availability() as a pipeline processor
    # that accepts Title records and adds availability data to metadata.
    # -------------------------
    async def process(self, record: Title) -> BaseRecord | None:
        """Process a Title record to add TMDB availability data.

        Works with Title records (which are BaseRecord subclasses).
        Adds availability observations to record.metadata["tmdb_availability"].

        Args:
            record: Title record to process

        Returns:
            Same record with enhanced metadata, or None if processing failed
        """
        try:
            # Get availability data for all regions
            observations = await self.get_availability(record, regions=None)

            # Extract unique regions from observations
            regions_found = list(set(obs.region for obs in observations if hasattr(obs, "region")))

            # Enhance metadata with availability data
            record.metadata["tmdb_availability"] = {
                "regions_found": regions_found,
                "observations_count": len(observations),
                "observations": [obs.model_dump() for obs in observations],
                "processed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }

            # Store observations if uploader configured
            if self.observations_uploader:
                for obs in observations:
                    await self.observations_uploader.add(obs)

            return record

        except Exception as e:
            from buttermilk._core.contract import ErrorEvent

            # Add error to record's error list
            if not hasattr(record, 'error'):
                record.error = []
            record.error.append(ErrorEvent(
                content=f"TMDBTool availability check failed: {e}",
                source="TMDBTool"
            ))

            # Also track in metadata
            record.metadata["tmdb_availability"] = {
                "status": "failed",
                "error": str(e),
                "processed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }

            # Return record even on error (let pipeline decide whether to filter)
            return record

    # -------------------------
    # Internal helpers
    # -------------------------
    @staticmethod
    def _normalize_region(region: Optional[str]) -> str:
        r = (region or "").strip().upper()
        return r if r else "US"

    @staticmethod
    def _as_list(results: Any) -> list[Any]:  # noqa: PLR0911 - explicit early returns aid clarity
        """Coerce various SDK return shapes into a plain list of items.

        Handles:
        - list/tuple already
        - dict with key "results"
        - object with attribute "results"
        - generic iterable (excluding str/bytes)
        """
        if results is None:
            return []
        if isinstance(results, (list, tuple)):
            return list(results)
        if isinstance(results, dict):
            if "results" in results and isinstance(results["results"], (list, tuple)):
                return list(results["results"])  # type: ignore[return-value]
            # Some SDKs return pagination dicts where items are directly in 'results'
            return []
        # Object with .results attribute
        items = getattr(results, "results", None)
        if isinstance(items, (list, tuple)):
            return list(items)
        # Fallback: if it's iterable (but not string/bytes), iterate
        if isinstance(results, Iterable) and not isinstance(results, (str, bytes)):
            try:
                return list(results)
            except Exception:
                return []
        return []

    @staticmethod
    def _get_value(obj: Any, *keys: str, default: Any = None) -> Any:
        for k in keys:
            if isinstance(obj, dict) and k in obj:
                return obj[k]
            v = getattr(obj, k, None)
            if v is not None:
                return v
        return default

    @staticmethod
    def _extract_title(item: Any) -> str | None:
        # Prefer localized/common title fields
        return TMDBTool._get_value(
            item,
            "title",
            "name",
            "original_title",
            "original_name",
            default=None,
        )

    @staticmethod
    def _sanitize_rate_limit_message(msg: str) -> str:
        lower = msg.lower()
        if "rate limit" in lower or "too many requests" in lower or "429" in lower:
            return "Temporary TMDB throttling encountered; retry budget exceeded"
        return msg

    async def search_movie(self, title: str, year: Optional[int] = None) -> Optional[Title]:
        """Search for a movie and return Title object with metadata.

        Args:
            title: Movie title to search for
            year: Optional release year to filter by

        Returns:
            Title object with movie metadata, or None if not found
        """

        # Search for movies using TMDB API
        async def do_search():
            result = await self._tmdb_client.search().movies(query=title, year=year)
            return result

        search_raw = await self._retry._execute_with_retry(do_search)
        search_results = self._as_list(search_raw)

        if not search_results:
            return None

        # Get the first/best match
        movie = search_results[0]
        movie_id = str(self._get_value(movie, "id"))
        movie_title = self._extract_title(movie)
        if not movie_title:
            raise ValueError("TMDB search result missing title")

        # Extract year from release_date if not provided
        release_date = self._get_value(movie, "release_date", "first_air_date")
        if year is None and release_date:
            try:
                # release_date can be a string like "YYYY-MM-DD" or a datetime/date object
                if isinstance(release_date, (datetime.date, datetime.datetime)):
                    year = release_date.year
                else:
                    year = int(str(release_date).split("-")[0])
            except (ValueError, IndexError, TypeError):
                pass

        # Build metadata from the movie result
        metadata = {
            "original_title": self._get_value(movie, "original_title", "original_name"),
            "overview": self._get_value(movie, "overview"),
            "release_date": release_date,
            "popularity": self._get_value(movie, "popularity"),
            "vote_average": self._get_value(movie, "vote_average"),
            "vote_count": self._get_value(movie, "vote_count"),
            "poster_path": self._get_value(movie, "poster_path"),
            "backdrop_path": self._get_value(movie, "backdrop_path"),
            "genre_ids": self._get_value(movie, "genre_ids"),
        }

        # Remove None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return Title(record_id=movie_id, title=movie_title, year=year, metadata=metadata)

    async def get_availability(self, title: Title, regions: Optional[list[str]] = None) -> list[Observation]:
        """Get availability observations for a movie in specified regions.

        Public wrapper that accepts a Title and delegates to the ID-based method.

        Args:
            title: Title object from search_movie
            regions: List of region codes to check (e.g., ["US", "GB", "AU"])

        Returns:
            List of Observations with availability data, including null observations for regions without providers
        """
        return await self.get_availability_by_id(
            record_id=int(title.record_id),
            regions=regions,
            title=title.title,
            year=title.year,
        )

    async def get_availability_by_id(
        self,
        record_id: int,
        regions: Optional[list[str]] = None,
        *,
        title: Optional[str] = None,
        year: Optional[int] = None,
    ) -> list[Observation]:
        """Get availability observations for a TMDB record_id in specified regions.

        Args:
            record_id: TMDB numeric movie ID
            regions: List of region codes to check (e.g., ["US", "GB", "AU"])
            title: Optional title string for metadata enrichment
            year: Optional year for metadata enrichment

        Returns:
            List of Observations with availability data, including null observations for regions without providers
        """
        if regions is None:
            regions = ["US"]

        observations: list[Observation] = []

        try:
            # Get watch providers for all regions
            async def do_get_providers():
                return await self._tmdb_client.movie(int(record_id)).watch_providers()

            providers_raw = await self._retry._execute_with_retry(do_get_providers)

            # Extract results by region
            providers_by_region: dict[str, Any] = {}
            if isinstance(providers_raw, dict) and "results" in providers_raw:
                providers_by_region = providers_raw["results"]

            # Process each requested region
            for r in regions:
                normalized_region = self._normalize_region(r)

                if normalized_region in providers_by_region:
                    region_data = providers_by_region[normalized_region]

                    # Process each provider type (flatrate, rent, buy)
                    for provider_type in ["flatrate", "rent", "buy"]:
                        if provider_type in region_data:
                            providers = region_data[provider_type]
                            if not isinstance(providers, list):
                                continue

                            for provider in providers:
                                obs = Observation(
                                    record_id=f"tmdb_{record_id}_{normalized_region}_{provider.get('provider_id')}",
                                    provider_id=str(provider.get("provider_id")) if provider.get("provider_id") is not None else None,
                                    provider_name=provider.get("provider_name"),
                                    provider_type=provider_type,
                                    region=normalized_region,
                                    price=None,
                                    currency=None,
                                    format=None,
                                    available=True,
                                    source="TMDB",
                                    metadata={
                                        "title": title,
                                        "year": year,
                                        "movie_id": str(record_id),
                                    },
                                )
                                observations.append(obs)

                # If no providers found for this region, add a null observation
                if not any(o.region == normalized_region for o in observations):
                    obs = Observation(
                        record_id=f"tmdb_{record_id}_{normalized_region}_null",
                        provider_name=None,
                        provider_id=None,
                        provider_type=None,
                        region=normalized_region,
                        price=None,
                        currency=None,
                        format=None,
                        available=False,
                        source="TMDB",
                        metadata={
                            "title": title,
                            "year": year,
                            "movie_id": str(record_id),
                        },
                    )
                    observations.append(obs)

        except Exception as e:
            # Return error observations for all regions
            safe_message = self._sanitize_rate_limit_message(str(e))
            error_event = ErrorEvent(content=safe_message, source="TMDB")

            for r in regions:
                normalized_region = self._normalize_region(r)
                error_obs = Observation(
                    record_id=f"tmdb_{record_id}_{normalized_region}_error",
                    provider_name=None,
                    provider_id=None,
                    provider_type=None,
                    region=normalized_region,
                    price=None,
                    currency=None,
                    format=None,
                    available=False,
                    source="TMDB",
                    metadata={
                        "title": title,
                        "year": year,
                        "movie_id": str(record_id),
                        "error_type": "availability_check_failure",
                    },
                    error=[error_event],
                )
                observations.append(error_obs)

        # Batch save observations if uploader is configured
        if self.observations_uploader and observations:
            for obs in observations:
                await self.observations_uploader.add(obs)

        return observations

    async def fetch_single_page(
        self,
        period: DatePeriod,
        page: int,
        include_adult: bool = True,
        include_video: bool = False
    ) -> tuple[list[Title], bool]:
        """Fetch a single page of movies for a specific period.

        This is the atomic unit for TMDB API operations. Returns exactly one page
        of results from the discover endpoint.

        Args:
            period: Date period to fetch movies for
            page: Page number to fetch (1-based)
            include_adult: Include adult content
            include_video: Include video content

        Returns:
            Tuple of (list of Title objects, has_more_pages boolean)
        """
        # Prepare API parameters
        api_params = {
            **period.api_params,
            "page": page,
            "include_adult": include_adult,
            "include_video": include_video,
            "sort_by": "primary_release_date.asc"
        }

        # Fetch movies for this page
        async def do_discover():
            return await self._tmdb_client.discover().movie(**api_params)

        movies_raw = await self._retry._execute_with_retry(do_discover)
        movies = self._as_list(movies_raw)

        # Convert raw movie data to Title objects
        title_objects = []
        for movie in movies:
            movie_id = str(self._get_value(movie, "id"))
            movie_title = self._get_value(movie, "title", "name", default="Unknown")
            release_date = self._get_value(movie, "release_date")

            # Extract year from release date
            year = None
            if release_date:
                try:
                    year = int(str(release_date)[:4])
                except (ValueError, IndexError, TypeError):
                    pass

            # Build metadata from all available fields except core ones
            metadata = {}
            core_fields = {"id", "title", "name", "release_date"}

            # Convert movie object to dictionary for metadata extraction
            if isinstance(movie, dict):
                movie_dict = movie
            else:
                movie_dict = getattr(movie, "__dict__", {"id": movie_id})

            for key, value in movie_dict.items():
                if key not in core_fields and value is not None:
                    metadata[key] = value

            # Create Title record
            record = Title(
                record_id=movie_id,
                title=movie_title,
                year=year,
                type=TitleType.MOVIE,
                metadata=metadata
            )
            title_objects.append(record)

            # Batch save titles if uploader is configured
            if self.titles_uploader:
                await self.titles_uploader.add(record)

        # Determine if there are more pages
        # If we got fewer movies than expected, or no movies, we're likely at the end
        has_more_pages = len(movies) > 0 and page < 500  # TMDB limit

        return title_objects, has_more_pages

    async def _fetch_period_movies(
        self,
        period: DatePeriod,
        include_adult: bool,
        include_video: bool,
        backup_dir: Path,
        progress: FetchProgress | None = None
    ) -> list[Title]:
        """Fetch all movies for a single date period using fetch_single_page.

        This method handles:
        - Resume from partial completion using progress tracking
        - Checkpoint saving every 10 pages
        - Final backup to period file when complete

        Args:
            period: Date period to fetch movies for
            include_adult: Include adult content
            include_video: Include video content
            backup_dir: Directory for backup files
            progress: Progress tracker (optional)

        Returns:
            List of Title objects for the period
        """
        period_movies = []

        # Get resume information
        start_page = 1
        checkpoint_file = None
        if progress:
            start_page, checkpoint_file = progress.get_resume_info(period)
            if start_page == 0:
                # Period already completed, load from backup
                backup_file = progress.get_period_backup_file(period)
                if backup_file.exists():
                    try:
                        with open(backup_file, "r", encoding="utf-8") as f:
                            movie_data = json.load(f)
                            return [Title(**movie) for movie in movie_data]
                    except (json.JSONDecodeError, OSError):
                        pass
                return []

        # Load any partial results from checkpoint
        if checkpoint_file and checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                    period_movies = [Title(**movie) for movie in checkpoint_data]
            except (json.JSONDecodeError, OSError):
                pass

        # Create progress tracking for this period
        period_pbar = tqdm(
            desc=f"Fetching {period}",
            unit="page",
            leave=False,
            initial=start_page - 1
        )

        try:
            page = start_page
            while page <= 500:  # TMDB limit per query
                # Fetch single page
                page_movies, has_more = await self.fetch_single_page(
                    period, page, include_adult, include_video
                )

                # Add to our collection
                period_movies.extend(page_movies)

                # Update progress
                period_pbar.update(1)
                period_pbar.set_postfix({
                    "movies": len(period_movies),
                    "page": f"{page}/500"
                })

                # Update progress tracking
                if progress:
                    progress.mark_page_complete(period, page, len(period_movies))

                # Save checkpoint every 10 pages
                if page % 10 == 0 or not has_more:
                    checkpoint_file = progress.get_checkpoint_file(period) if progress else backup_dir / f"checkpoint_{period}.json"
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(
                            [scrub_serializable(movie.model_dump()) for movie in period_movies],
                            f,
                            indent=2,
                            default=str
                        )

                # If no more pages, we're done
                if not has_more:
                    break

                page += 1

                # Small delay to respect rate limits
                await asyncio.sleep(0.1)

            # Save final period backup
            backup_file = backup_dir / f"period_{period}.json"
            backup_dir.mkdir(parents=True, exist_ok=True)

            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(
                    [scrub_serializable(movie.model_dump()) for movie in period_movies],
                    f,
                    indent=2,
                    default=str
                )

            # Clean up checkpoint file if we completed successfully
            if progress:
                checkpoint_file = progress.get_checkpoint_file(period)
                if checkpoint_file.exists():
                    checkpoint_file.unlink()

        finally:
            period_pbar.close()

        return period_movies

    async def _fetch_periods_parallel(
        self,
        periods: list[DatePeriod],
        max_concurrent: int,
        include_adult: bool,
        include_video: bool,
        backup_dir: Path,
        progress: FetchProgress
    ) -> list[Title]:
        """Fetch multiple periods concurrently using asyncio.Semaphore.

        Args:
            periods: List of periods to fetch
            max_concurrent: Maximum concurrent period fetches
            include_adult: Include adult content
            include_video: Include video content
            backup_dir: Directory for backup files
            progress: Progress tracker for marking completed periods

        Returns:
            List of all Title objects from all periods
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        all_movies = []

        # Create overall progress bar
        overall_pbar = tqdm(total=len(periods), desc="Processing periods", unit="period")

        async def fetch_single_period(period: DatePeriod) -> list[Title]:
            """Fetch a single period with semaphore control."""
            async with semaphore:
                try:
                    # Check if already completed
                    if progress.is_completed(period):
                        overall_pbar.set_postfix({"status": f"Skipping {period} (completed)"})
                        overall_pbar.update(1)
                        # Load from backup file
                        backup_file = progress.get_period_backup_file(period)
                        if backup_file.exists():
                            try:
                                with open(backup_file, "r", encoding="utf-8") as f:
                                    movie_data = json.load(f)
                                    return [Title(**movie) for movie in movie_data]
                            except (json.JSONDecodeError, OSError):
                                pass
                        return []

                    # Fetch the period
                    overall_pbar.set_postfix({"status": f"Fetching {period}"})
                    period_movies = await self._fetch_period_movies(
                        period, include_adult, include_video, backup_dir, progress
                    )

                    # Mark as completed
                    progress.mark_completed(period)
                    overall_pbar.set_postfix({
                        "status": f"Completed {period}",
                        "movies": len(period_movies)
                    })
                    overall_pbar.update(1)

                    return period_movies

                except Exception as e:
                    overall_pbar.set_postfix({"status": f"Error in {period}: {e}"})
                    overall_pbar.update(1)
                    print(f"Error fetching period {period}: {e}")
                    return []

        # Create tasks for all periods
        tasks = [fetch_single_period(period) for period in periods]

        # Execute all tasks and collect results
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    print(f"Period fetch failed: {result}")
                    continue
                if isinstance(result, list):
                    all_movies.extend(result)

        finally:
            overall_pbar.close()

        return all_movies

    async def get_all_movies(
        self,
        start_year: int = 1900,
        end_year: int = 2025,
        max_concurrent: int = 10,
        backup_dir: Optional[Path] = None,
        include_adult: bool = True,
        include_video: bool = False,
        resume: bool = True,
    ) -> list[Title]:
        """Get all movies from TMDB using parallel month-based queries.

        This method fetches movies by breaking the search into monthly periods
        and processing multiple periods concurrently to avoid TMDB's 500-page limit.

        Args:
            start_year: First year to include (inclusive)
            end_year: Last year to include (inclusive)
            max_concurrent: Maximum number of concurrent period fetches
            backup_dir: Directory to save progress and backup files
            include_adult: Include adult content
            include_video: Include video content
            resume: Whether to resume from previous progress

        Returns:
            List of Title objects representing discovered movies
        """
        # Use default cache directory if not specified
        if backup_dir is None:
            backup_dir = Path.home() / ".cache" / "buttermilk" / "tmdb"
        else:
            backup_dir = Path(backup_dir)

        # Ensure backup directory exists
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Generate monthly periods
        all_periods = _generate_monthly_periods(start_year, end_year)

        # Set up progress tracking
        progress = FetchProgress(backup_dir) if resume else None

        # Filter out completed periods if resuming
        if resume and progress:
            remaining_periods = [p for p in all_periods if not progress.is_completed(p)]
            print(f"Total periods: {len(all_periods)}, Remaining: {len(remaining_periods)}")
        else:
            remaining_periods = all_periods
            progress = FetchProgress(backup_dir)

        try:
            # Fetch periods in parallel
            all_movies = await self._fetch_periods_parallel(
                remaining_periods,
                max_concurrent,
                include_adult,
                include_video,
                backup_dir,
                progress
            )

            # Load previously completed movies if resuming
            if resume and progress:
                completed_periods = [p for p in all_periods if progress.is_completed(p) and p not in remaining_periods]
                for period in completed_periods:
                    backup_file = progress.get_period_backup_file(period)
                    if backup_file.exists():
                        try:
                            with open(backup_file, "r", encoding="utf-8") as f:
                                movie_data = json.load(f)
                                completed_movies = [Title(**movie) for movie in movie_data]
                                all_movies.extend(completed_movies)
                        except (json.JSONDecodeError, OSError):
                            pass

            print(f"Total movies collected: {len(all_movies)}")
            return all_movies

        finally:
            # Ensure uploaders are flushed
            if self.titles_uploader:
                self.titles_uploader.shutdown()

    def as_tool(self) -> FunctionTool:
        """Return as autogen FunctionTool for agent integration."""
        return FunctionTool(
            name="tmdb_search",
            description=(
                "Search for movie availability information using The Movie Database (TMDB). "
                "Provide movie title and optional year to find movie metadata."
            ),
            func=self.search_movie,
            strict=False,  # Allow default parameters for better UX
        )

import datetime
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import shortuuid
from autogen_core.tools import FunctionTool
from pydantic import BaseModel, ConfigDict, Field, field_validator
from themoviedb import aioTMDb
from tqdm.asyncio import tqdm

from buttermilk import get_bm
from buttermilk._core.contract import ErrorEvent
from buttermilk._core.retry import RetryWrapper
from buttermilk._core.storage_config import StorageConfig, StorageFactory
from buttermilk.utils.uploader import AsyncDataUploader
from buttermilk.utils.utils import scrub_serializable
from buttermilk.utils.validators import make_list_validator  # Pydantic validators


# Data Models
class TitleType(str, Enum):
    """Type of media title."""

    MOVIE = "movie"
    TV = "tv"


class Observation(BaseModel):
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
    metadata: dict = Field(..., description="Metadata about the title availability")

    error: list[ErrorEvent] = Field(
        default_factory=list,
        description="List of error messages accumulated during processing related to this message.",
    )
    _ensure_error_list: classmethod = field_validator("error", mode="before")(make_list_validator())  # type: ignore

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        use_enum_values=True,
        validate_assignment=True,
    )


class Title(BaseModel):
    """Represents a movie title with metadata from TMDB search."""

    record_id: str = Field(..., description="TMDB movie ID")
    title: str = Field(..., description="Movie title")
    year: int | None = Field(None, description="Release year if available")
    type: TitleType = Field(default=TitleType.MOVIE, description="Type of title (movie or tv)")
    metadata: dict = Field(default_factory=dict, description="Additional movie metadata from TMDB")

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        use_enum_values=True,
        validate_assignment=True,
    )


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
        batch_size: int = 10
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

    def _setup_observations_storage(self, config: str | StorageConfig, batch_size: int):
        """Set up storage for observations data."""
        bm = get_bm()
        if isinstance(config, str):
            # Load config by name from Buttermilk's config system
            storage_config = bm.cfg.storage[config]
        else:
            storage_config = config

        storage_config = StorageFactory.create_config(storage_config)
        storage = bm.get_storage(storage_config)
        self.observations_uploader = AsyncDataUploader(storage=storage, buffer_size=batch_size)

    def _setup_titles_storage(self, config: str | StorageConfig, batch_size: int):
        """Set up storage for titles data."""
        bm = get_bm()
        if isinstance(config, str):
            # Load config by name from Buttermilk's config system
            storage_config = bm.cfg.storage[config]
        else:
            storage_config = config

        storage_config = StorageFactory.create_config(storage_config)
        storage = bm.get_storage(storage_config)
        self.titles_uploader = AsyncDataUploader(storage=storage, buffer_size=batch_size)

    async def cleanup(self):
        """Gracefully shutdown uploaders and ensure all data is flushed."""
        if self.observations_uploader:
            self.observations_uploader.shutdown()
        if self.titles_uploader:
            self.titles_uploader.shutdown()

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
        try:
            # Search for movies using TMDB API
            async def do_search():
                return await self._tmdb_client.search().movies(query=title, year=year)

            search_raw = await self._retry._execute_with_retry(do_search)
            search_results = self._as_list(search_raw)

            if not search_results:
                return None

            # Get the first/best match
            movie = search_results[0]
            movie_id = str(self._get_value(movie, "id"))
            movie_title = self._extract_title(movie)

            # Extract year from release_date if not provided
            release_date = self._get_value(movie, "release_date", "first_air_date")
            if year is None and release_date:
                try:
                    year = int(release_date.split("-")[0])
                except (ValueError, IndexError):
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

        except Exception:
            # Return None on search failure
            return None

    async def get_availability(self, title: Title, regions: Optional[list[str]] = None) -> list[Observation]:
        """Get availability observations for a movie in specified regions.

        Args:
            title: Title object from search_movie
            regions: List of region codes to check (e.g., ["US", "GB", "AU"])

        Returns:
            List of Observations with availability data, including null observations for regions without providers
        """
        if regions is None:
            regions = ["US"]

        observations = []

        try:
            # Get watch providers for all regions
            async def do_get_providers():
                return await self._tmdb_client.movie(int(title.record_id)).watch_providers()

            providers_raw = await self._retry._execute_with_retry(do_get_providers)

            # Extract results by region
            providers_by_region = {}
            if isinstance(providers_raw, dict) and "results" in providers_raw:
                providers_by_region = providers_raw["results"]

            # Process each requested region
            for region in regions:
                region = self._normalize_region(region)

                if region in providers_by_region:
                    region_data = providers_by_region[region]

                    # Process each provider type (flatrate, rent, buy)
                    for provider_type in ["flatrate", "rent", "buy"]:
                        if provider_type in region_data:
                            providers = region_data[provider_type]
                            if not isinstance(providers, list):
                                continue

                            for provider in providers:
                                obs = Observation(
                                    record_id=f"tmdb_{title.record_id}_{region}_{provider.get('provider_id')}",
                                    provider_id=str(provider.get("provider_id")),
                                    provider_name=provider.get("provider_name"),
                                    provider_type=provider_type,
                                    region=region,
                                    available=True,
                                    source="TMDB",
                                    metadata={
                                        "title": title.title,
                                        "year": title.year,
                                        "movie_id": title.record_id,
                                    },
                                )
                                observations.append(obs)

                # If no providers found for this region, add a null observation
                if not any(obs.region == region for obs in observations):
                    obs = Observation(
                        record_id=f"tmdb_{title.record_id}_{region}_null",
                        provider_name=None,
                        provider_id=None,
                        provider_type=None,
                        region=region,
                        available=False,
                        source="TMDB",
                        metadata={
                            "title": title.title,
                            "year": title.year,
                            "movie_id": title.record_id,
                        },
                    )
                    observations.append(obs)

        except Exception as e:
            # Return error observations for all regions
            safe_message = self._sanitize_rate_limit_message(str(e))
            error_event = ErrorEvent(content=safe_message, source="TMDB")

            for region in regions:
                region = self._normalize_region(region)
                error_obs = Observation(
                    record_id=f"tmdb_{title.record_id}_{region}_error",
                    provider_name=None,
                    provider_id=None,
                    provider_type=None,
                    region=region,
                    available=False,
                    source="TMDB",
                    metadata={
                        "title": title.title,
                        "year": title.year,
                        "movie_id": title.record_id,
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

    async def get_all_movies(
        self,
        backup_dir: Optional[Path] = None,
        max_results: Optional[int] = None,
        include_adult: bool = True,
        include_video: bool = False,
        sort_by: str = "primary_release_date.asc",
        page: int = 1,
        **kwargs,
    ) -> list[Title]:
        """Get all movies from TMDB discover endpoint with pagination support.

        Args:
            backup_dir: Optional directory to save JSON backups of each movie.
                       Defaults to ~/.cache/buttermilk/tmdb
            max_results: Optional limit on number of results to return
            include_adult: Include adult content
            include_video: Include video content
            sort_by: Sort order for results
            **kwargs: Additional parameters for discover.movie()

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

        all_movies = []
        total_fetched = 0
        max_page = page + 500  # Enough for one batch

        # Create progress bar for pages (we don't know total pages upfront)
        pbar = tqdm(desc="Fetching TMDB pages", unit="page", initial=0)

        try:
            while True:
                # Get movies from discover endpoint for current page
                async def do_discover():
                    return await self._tmdb_client.discover().movie(
                        page=page, include_adult=include_adult, include_video=include_video, sort_by=sort_by, **kwargs
                    )

                movies_raw = await self._retry._execute_with_retry(do_discover)
                movies = self._as_list(movies_raw)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"movies": total_fetched, "current_page": page})

                # If no movies returned, we've reached the end
                if not movies:
                    break

                # Process each movie on this page
                for movie in movies:
                    # Apply max_results limit if specified
                    if max_results and total_fetched >= max_results:
                        pbar.close()
                        # Ensure uploaders are flushed before early return
                        if self.titles_uploader:
                            self.titles_uploader.shutdown()
                        return all_movies

                    movie_dict = None
                    total_fetched += 1

                    # Update progress bar with current movie count
                    pbar.set_postfix({"movies": total_fetched, "current_page": page})

                    # First save raw JSON backup
                    movie_id = self._get_value(movie, "id")
                    backup_file = backup_dir / f"movie_{movie_id}.json"

                    try:
                        # Convert movie object to dictionary for JSON serialization
                        if isinstance(movie, dict):
                            # Already a dictionary
                            movie_dict = movie
                        else:
                            # Convert from object
                            movie_dict = movie.__dict__

                        # Save raw data to disk
                        with open(backup_file, "w", encoding="utf-8") as f:
                            json.dump(scrub_serializable(movie_dict), f, indent=2, default=str)
                    except Exception as e:
                        # Log but don't fail on backup error
                        print(f"Warning: Failed to save backup for movie {movie_id}: {e}")

                    # Now convert to Title record
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

                    # Build metadata from all available fields except the core ones
                    metadata = {}
                    core_fields = {"id", "title", "name", "release_date"}  # Core fields handled separately

                    # If we don't have movie_dict, create a minimal one
                    if not movie_dict:
                        movie_dict = {"id": movie_id}

                    # Use the movie_dict we created for backup as the source of metadata
                    for key, value in movie_dict.items():
                        if key not in core_fields and value is not None:
                            metadata[key] = value

                    record = Title(record_id=movie_id, title=movie_title, year=year, type=TitleType.MOVIE, metadata=metadata)
                    all_movies.append(record)

                    # Batch save titles if uploader is configured
                    if self.titles_uploader:
                        await self.titles_uploader.add(record)

                # Move to next page
                page += 1

                # Safety limit to prevent infinite loops (TMDB typically has max 500 pages)
                if page > max_page:
                    print(f"Reached maximum page limit for get_all_movies; stopping at page {page}.")
                    break

        except Exception as e:
            # Return what we have so far on error
            print(f"Error in get_all_movies: {e}")
        finally:
            # Always close the progress bar
            pbar.close()
            # Ensure uploaders are flushed
            if self.titles_uploader:
                self.titles_uploader.shutdown()

        return all_movies

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

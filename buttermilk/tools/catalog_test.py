import datetime
import os
from typing import Any, Iterable, Optional

import shortuuid
from autogen_core.tools import FunctionTool
from pydantic import BaseModel, ConfigDict, Field, field_validator
from themoviedb import aioTMDb

from buttermilk._core.contract import ErrorEvent
from buttermilk._core.retry import RetryWrapper
from buttermilk.utils.validators import make_list_validator  # Pydantic validators


# Data Models
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


class TMDBTool:
    """Tool for searching movie availability using TMDB API.

    This tool searches for movies and returns availability information for the specified region.
    Returns null observations when no results are found.
    """

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.themoviedb.org/3", language: str = "en-US", region: str = "AU"):
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        self.base_url = base_url
        self.language = language
        self.region = (region or "").upper() or "AU"

        if not self.api_key:
            raise ValueError("TMDB API key is required. Please set TMDB_API_KEY environment variable or pass api_key parameter.")
        # Underlying client + retry wrapper
        self._tmdb_client = aioTMDb(key=self.api_key, language=language, region=self.region)
        self._retry = RetryWrapper(client=self._tmdb_client)

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

    async def search_movie_availability(self, title: str, year: Optional[int] = None, region: str = "US") -> list[Observation]:  # noqa: PLR0912, PLR0914
        """Search for a movie and return availability information.

        Args:
            title: Movie title to search for
            year: Optional release year to filter by
            region: Region code for availability checking

        Returns:
            List of Observations with availability data, or list with one null result if not found
        """
        region = self._normalize_region(region)

        try:
            # Search for movies using TMDB API (robust to SDK return shapes)
            async def do_search():
                return await self._tmdb_client.search().movies(query=title, year=year)

            search_raw = await self._retry._execute_with_retry(do_search)
            search_results = self._as_list(search_raw)

            if not search_results:
                # No movies found - return null observation
                obs = Observation(
                    record_id=f"tmdb_search_{title}_{region}",
                    provider_name=None,
                    provider_id=None,
                    provider_type=None,
                    region=region,
                    available=False,
                    source="TMDB",
                    price=None,
                    currency=None,
                    format=None,
                    metadata={"search_title": title, "search_year": year, "total_results": 0},
                )
                return [obs]

            # Get the first/best match and return a single observation capturing movie info
            movie = search_results[0]
            movie_id = TMDBTool._get_value(movie, "id")

            # Build rich metadata from the movie result; include common fields when present
            movie_meta: dict[str, Any] = {
                "id": TMDBTool._get_value(movie, "id"),
                "title": TMDBTool._get_value(movie, "title", "name", "original_title", "original_name"),
                "original_title": TMDBTool._get_value(movie, "original_title", "original_name"),
                "overview": TMDBTool._get_value(movie, "overview"),
                "release_date": TMDBTool._get_value(movie, "release_date", "first_air_date"),
                "popularity": TMDBTool._get_value(movie, "popularity"),
                "vote_average": TMDBTool._get_value(movie, "vote_average"),
                "vote_count": TMDBTool._get_value(movie, "vote_count"),
                "poster_path": TMDBTool._get_value(movie, "poster_path"),
                "backdrop_path": TMDBTool._get_value(movie, "backdrop_path"),
                "genre_ids": TMDBTool._get_value(movie, "genre_ids"),
            }

            metadata = {
                "search_title": title,
                "search_year": year,
                "movie_id": movie_id,
                "tmdb_movie": {k: v for k, v in movie_meta.items() if v is not None},
            }

            obs = Observation(
                record_id=f"tmdb_{movie_id}_{region}_search_result",
                provider_name=None,
                provider_id=None,
                provider_type=None,
                region=region,
                available=False,  # Availability unknown without providers lookup; default to False
                source="TMDB",
                price=None,
                currency=None,
                format=None,
                metadata=metadata,
            )
            return [obs]

        except Exception as e:
            # Create error observation for search failures
            safe_message = TMDBTool._sanitize_rate_limit_message(str(e))
            error_event = ErrorEvent(content=safe_message, source="TMDB")
            error_observation = Observation(
                record_id=f"tmdb_search_error_{title}_{region}",
                provider_name=None,
                provider_id=None,
                provider_type=None,
                region=region,
                available=False,
                source="TMDB",
                price=None,
                currency=None,
                format=None,
                metadata={"search_title": title, "search_year": year, "error_type": "search_failure"},
                error=[error_event],
            )
            return [error_observation]

    def as_tool(self) -> FunctionTool:
        """Return as autogen FunctionTool for agent integration."""
        return FunctionTool(
            name="tmdb_search",
            description=(
                "Search for movie availability information using The Movie Database (TMDB). "
                "Provide movie title, optional year, and region to get availability data."
            ),
            func=self.search_movie_availability,
            strict=False,  # Allow default parameters for better UX
        )

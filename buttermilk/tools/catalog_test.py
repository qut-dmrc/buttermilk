import datetime
import os
from typing import Optional

import numpy as np
import shortuuid
from autogen_core.tools import FunctionTool
from pydantic import BaseModel, ConfigDict, Field, field_validator
from themoviedb import aioTMDb

from buttermilk._core.contract import ErrorEvent
from buttermilk.utils.validators import make_list_validator  # Pydantic validators


# Data Models
class Observation(BaseModel):
    """Represents availability information for a movie in a specific region."""

    record_id: str = Field(..., description="Record under test")
    call_id: str = Field(
        default_factory=lambda: str(shortuuid.uuid()),  # Use shortuuid for brevity
        description="A unique ID for this specific agent execution/response. Attempts to use Weave call ID if available.",
    )
    test_date: Optional[datetime.datetime] = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="Timestamp when the observation was made",
    )

    provider_id: Optional[str] = Field(..., description="Unique ID of the streaming/availability provider")
    provider_name: Optional[str] = Field(..., description="Name of the streaming/availability provider")
    provider_type: Optional[str] = Field(..., description="Type of availability (flatrate, rent, buy)")
    region: str = Field(..., description="Geographical region of the observation (e.g., US, UK)")
    price: Optional[float] = Field(None, description="Price for renting or buying, if applicable")
    currency: Optional[str] = Field(None, description="Currency of the price, if applicable")
    format: Optional[str] = Field(None, description="Format of the content (e.g., HD, SD, 4K)")
    season: Optional[str] = Field(None, description="Season number if applicable")
    episode: Optional[str] = Field(None, description="Episode number if applicable")
    match_title: Optional[str] = Field(None, description="Matched title from the provider")
    match_author: Optional[str] = Field(None, description="Matched author/director from the provider")
    available: bool = Field(..., description="Whether the title is available")
    source: Optional[str] = Field(None, description="Source of the observation data")
    metadata: dict = Field(..., description="Metadata about the title availability")

    error: list[ErrorEvent] = Field(
        default_factory=list,
        description="List of error messages accumulated during processing related to this message.",
    )
    _ensure_error_list: classmethod = field_validator("error", mode="before")(make_list_validator())  # type: ignore

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,  # Be strict by default
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={
            np.bool_: bool,  # Handle numpy bools
            datetime.datetime: lambda v: v.isoformat(),  # Standard ISO format for datetimes
        },
        validate_assignment=True,
        exclude_unset=True,  # Exclude fields not explicitly set
        exclude_none=True,  # Exclude fields with None values
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
        self.region = region
        
        if not self.api_key:
            raise ValueError(
                "TMDB API key is required. Please set TMDB_API_KEY environment variable or pass api_key parameter."
            )
        
        self.tmdb = aioTMDb(key=self.api_key, language=language, region=region)

    async def search_movie_availability(self, title: str, year: Optional[int] = None, region: str = "US") -> list[Observation]:
        """Search for a movie and return availability information.

        Args:
            title: Movie title to search for
            year: Optional release year to filter by
            region: Region code for availability checking

        Returns:
            List of Observations with availability data, or list with one null result if not found
        """
        try:
            # Search for movies using TMDB API
            search_results = await self.tmdb.search().movies(query=title, year=year)
            
            if not search_results or len(search_results) == 0:
                # No movies found - return null observation
                obs = Observation(
                    record_id=f"tmdb_search_{title}_{region}",
                    provider_name=None,
                    provider_id=None,
                    provider_type=None,
                    region=region,
                    available=False,
                    match_title=title,
                    source="TMDB",
                    metadata={"search_title": title, "search_year": year, "total_results": 0}
                )
                return [obs]

            # Get the first/best match
            movie = search_results[0]
            movie_id = movie.id
            match_title = movie.title

            # Get watch providers for this movie in the specified region
            try:
                watch_providers = await self.tmdb.movies(movie_id).watch_providers()
                
                # Check if there are providers for the specified region
                if region not in watch_providers.get("results", {}):
                    # Movie found but no availability in this region
                    obs = Observation(
                        record_id=f"tmdb_{movie_id}_{region}_unavailable",
                        provider_name=None,
                        provider_id=None,
                        provider_type=None,
                        region=region,
                        available=False,
                        match_title=match_title,
                        source="TMDB",
                        metadata={
                            "search_title": title,
                            "search_year": year,
                            "movie_id": movie_id,
                            "available_regions": list(watch_providers.get("results", {}).keys())
                        }
                    )
                    return [obs]

                region_data = watch_providers["results"][region]
                observations = []

                # Process different provider types (flatrate, rent, buy)
                for provider_type in ["flatrate", "rent", "buy"]:
                    if provider_type in region_data:
                        for provider in region_data[provider_type]:
                            obs = Observation(
                                record_id=f"tmdb_{provider['provider_id']}_{movie_id}_{region}_{provider_type}",
                                provider_name=provider["provider_name"],
                                provider_id=str(provider["provider_id"]),
                                provider_type=provider_type,
                                region=region,
                                available=True,
                                match_title=match_title,
                                source="TMDB",
                                metadata={
                                    "search_title": title,
                                    "search_year": year,
                                    "movie_id": movie_id,
                                    "provider_display_priority": provider.get("display_priority"),
                                    "provider_logo_path": provider.get("logo_path")
                                }
                            )
                            observations.append(obs)

                if not observations:
                    # Movie found but no providers available in region
                    obs = Observation(
                        record_id=f"tmdb_{movie_id}_{region}_no_providers",
                        provider_name=None,
                        provider_id=None,
                        provider_type=None,
                        region=region,
                        available=False,
                        match_title=match_title,
                        source="TMDB",
                        metadata={
                            "search_title": title,
                            "search_year": year,
                            "movie_id": movie_id,
                            "region_data_available": True
                        }
                    )
                    observations.append(obs)

                return observations

            except Exception as provider_error:
                # Error getting watch providers, but movie was found
                error_event = ErrorEvent(content=f"Watch provider lookup failed: {str(provider_error)}", source="TMDB")
                obs = Observation(
                    record_id=f"tmdb_{movie_id}_{region}_provider_error",
                    provider_name=None,
                    provider_id=None,
                    provider_type=None,
                    region=region,
                    available=False,
                    match_title=match_title,
                    source="TMDB",
                    metadata={
                        "search_title": title,
                        "search_year": year,
                        "movie_id": movie_id,
                        "error_type": "provider_lookup"
                    },
                    error=[error_event]
                )
                return [obs]

        except Exception as e:
            # Create error observation for search failures
            error_event = ErrorEvent(content=str(e), source="TMDB")
            error_observation = Observation(
                record_id=f"tmdb_search_error_{title}_{region}",
                provider_name=None,
                provider_id=None,
                provider_type=None,
                region=region,
                available=False,
                match_title=title,
                source="TMDB",
                metadata={
                    "search_title": title,
                    "search_year": year,
                    "error_type": "search_failure"
                },
                error=[error_event]
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

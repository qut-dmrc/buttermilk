"""Test TMDBTool as a processor yielding Observations."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk.pipeline import PipelineOrchestrator
from buttermilk.tools.catalog_test import Observation, Title, TMDBTool


class TestTMDBMultiProcessor:
    """Test TMDBTool yields Observations instead of mutating Titles."""

    @pytest.mark.anyio
    async def test_tmdb_yields_observations(self):
        """Test that TMDBTool.process() yields Observation records."""
        # Create a mock TMDBTool
        tool = TMDBTool(api_key="fake_key", region="US")

        # Create a test Title
        title = Title(
            record_id="tmdb_123",
            title="Test Movie",
            year=2024
        )

        # Mock get_availability to return test observations
        mock_observations = [
            Observation(
                record_id="tmdb_123",
                title="Test Movie",
                year=2024,
                provider_name="Netflix",
                provider_id="8",
                provider_type="flatrate",
                region="US",
                available=True,
                source="TMDB"
            ),
            Observation(
                record_id="tmdb_123",
                title="Test Movie",
                year=2024,
                provider_name="Amazon Prime",
                provider_id="9",
                provider_type="flatrate",
                region="US",
                available=True,
                source="TMDB"
            )
        ]

        tool.get_availability = AsyncMock(return_value=mock_observations)

        # Process the title and collect observations
        observations = []
        async for obs in tool.process(title):
            observations.append(obs)

        # Verify we got the expected observations
        assert len(observations) == 2
        assert all(isinstance(obs, Observation) for obs in observations)
        assert observations[0].provider_name == "Netflix"
        assert observations[1].provider_name == "Amazon Prime"
        assert all(obs.record_id == "tmdb_123" for obs in observations)
        assert all(obs.title == "Test Movie" for obs in observations)

    @pytest.mark.anyio
    async def test_tmdb_yields_error_observation_on_failure(self):
        """Test that TMDBTool yields an error Observation on failure."""
        tool = TMDBTool(api_key="fake_key", region="US")

        title = Title(
            record_id="tmdb_456",
            title="Error Movie",
            year=2023
        )

        # Mock get_availability to raise an error
        tool.get_availability = AsyncMock(side_effect=Exception("API Error"))

        # Process and collect observations
        observations = []
        async for obs in tool.process(title):
            observations.append(obs)

        # Should yield one error observation
        assert len(observations) == 1
        error_obs = observations[0]
        assert isinstance(error_obs, Observation)
        assert error_obs.record_id == "tmdb_456"
        assert error_obs.title == "Error Movie"
        assert error_obs.year == 2023
        assert error_obs.region == "UNKNOWN"
        assert error_obs.available is False
        assert len(error_obs.error) == 1
        assert "API Error" in error_obs.error[0].content

    @pytest.mark.anyio
    async def test_title_not_mutated(self):
        """Test that original Title record is not mutated."""
        tool = TMDBTool(api_key="fake_key")

        title = Title(
            record_id="tmdb_789",
            title="Immutable Movie",
            year=2024,
            metadata={"original": "data"}
        )

        # Mock get_availability
        tool.get_availability = AsyncMock(return_value=[
            Observation(
                record_id="tmdb_789",
                title="Immutable Movie",
                year=2024,
                provider_name="Test",
                provider_id="1",
                provider_type="flatrate",
                region="US",
                available=True,
                source="TMDB"
            )
        ])

        # Process the title
        observations = []
        async for obs in tool.process(title):
            observations.append(obs)

        # Verify title was not mutated
        assert title.metadata == {"original": "data"}
        assert "tmdb_availability" not in title.metadata
        assert not hasattr(title, "error") or title.error == []
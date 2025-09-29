"""Tests for catalog_test Observation model."""

from datetime import datetime

import pytest

from buttermilk.tools.catalog_test import Observation, Title


class TestObservationModel:
    """Test the Observation model and its serialization."""

    def test_title_inherits_from_basemodel(self):
        """Verify Title properly inherits from BaseModel."""
        # Create a Title instance
        title = Title(
            record_id="tmdb_123",
            title="Test Movie",
            year=2024
        )

        assert title.record_id == "tmdb_123"
        assert title.title == "Test Movie"
        assert title.year == 2024
        assert title.dataset_name == "tmdb"  # Default value
        assert title.split_type == "default"  # Default value
        assert title.error == []  # Default empty list

    def test_observation_includes_call_id(self):
        """Test that Observation automatically generates call_id."""
        obs = Observation(
            record_id="123",  # Foreign key to titles table
            title="Test Movie",
            year=2024,
            region="US",
            available=True,
            source="TMDB"
        )

        # Verify call_id is generated
        assert obs.call_id is not None
        assert len(obs.call_id) > 0
        assert isinstance(obs.call_id, str)

        # Verify test_date is generated
        assert obs.test_date is not None
        assert isinstance(obs.test_date, datetime)

    def test_observation_serialization_includes_defaults(self):
        """Test that model_dump includes fields with default factories."""
        obs = Observation(
            record_id="123",  # Foreign key to titles table
            title="Test Movie",
            year=2024,
            region="US",
            available=True,
            source="TMDB"
        )

        # Test default serialization
        dumped = obs.model_dump()
        assert "call_id" in dumped
        assert "test_date" in dumped
        assert "title" in dumped
        assert "year" in dumped

        # Test JSON mode (used by BigQuery)
        json_dumped = obs.model_dump(mode="json")
        assert "call_id" in json_dumped
        assert "test_date" in json_dumped
        assert json_dumped["call_id"] == obs.call_id

    def test_observation_with_provider_details(self):
        """Test creating observation with full provider details."""
        obs = Observation(
            record_id="123",  # Foreign key to titles table
            title="Test Movie",
            year=2024,
            provider_id="456",
            provider_name="Netflix",
            provider_type="flatrate",
            region="US",
            available=True,
            source="TMDB",
            metadata={"movie_id": "123"}
        )

        assert obs.provider_id == "456"
        assert obs.provider_name == "Netflix"
        assert obs.provider_type == "flatrate"
        assert obs.metadata == {"movie_id": "123"}

    def test_observation_null_availability(self):
        """Test creating observation for unavailable content."""
        obs = Observation(
            record_id="123",  # Foreign key to titles table
            title="Test Movie",
            year=2024,
            provider_name=None,
            provider_id=None,
            provider_type=None,
            region="US",
            available=False,
            source="TMDB",
            metadata={"movie_id": "123"}
        )

        assert obs.available is False
        assert obs.provider_name is None
        assert obs.provider_id is None
        assert obs.provider_type is None

        # Verify serialization still includes call_id
        dumped = obs.model_dump(mode="json")
        assert "call_id" in dumped

    def test_observation_with_error(self):
        """Test creating observation with error information."""
        from buttermilk._core.contract import ErrorEvent

        error_event = ErrorEvent(
            content="Rate limit exceeded",
            source="TMDB"
        )

        obs = Observation(
            record_id="123",  # Foreign key to titles table
            title="Test Movie",
            year=2024,
            region="US",
            available=False,
            source="TMDB",
            metadata={
                "movie_id": "123",
                "error_type": "availability_check_failure"
            },
            error=[error_event]
        )

        assert obs.error == [error_event]
        assert obs.metadata["error_type"] == "availability_check_failure"

        # Verify serialization includes all fields
        dumped = obs.model_dump(mode="json")
        assert "call_id" in dumped
        assert "error" in dumped
        assert len(dumped["error"]) == 1

    def test_title_field_is_required(self):
        """Test that title field is required for Observation."""
        with pytest.raises(Exception) as exc_info:
            Observation(
                record_id="123",  # Foreign key to titles table
                # Missing title field
                region="US",
                available=True,
                source="TMDB"
            )

        assert "title" in str(exc_info.value).lower()
        assert "required" in str(exc_info.value).lower()

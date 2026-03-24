"""Test ToxicityModel credentials support.

Tests that the ToxicityModel base class accepts a credentials dict
and provides a _get_credential() helper method for secure credential access.
"""

from typing import Any

import pytest

from buttermilk.toxicity.toxicity import ToxicityModel
from buttermilk.toxicity.types import EvalRecord


class MockToxicityModel(ToxicityModel):
    """Minimal mock ToxicityModel for testing credentials."""

    def init_client(self) -> None:
        """Mock init_client to avoid NotImplementedError."""
        self._client = "mock_client"

    def make_prompt(self, content: str) -> str:
        """Mock make_prompt implementation."""
        return content

    def interpret(self, response: Any) -> EvalRecord:
        """Mock interpret implementation."""
        return EvalRecord(prediction=False)

    async def _process_record(self, context):
        """Mock _process_record to satisfy ProcessorCore abstract method."""
        yield context.record


class TestToxicityModelCredentials:
    """Test credentials handling in ToxicityModel base class."""

    def test_accepts_credentials_dict(self):
        """Test that ToxicityModel accepts credentials dict field."""
        credentials = {
            "api_key": "test_key_123",
            "endpoint": "https://api.example.com",
        }

        model = MockToxicityModel(
            model="test_model",
            process_chain="test_chain",
            standard="test_standard",
            credentials=credentials,
        )

        assert model.credentials == credentials

    def test_get_credential_returns_value_when_present(self):
        """Test _get_credential() returns value from credentials dict."""
        credentials = {
            "api_key": "secret_key_456",
            "region": "us-east-1",
        }

        model = MockToxicityModel(
            model="test_model",
            process_chain="test_chain",
            standard="test_standard",
            credentials=credentials,
        )

        # Should return the credential value
        assert model._get_credential("api_key") == "secret_key_456"
        assert model._get_credential("region") == "us-east-1"

    def test_get_credential_raises_keyerror_when_missing(self):
        """Test _get_credential() raises KeyError for missing credentials."""
        credentials = {
            "api_key": "test_key",
        }

        model = MockToxicityModel(
            model="test_model",
            process_chain="test_chain",
            standard="test_standard",
            credentials=credentials,
        )

        # Should raise KeyError for missing credential
        with pytest.raises(KeyError, match="missing_key"):
            model._get_credential("missing_key")

    def test_get_credential_with_empty_credentials(self):
        """Test _get_credential() with empty credentials dict."""
        model = MockToxicityModel(
            model="test_model",
            process_chain="test_chain",
            standard="test_standard",
            credentials={},
        )

        # Should raise KeyError when credentials dict is empty
        with pytest.raises(KeyError, match="api_key"):
            model._get_credential("api_key")

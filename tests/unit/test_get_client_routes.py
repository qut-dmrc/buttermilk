"""Unit tests for LLMs.get_client() route construction.

Covers the two key routes added in PR #438:
- VERTEX_XAI: GCP-bearer-authed OpenAI-compat endpoint for Grok-on-Vertex
- GEMINI_VERTEX: native litellm vertex_ai/ path for Gemini models

Mocks only the GCP credential boundary (bm.gcp_credentials / bm.get_gcp_access_token)
and the litellm network call (_get_acompletion). No buttermilk internals are mocked.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from buttermilk._core.llms import ClientType, LLMConfig, LiteLLMWrapper, LLMs
from buttermilk._core.messages import ModelInfo, UserMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_MODEL_INFO: ModelInfo = {
    "family": "test",
    "vision": False,
    "function_calling": False,
    "json_output": False,
    "structured_output": False,
}

_GROK_BASE_URL = "https://us-east5-aiplatform.googleapis.com/v1/projects/test-project/locations/us-east5/endpoints/openapi"
_GROK_MODEL = "xai/grok-4.1-fast-reasoning"
_GEMINI_MODEL = "google/gemini-3-flash-preview"


def _make_llms_with(name: str, config: LLMConfig) -> LLMs:
    return LLMs(connections={name: config})


def _mock_bm(token: str = "fake-gcp-token") -> Mock:
    m = Mock()
    m.gcp_credentials = Mock()  # truthy
    m.get_gcp_access_token = Mock(return_value=token)
    return m


# ---------------------------------------------------------------------------
# VERTEX_XAI route
# ---------------------------------------------------------------------------


class TestVertexXaiGetClient:
    """VERTEX_XAI get_client() route: bearer header, api_model, token_provider."""

    def _make_config(self) -> LLMConfig:
        return LLMConfig(
            client_type=ClientType.VERTEX_XAI,
            base_url=_GROK_BASE_URL,
            model_info=_MINIMAL_MODEL_INFO,
            configs={"model": _GROK_MODEL},
        )

    def test_bearer_header_set(self):
        """VERTEX_XAI wrapper carries Authorization: Bearer <gcp-token> in extra_headers."""
        config = self._make_config()
        llms = _make_llms_with("grok", config)
        mock_bm = _mock_bm("sentinel-token")

        with patch("buttermilk._core.llms.bm", mock_bm):
            wrapper = llms.get_client("grok")

        assert isinstance(wrapper, LiteLLMWrapper)
        assert wrapper.extra_headers is not None
        assert "Authorization" in wrapper.extra_headers
        assert wrapper.extra_headers["Authorization"] == "Bearer sentinel-token"

    def test_token_provider_wired(self):
        """VERTEX_XAI wrapper has a callable token_provider that fetches fresh GCP tokens."""
        config = self._make_config()
        llms = _make_llms_with("grok", config)
        mock_bm = _mock_bm("initial-token")

        with patch("buttermilk._core.llms.bm", mock_bm):
            wrapper = llms.get_client("grok")
            assert callable(wrapper.token_provider)
            # token_provider delegates to bm.get_gcp_access_token on each call
            mock_bm.get_gcp_access_token.return_value = "refreshed-token"
            assert wrapper.token_provider() == "refreshed-token"

    @pytest.mark.anyio
    async def test_api_model_uses_openai_prefix(self):
        """VERTEX_XAI routes create() with model='openai/<model>' (OpenAI-compat transport)."""
        config = self._make_config()
        llms = _make_llms_with("grok", config)
        mock_bm = _mock_bm()

        captured: list[dict] = []

        async def mock_acompletion(**kwargs):
            captured.append(kwargs)
            resp = Mock()
            choice = Mock()
            choice.message = Mock(content="ok", tool_calls=None, reasoning_content=None, thought=None)
            choice.finish_reason = "stop"
            resp.choices = [choice]
            resp.usage = Mock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
            resp.cached = False
            resp.model = "test"
            return resp

        with (
            patch("buttermilk._core.llms.bm", mock_bm),
            patch("buttermilk._core.llms._get_acompletion", return_value=mock_acompletion),
        ):
            wrapper = llms.get_client("grok")
            await wrapper.create(messages=[UserMessage(content="hello", source="user")])

        assert captured, "acompletion was not called"
        model_arg = captured[0]["model"]
        assert model_arg == f"openai/{_GROK_MODEL}", (
            f"Expected openai/<model> for VERTEX_XAI transport, got: {model_arg}"
        )

    def test_raises_without_gcp_credentials(self):
        """VERTEX_XAI raises ValueError when GCP credentials are absent."""
        config = self._make_config()
        llms = _make_llms_with("grok", config)
        mock_bm = Mock()
        mock_bm.gcp_credentials = None  # falsy — no credentials

        with patch("buttermilk._core.llms.bm", mock_bm), pytest.raises(ValueError, match="GCP credentials"):
            llms.get_client("grok")


# ---------------------------------------------------------------------------
# GEMINI_VERTEX route
# ---------------------------------------------------------------------------


class TestGeminiVertexGetClient:
    """GEMINI_VERTEX get_client() route: vertex_project, vertex_location, token_provider."""

    def _make_config(self, region: str | None = "us-central1") -> LLMConfig:
        configs: dict = {"model": _GEMINI_MODEL, "project_id": "my-gcp-project"}
        if region is not None:
            configs["region"] = region
        return LLMConfig(
            client_type=ClientType.GEMINI_VERTEX,
            model_info=_MINIMAL_MODEL_INFO,
            configs=configs,
        )

    def test_vertex_project_and_location_set(self):
        """GEMINI_VERTEX wrapper carries vertex_project and vertex_location from config."""
        config = self._make_config(region="us-central1")
        llms = _make_llms_with("gemini", config)
        mock_bm = _mock_bm()

        with patch("buttermilk._core.llms.bm", mock_bm):
            wrapper = llms.get_client("gemini")

        assert wrapper.vertex_project == "my-gcp-project"
        assert wrapper.vertex_location == "us-central1"

    def test_region_defaults_to_global(self):
        """GEMINI_VERTEX defaults vertex_location to 'global' when registry omits region."""
        config = self._make_config(region=None)
        llms = _make_llms_with("gemini", config)
        mock_bm = _mock_bm()

        with patch("buttermilk._core.llms.bm", mock_bm):
            wrapper = llms.get_client("gemini")

        assert wrapper.vertex_location == "global", (
            "gemini-3.x models only resolve under the 'global' location; "
            "get_client() must default to 'global' when region is absent"
        )

    def test_token_provider_wired(self):
        """GEMINI_VERTEX wrapper has a callable token_provider for GCP auth."""
        config = self._make_config()
        llms = _make_llms_with("gemini", config)
        mock_bm = _mock_bm("gemini-token")

        with patch("buttermilk._core.llms.bm", mock_bm):
            wrapper = llms.get_client("gemini")
            assert callable(wrapper.token_provider)
            assert wrapper.token_provider() == "gemini-token"

    def test_no_extra_headers_for_gemini(self):
        """GEMINI_VERTEX does NOT inject Authorization extra_header (uses litellm native auth)."""
        config = self._make_config()
        llms = _make_llms_with("gemini", config)
        mock_bm = _mock_bm()

        with patch("buttermilk._core.llms.bm", mock_bm):
            wrapper = llms.get_client("gemini")

        assert wrapper.extra_headers is None

"""Tests for the LLMConfig schema after the ClientType deletion.

Routing is driven entirely by the full provider-prefixed `litellm_model`; there is
no `client_type` field/enum any more.
"""

import pytest

from buttermilk._core.llms import LLMConfig
from buttermilk._core.messages import ModelInfo


def _model_info() -> ModelInfo:
    return ModelInfo(
        family="test-family",
        function_calling=True,
        json_output=True,
        vision=False,
        structured_output=False,
    )


def test_llm_config_requires_litellm_model():
    """litellm_model is required; constructing without it raises."""
    with pytest.raises(Exception):
        LLMConfig(api_key="test-key", model_info=_model_info(), configs={})


def test_llm_config_fields_and_provider_segment():
    """LLMConfig carries plain optionals and exposes the provider segment."""
    config = LLMConfig(
        litellm_model="azure/gpt-5-mini",
        api_key="test-key",
        base_url="https://example.azure.com",
        api_version="2024-12-01-preview",
        model_info=_model_info(),
        configs={"model": "gpt-5-mini"},
    )
    assert config.litellm_model == "azure/gpt-5-mini"
    assert config.provider_segment == "azure"
    assert config.api_version == "2024-12-01-preview"
    assert config.base_url == "https://example.azure.com"

    vertex = LLMConfig(
        litellm_model="vertex_ai/gemini-3-flash-preview",
        region="global",
        model_info=_model_info(),
        configs={"model": "google/gemini-3-flash-preview"},
    )
    assert vertex.provider_segment == "vertex_ai"
    assert vertex.region == "global"
    assert vertex.api_key is None  # ambient ADC


def test_no_client_type_field():
    """The client_type field (and old api_type/obj/connection) are gone."""
    config = LLMConfig(litellm_model="openai/gpt-4o", model_info=_model_info(), configs={})
    assert not hasattr(config, "client_type")
    assert not hasattr(config, "api_type")
    assert not hasattr(config, "obj")
    assert not hasattr(config, "connection")

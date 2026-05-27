"""Tests for the refactored LLM configuration schema."""

import pytest

from buttermilk._core.llms import ClientType, LLMConfig
from buttermilk._core.messages import ModelInfo


def test_client_type_enum():
    """Test that ClientType enum has all expected values."""
    expected_types = [
        "openai",
        "azure",
        "anthropic",
        "anthropic_vertex",
        "gemini",
        "gemini_vertex",
        "vertex_openai",
        "llama_vertex",
        "deepseek_vertex",
        "mistral_vertex",
        "huggingface",
        "zentropi",
    ]
    actual_types = [ct.value for ct in ClientType]
    assert set(expected_types) == set(actual_types)


def test_llm_config_validates_client_type():
    """Test that LLMConfig properly validates client_type field."""
    model_info = ModelInfo(
        family="test-family",
        function_calling=True,
        json_output=True,
        vision=False,
        structured_output=False,
    )

    # Valid client type as string
    config = LLMConfig(
        client_type="openai",
        api_key="test-key",
        model_info=model_info,
        configs={"model": "gpt-4"},
    )
    assert config.client_type == ClientType.OPENAI

    # Valid client type as enum
    config = LLMConfig(
        client_type=ClientType.AZURE,
        api_key="test-key",
        base_url="https://example.azure.com",
        model_info=model_info,
        configs={"model": "gpt-4"},
    )
    assert config.client_type == ClientType.AZURE

    # Invalid client type should raise error
    with pytest.raises(ValueError, match="Unsupported client_type"):
        LLMConfig(
            client_type="invalid_type",
            api_key="test-key",
            model_info=model_info,
            configs={},
        )


def test_no_api_type_field():
    """Test that api_type field is no longer used."""
    model_info = ModelInfo(
        family="test-family",
        function_calling=True,
        json_output=True,
        vision=False,
        structured_output=False,
    )

    # Verify LLMConfig doesn't have api_type field
    config = LLMConfig(client_type="openai", api_key="test-key", model_info=model_info, configs={})

    assert not hasattr(config, "api_type")
    assert not hasattr(config, "obj")
    assert not hasattr(config, "connection")

"""Tests for the refactored LLM configuration schema."""

import pytest
from autogen_core.models import ModelInfo

from buttermilk._core.llms import ClientType, LLMConfig, LLMs


def test_client_type_enum():
    """Test that ClientType enum has all expected values."""
    expected_types = [
        "openai", "azure", "anthropic", "anthropic_vertex",
        "gemini", "gemini_vertex", "vertex_openai"
    ]
    actual_types = [ct.value for ct in ClientType]
    assert set(expected_types) == set(actual_types)


def test_llm_config_validates_client_type():
    """Test that LLMConfig properly validates client_type field."""
    model_info = ModelInfo(
        family="test-family",
        function_calling=True,
        json_output=True
    )
    
    # Valid client type as string
    config = LLMConfig(
        client_type="openai",
        api_key="test-key",
        model_info=model_info,
        configs={"model": "gpt-4"}
    )
    assert config.client_type == ClientType.OPENAI
    
    # Valid client type as enum
    config = LLMConfig(
        client_type=ClientType.AZURE,
        api_key="test-key",
        base_url="https://example.azure.com",
        model_info=model_info,
        configs={"model": "gpt-4"}
    )
    assert config.client_type == ClientType.AZURE
    
    # Invalid client type should raise error
    with pytest.raises(ValueError, match="Unsupported client_type"):
        LLMConfig(
            client_type="invalid_type",
            api_key="test-key",
            model_info=model_info,
            configs={}
        )


def test_clean_branching_logic():
    """Test that get_autogen_chat_client has clean branching per client_type."""
    # This test verifies the branching logic is clean by checking that
    # each client_type has exactly one branch in the implementation
    
    import inspect
    
    # Get the source code of get_autogen_chat_client
    source = inspect.getsource(LLMs.get_autogen_chat_client)
    
    # Count occurrences of each client type check
    for client_type in ClientType:
        # Each client type should appear exactly once in an if/elif statement
        pattern = f"client_type == ClientType.{client_type.name}"
        occurrences = source.count(pattern)
        assert occurrences == 1, f"ClientType.{client_type.name} appears {occurrences} times, expected 1"
    
    # Ensure no string matching on config.obj
    assert "'anthropic' in config.obj.lower()" not in source
    assert "config.obj.lower()" not in source
    assert "config.obj" not in source


def test_no_api_type_field():
    """Test that api_type field is no longer used."""
    model_info = ModelInfo(
        family="test-family",
        function_calling=True,
        json_output=True
    )
    
    # Verify LLMConfig doesn't have api_type field
    config = LLMConfig(
        client_type="openai",
        api_key="test-key",
        model_info=model_info,
        configs={}
    )
    
    assert not hasattr(config, "api_type")
    assert not hasattr(config, "obj")
    assert not hasattr(config, "connection")

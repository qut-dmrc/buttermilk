"""TRUE end-to-end test for model_parameters configuration.

This test uses REAL components:
- Real LLMs instance from testing.yaml (via debug.yaml llms config)
- Real model_parameters from YAML config
- Real API calls to verify parameters are passed through

NO mocks - validates the complete workflow from YAML config to API call.
"""

from unittest.mock import patch

import pytest
from autogen_core.models import UserMessage

from buttermilk import BM
from buttermilk._core.llms import LLMs, ModelParameters


@pytest.mark.anyio
async def test_model_parameters_loaded_from_yaml(real_llms: LLMs):
    """Verify model_parameters from YAML config are loaded into LLMs instance.

    The debug.yaml config sets:
      model_parameters:
        gemini25flash:
          temperature: 0.5
          max_tokens: 2048
    """
    # Verify model_parameters dict is populated
    assert real_llms.model_parameters, "model_parameters should be loaded from YAML"

    # Check gemini25flash has expected parameters from debug.yaml
    if "gemini25flash" in real_llms.model_parameters:
        params = real_llms.model_parameters["gemini25flash"]
        if isinstance(params, dict):
            params = ModelParameters(**params)

        assert params.temperature == 0.5, (
            f"Expected temperature=0.5 from debug.yaml, got {params.temperature}"
        )
        assert params.max_tokens == 2048, (
            f"Expected max_tokens=2048 from debug.yaml, got {params.max_tokens}"
        )


@pytest.mark.anyio
async def test_get_merged_parameters_returns_yaml_values(real_llms: LLMs):
    """Verify get_merged_parameters() returns YAML-configured values.

    Tests the parameter merge precedence:
    1. LLMConfig.parameters (from models.json) - lowest priority
    2. LLMs.model_parameters (from YAML config) - highest priority
    """
    # Get merged parameters for a model with YAML overrides
    merged = real_llms.get_merged_parameters("gemini25flash")

    assert isinstance(merged, ModelParameters), (
        f"Expected ModelParameters, got {type(merged)}"
    )

    # YAML values should override any models.json defaults
    assert merged.temperature == 0.5, (
        f"Expected temperature=0.5 from YAML override, got {merged.temperature}"
    )
    assert merged.max_tokens == 2048, (
        f"Expected max_tokens=2048 from YAML override, got {merged.max_tokens}"
    )


@pytest.mark.anyio
async def test_get_merged_parameters_model_without_yaml_override(real_llms: LLMs):
    """Verify get_merged_parameters() returns base config for models without YAML overrides."""
    # Pick a model that exists but has no YAML override
    # First find such a model
    for model_name in real_llms.connections:
        if model_name not in real_llms.model_parameters:
            merged = real_llms.get_merged_parameters(model_name)
            # Should still return valid ModelParameters (from LLMConfig or empty)
            assert isinstance(merged, ModelParameters), (
                f"Expected ModelParameters for {model_name}, got {type(merged)}"
            )
            break


@pytest.mark.anyio
async def test_parameters_passed_to_llm_api_call(real_bm: BM, session_runner):
    """TRUE E2E test: verify parameters from YAML are passed through to actual API call.

    This test makes a REAL API call and verifies the parameters were applied.
    We intercept the litellm.acompletion call to verify parameters are passed correctly.
    """
    llm = real_bm.llms["gemini25flash"]

    messages = [
        UserMessage(content="Say 'hello' and nothing else.", source="user"),
    ]

    # Track what parameters were passed to the API
    captured_kwargs = {}

    original_acompletion = None
    try:
        # Import litellm to get original function
        from litellm import acompletion as original_acompletion_func

        original_acompletion = original_acompletion_func
    except ImportError:
        pytest.skip("LiteLLM not installed")

    async def capturing_acompletion(*args, **kwargs):
        """Wrapper that captures kwargs and calls real API."""
        captured_kwargs.update(kwargs)
        return await original_acompletion(*args, **kwargs)

    # Patch acompletion to capture parameters while still calling real API
    with patch("buttermilk._core.llms.acompletion", side_effect=capturing_acompletion):
        result = await llm.create(messages=messages)

    # Verify the call succeeded
    assert result.content, "Should get response from real API"

    # Verify YAML parameters were passed through
    # The debug.yaml sets temperature=0.5 and max_tokens=2048 for gemini25flash
    assert "temperature" in captured_kwargs, (
        f"temperature should be in API call, got keys: {list(captured_kwargs.keys())}"
    )
    assert captured_kwargs["temperature"] == 0.5, (
        f"Expected temperature=0.5 from YAML, got {captured_kwargs.get('temperature')}"
    )

    assert "max_tokens" in captured_kwargs, (
        f"max_tokens should be in API call, got keys: {list(captured_kwargs.keys())}"
    )
    assert captured_kwargs["max_tokens"] == 2048, (
        f"Expected max_tokens=2048 from YAML, got {captured_kwargs.get('max_tokens')}"
    )


@pytest.mark.anyio
async def test_runtime_kwargs_override_yaml_parameters(real_bm: BM, session_runner):
    """Verify runtime kwargs override YAML-configured parameters.

    Parameter precedence (lowest to highest):
    1. LLMConfig.parameters (models.json)
    2. LLMs.model_parameters (YAML)
    3. Runtime kwargs (call-time overrides)
    """
    llm = real_bm.llms["gemini25flash"]

    messages = [
        UserMessage(content="Say 'hi' and nothing else.", source="user"),
    ]

    captured_kwargs = {}

    original_acompletion = None
    try:
        from litellm import acompletion as original_acompletion_func

        original_acompletion = original_acompletion_func
    except ImportError:
        pytest.skip("LiteLLM not installed")

    async def capturing_acompletion(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return await original_acompletion(*args, **kwargs)

    # Call with runtime override - should override YAML's temperature=0.5
    with patch("buttermilk._core.llms.acompletion", side_effect=capturing_acompletion):
        result = await llm.create(messages=messages, temperature=0.9)

    assert result.content, "Should get response from real API"

    # Runtime temperature=0.9 should override YAML's temperature=0.5
    assert captured_kwargs.get("temperature") == 0.9, (
        f"Runtime temperature=0.9 should override YAML, got {captured_kwargs.get('temperature')}"
    )

    # max_tokens should still come from YAML since no runtime override
    assert captured_kwargs.get("max_tokens") == 2048, (
        f"max_tokens should still be 2048 from YAML, got {captured_kwargs.get('max_tokens')}"
    )


@pytest.mark.anyio
async def test_model_parameters_to_api_params_conversion(real_llms: LLMs):
    """Verify to_api_params() correctly converts ModelParameters to API dict."""
    merged = real_llms.get_merged_parameters("gemini25flash")

    api_params = merged.to_api_params()

    assert isinstance(api_params, dict), f"Expected dict, got {type(api_params)}"

    # Only non-None values should be included
    assert "temperature" in api_params
    assert api_params["temperature"] == 0.5

    assert "max_tokens" in api_params
    assert api_params["max_tokens"] == 2048

    # None values should NOT be in output
    if merged.top_p is None:
        assert "top_p" not in api_params, "None values should be excluded"


@pytest.mark.anyio
async def test_model_parameters_merge_with(real_llms: LLMs):
    """Verify merge_with() correctly merges parameters with override precedence."""
    base = real_llms.get_merged_parameters("gemini25flash")

    # Create override parameters
    override = ModelParameters(temperature=0.1, top_p=0.95)

    merged = base.merge_with(override)

    # Override values should take precedence
    assert merged.temperature == 0.1, "override temperature should win"
    assert merged.top_p == 0.95, "override top_p should be set"

    # Non-overridden values should come from base
    assert merged.max_tokens == 2048, "base max_tokens should be preserved"

"""Unit tests for LLM wrapper parameterization.

This test verifies that E2E tests can be parameterized to run with both
AutoGenWrapper and LiteLLMWrapper by using a fixture that switches between
wrapper types.

Test Strategy:
- Use existing fixtures from conftest.py (real_bm, real_llms)
- Introduce new fixture llm_wrapper_type that returns "autogen" or "litellm"
- Verify that the wrapper returned matches the requested type

Expected Failure:
This test WILL FAIL because:
1. The llm_wrapper_type fixture doesn't exist yet
2. The fixture logic to switch wrapper types doesn't exist in conftest.py
"""

from __future__ import annotations

import pytest
from autogen_core.models import UserMessage

from buttermilk._core.llms import AutoGenWrapper, LiteLLMWrapper


@pytest.mark.anyio
async def test_wrapper_type_matches_requested_type(
    real_bm,
    real_llms,
    llm_wrapper_type,
    session_runner,
):
    """Verify that llm_wrapper_type fixture returns correct wrapper class.

    This test uses a parameterized fixture (llm_wrapper_type) that should
    yield "autogen" or "litellm". Based on that value, we should get the
    corresponding wrapper type from real_llms.

    Args:
        real_bm: Real BM instance from testing.yaml
        real_llms: Real LLMs instance
        llm_wrapper_type: Fixture that yields "autogen" or "litellm"
        session_runner: Session-scoped async fixture for event loop

    Expected Behavior:
        - When llm_wrapper_type == "autogen", wrapper is AutoGenWrapper
        - When llm_wrapper_type == "litellm", wrapper is LiteLLMWrapper

    Expected Failure:
        - Fixture 'llm_wrapper_type' not found in conftest.py
    """
    # Get a real model from the cheap models list
    # We use a fixed model name for consistency
    from buttermilk._core.llms import CHEAP_CHAT_MODELS

    model_name = CHEAP_CHAT_MODELS[0]  # Use first available cheap model

    # Get the wrapper from real_llms
    wrapper = real_llms[model_name]

    # Verify the wrapper type matches the requested type
    if llm_wrapper_type == "autogen":
        assert isinstance(
            wrapper, AutoGenWrapper
        ), f"Expected AutoGenWrapper but got {type(wrapper).__name__}"
    elif llm_wrapper_type == "litellm":
        assert isinstance(
            wrapper, LiteLLMWrapper
        ), f"Expected LiteLLMWrapper but got {type(wrapper).__name__}"
    else:
        pytest.fail(f"Unexpected wrapper type: {llm_wrapper_type}")

@pytest.mark.slow
@pytest.mark.anyio
async def test_wrapper_parameterization_creates_functional_wrapper(
    real_bm,
    real_llms,
    llm_wrapper_type,
    session_runner,
):
    """Verify that parameterized wrapper can make basic LLM calls.

    This test ensures that both AutoGenWrapper and LiteLLMWrapper work
    correctly when created via the parameterized fixture.

    Args:
        real_bm: Real BM instance from testing.yaml
        real_llms: Real LLMs instance
        llm_wrapper_type: Fixture that yields "autogen" or "litellm"
        session_runner: Session-scoped async fixture for event loop

    Expected Behavior:
        - Both wrapper types should successfully make a simple LLM call
        - Response should contain expected content

    Expected Failure:
        - Fixture 'llm_wrapper_type' not found in conftest.py
    """
    # Get a real model
    from buttermilk._core.llms import CHEAP_CHAT_MODELS

    model_name = CHEAP_CHAT_MODELS[0]  # Use first available cheap model
    wrapper = real_llms[model_name]

    # Verify wrapper type
    if llm_wrapper_type == "autogen":
        assert isinstance(wrapper, AutoGenWrapper)
    elif llm_wrapper_type == "litellm":
        assert isinstance(wrapper, LiteLLMWrapper)

    # Make a simple call to verify functionality
    # Note: This is a real API call, so we keep it minimal
    messages = [
        UserMessage(content="Say 'test successful' and nothing else.", source="user")
    ]

    response = await wrapper.create(messages=messages)

    # Verify we got a response
    assert response is not None
    assert hasattr(response, "content")
    assert len(response.content) > 0

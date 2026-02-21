"""Unit tests for LLM wrapper functionality.

This test verifies that LiteLLMWrapper works correctly for LLM calls.
"""

from __future__ import annotations

import pytest
from autogen_core.models import UserMessage

from buttermilk._core.llms import LiteLLMWrapper


@pytest.mark.anyio
async def test_wrapper_type_is_litellm(
    real_bm,
    real_llms,
    session_runner,
):
    """Verify that wrapper returned is LiteLLMWrapper.

    Args:
        real_bm: Real BM instance from testing.yaml
        real_llms: Real LLMs instance
        session_runner: Session-scoped async fixture for event loop
    """
    # Get a real model from the cheap models list
    from buttermilk._core.llms import CHEAP_CHAT_MODELS

    model_name = CHEAP_CHAT_MODELS[0]  # Use first available cheap model

    # Get the wrapper from real_llms
    wrapper = real_llms[model_name]

    # Verify the wrapper type is LiteLLMWrapper
<<<<<<< HEAD
    assert isinstance(wrapper, LiteLLMWrapper), f"Expected LiteLLMWrapper but got {type(wrapper).__name__}"
=======
    assert isinstance(
        wrapper, LiteLLMWrapper
    ), f"Expected LiteLLMWrapper but got {type(wrapper).__name__}"
>>>>>>> origin/stable


@pytest.mark.slow
@pytest.mark.anyio
async def test_litellm_wrapper_creates_functional_wrapper(
    real_bm,
    real_llms,
    session_runner,
):
    """Verify that LiteLLMWrapper can make basic LLM calls.

    Args:
        real_bm: Real BM instance from testing.yaml
        real_llms: Real LLMs instance
        session_runner: Session-scoped async fixture for event loop
    """
    # Get a real model
    from buttermilk._core.llms import CHEAP_CHAT_MODELS

    model_name = CHEAP_CHAT_MODELS[0]  # Use first available cheap model
    wrapper = real_llms[model_name]

    # Verify wrapper type
    assert isinstance(wrapper, LiteLLMWrapper)

    # Make a simple call to verify functionality
    # Note: This is a real API call, so we keep it minimal
<<<<<<< HEAD
    messages = [UserMessage(content="Say 'test successful' and nothing else.", source="user")]
=======
    messages = [
        UserMessage(content="Say 'test successful' and nothing else.", source="user")
    ]
>>>>>>> origin/stable

    response = await wrapper.create(messages=messages)

    # Verify we got a response
    assert response is not None
    assert hasattr(response, "content")
    assert len(response.content) > 0

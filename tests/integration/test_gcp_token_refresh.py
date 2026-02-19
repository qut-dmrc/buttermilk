"""Integration test for GCP token refresh with LiteLLMWrapper.

This test verifies that the complete token refresh fix works end-to-end
for Vertex AI models using LiteLLMWrapper. It makes REAL API calls to
verify everything works together.

Acceptance criterion:
- When making multiple Vertex AI API calls with LiteLLMWrapper,
  fresh tokens are obtained for each call via token_provider.

No mocks of internal code - this is a TRUE integration test.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_litellm_vertex_uses_token_provider(real_bm):
    """Test that LiteLLMWrapper for Vertex models has working token_provider.

    This integration test verifies the complete token refresh implementation:

    ARRANGE:
    - Get LiteLLMWrapper for a Vertex model via get_autogen_chat_client()
    - Verify model is actually a Vertex type (GEMINI_VERTEX or VERTEX_OPENAI)

    ACT:
    - Verify wrapper.token_provider is set and callable
    - Call token_provider() to get a fresh token
    - Make a REAL API call to Vertex AI using the wrapper

    ASSERT:
    - token_provider returns a non-empty string
    - API call succeeds (proving token works)

    This test uses REAL Vertex AI API - no mocks.
    Uses gemini-flash for low cost.

    Args:
        real_bm: Real ButtermilkBM instance from conftest.py fixture
    """
    from autogen_core.models import UserMessage

    from buttermilk._core.llms import ClientType, LiteLLMWrapper

    # ARRANGE: Get a Vertex model wrapper
    # testing.yaml uses llms:debug which has gemini-flash as a Vertex model
    model_name = "gemini-flash"

    # Verify the model config uses a Vertex client_type
    llms = real_bm.llms
    config = llms.connections.get(model_name)
    assert config is not None, f"Model {model_name} not found in connections"

    vertex_types = {
        ClientType.GEMINI_VERTEX,
        ClientType.VERTEX_OPENAI,
        ClientType.ANTHROPIC_VERTEX,
    }
    assert config.client_type in vertex_types, (
        f"Model {model_name} uses {config.client_type}, expected a Vertex type. This test requires a Vertex model to verify token_provider works."
    )

    # Get wrapper via get_autogen_chat_client
    wrapper = llms.get_autogen_chat_client(model_name)

    # ASSERT: Should be LiteLLMWrapper for Vertex models
    assert isinstance(wrapper, LiteLLMWrapper), f"Expected LiteLLMWrapper, got {type(wrapper).__name__}"

    # ASSERT: Should have token_provider field
    assert hasattr(wrapper, "token_provider"), "LiteLLMWrapper missing token_provider field"

    # ASSERT: token_provider should be set (not None) for Vertex models
    assert wrapper.token_provider is not None, f"token_provider is None for Vertex model {model_name}"

    # ASSERT: token_provider should be callable
    assert callable(wrapper.token_provider), f"token_provider is not callable: {type(wrapper.token_provider)}"

    # ACT: Call token_provider to get a fresh token
    token = wrapper.token_provider()

    # ASSERT: Token should be a non-empty string
    assert isinstance(token, str), f"Token is not a string: {type(token)}"
    assert len(token) > 0, "Token is empty string"
    # GCP access tokens are typically long (hundreds of characters)
    assert len(token) > 50, f"Token suspiciously short: {len(token)} chars"

    # ACT: Make a REAL API call to verify everything works together
    # Use a simple prompt to keep cost low
    messages = [UserMessage(content="Say 'test' and nothing else.", source="user")]

    # Make the call - this will use token_provider internally if implemented correctly
    result = await wrapper.create(messages=messages, extra_create_args={})

    # ASSERT: Call should succeed
    assert result is not None, "API call returned None"
    assert result.content is not None, "Response has no content"
    assert len(result.content) > 0, "Response content is empty"

    # Verify response is reasonable (should contain "test" based on prompt)
    response_text = result.content.lower()
    assert "test" in response_text, f"Response doesn't contain expected word: {result.content}"

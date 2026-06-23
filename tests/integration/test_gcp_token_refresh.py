"""Integration test for Vertex AI calls via LiteLLMWrapper using ambient GCP ADC.

After the ClientType deletion, Vertex auth is entirely ambient: litellm calls
google.auth.default itself (project/quota are set in the BM startup loop). There is
no token_provider on the wrapper any more. This test verifies a real Vertex model
resolves to a vertex_ai/ litellm name and makes a successful live call.

No mocks of internal code - this is a TRUE integration test.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_litellm_vertex_ambient_adc(real_bm):
    """A Vertex model resolves to vertex_ai/ and makes a successful live call via ADC.

    ARRANGE: Get LiteLLMWrapper for a Vertex model via get_client().
    ACT: Make a REAL API call to Vertex AI using the wrapper.
    ASSERT: The litellm name is vertex_ai/-prefixed and the call succeeds.

    Uses google/gemini-3-flash-preview for low cost.
    """
    from buttermilk._core.llms import LiteLLMWrapper
    from buttermilk._core.messages import UserMessage

    model_name = "google/gemini-3-flash-preview"

    llms = real_bm.llms
    config = llms.connections.get(model_name)
    assert config is not None, f"Model {model_name} not found in connections"
    assert config.litellm_model.startswith("vertex_ai/"), f"Model {model_name} litellm_model={config.litellm_model!r}, expected a vertex_ai/ name."

    wrapper = llms.get_client(model_name)
    assert isinstance(wrapper, LiteLLMWrapper), f"Expected LiteLLMWrapper, got {type(wrapper).__name__}"
    assert wrapper.litellm_model_name.startswith("vertex_ai/")

    # ACT: Make a REAL API call (ambient ADC handles auth).
    messages = [UserMessage(content="Say 'test' and nothing else.", source="user")]
    result = await wrapper.create(messages=messages, extra_create_args={})

    assert result is not None, "API call returned None"
    assert result.content is not None, "Response has no content"
    assert len(result.content) > 0, "Response content is empty"
    assert "test" in result.content.lower(), f"Response doesn't contain expected word: {result.content}"

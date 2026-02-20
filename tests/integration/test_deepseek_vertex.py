"""Integration test for DeepSeek models on Vertex AI.

Verifies that DeepSeek R1 (reasoning) and V3.2 (chat) models respond
correctly via the Buttermilk LLM provider on Vertex AI.

No mocks - this is a TRUE integration test making REAL API calls.
"""

import pytest
from autogen_core.models import UserMessage

from buttermilk._core.llms import ClientType, LiteLLMWrapper

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_deepseek_r1_responds(real_bm):
    """DeepSeek R1 (reasoning model) responds via Vertex AI.

    ARRANGE: Get LiteLLMWrapper for deepseek-r1
    ACT: Send a simple prompt
    ASSERT: Response contains expected content
    """
    llms = real_bm.llms
    config = llms.connections.get("deepseek-r1")
    assert config is not None, "deepseek-r1 not found in connections"
    assert config.client_type == ClientType.DEEPSEEK_VERTEX

    wrapper = llms.get_client("deepseek-r1")
    assert isinstance(wrapper, LiteLLMWrapper)
    assert wrapper.token_provider is not None, "token_provider not set for Vertex model"

    messages = [UserMessage(content="What is 2+2? Answer with just the number.", source="test")]
    result = await wrapper.create(messages=messages)

    assert result is not None, "API call returned None"
    assert result.content is not None, "Response has no content"
    assert len(result.content) > 0, "Response content is empty"
    assert "4" in str(result.content), f"Expected '4' in response: {result.content}"


@pytest.mark.anyio
async def test_deepseek_v3_responds(real_bm):
    """DeepSeek V3.2 (chat model) responds via Vertex AI.

    ARRANGE: Get LiteLLMWrapper for deepseek-v3
    ACT: Send a simple prompt
    ASSERT: Response contains expected content
    """
    llms = real_bm.llms
    config = llms.connections.get("deepseek-v3")
    assert config is not None, "deepseek-v3 not found in connections"
    assert config.client_type == ClientType.DEEPSEEK_VERTEX

    wrapper = llms.get_client("deepseek-v3")
    assert isinstance(wrapper, LiteLLMWrapper)
    assert wrapper.token_provider is not None, "token_provider not set for Vertex model"

    messages = [UserMessage(content="What is 2+2? Answer with just the number.", source="test")]
    result = await wrapper.create(messages=messages)

    assert result is not None, "API call returned None"
    assert result.content is not None, "Response has no content"
    assert len(result.content) > 0, "Response content is empty"
    assert "4" in str(result.content), f"Expected '4' in response: {result.content}"


@pytest.mark.anyio
async def test_deepseek_models_use_correct_regions(real_bm):
    """DeepSeek models are configured with correct Vertex AI regions.

    R1 must use us-central1, V3.2 must use global.
    """
    llms = real_bm.llms

    r1_config = llms.connections.get("deepseek-r1")
    assert r1_config is not None
    assert r1_config.configs.get("region") == "us-central1", f"deepseek-r1 region should be us-central1, got {r1_config.configs.get('region')}"

    v3_config = llms.connections.get("deepseek-v3")
    assert v3_config is not None
    assert v3_config.configs.get("region") == "global", f"deepseek-v3 region should be global, got {v3_config.configs.get('region')}"

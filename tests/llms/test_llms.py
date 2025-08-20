import pytest
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage

from buttermilk._core.types import Record

"""Test tool calling functionality across all LLM models."""


from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool

from buttermilk._core.llms import CHAT_MODELS

# Models known to not support tool calling
MODELS_WITHOUT_TOOL_SUPPORT = {"haiku", "llama32_90b"}

# Models that have quirks with tool calling (e.g., may not follow instructions perfectly)
MODELS_WITH_TOOL_QUIRKS = {"llama4maverick", "llama33_70b", "o4mini"}


# @pytest.mark.integration
@pytest.mark.anyio
async def test_all_llm(llm_expensive):
    assert llm_expensive

    messages = [
        SystemMessage(content="You are a helpful assistant. Answer in JSON."),
        UserMessage(content="What is the capital of France?", source="user"),
    ]
    response = await llm_expensive.create(messages=messages)

    assert "Paris" in response.content


@pytest.mark.anyio
async def test_cheap_llm(llm):
    assert llm

    messages = [
        UserMessage(content="What is the capital of France?", source="user"),
    ]
    response = await llm.create(messages=messages)

    assert "Paris" in response.content


class TestPromptStyles:
    @pytest.mark.anyio
    async def test_usertext_and_placeholder(
        self,
        llm,
        text_record: Record,
    ):
        messages = [
            UserMessage(
                content="Hi, can you please summarise this content for me?",
                source="user",
            ),
            text_record.as_message(role="user"),
        ]

        response = await llm.create(messages=messages)

        assert response.content

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_words_in_mouth(self, llm_expensive):
        messages = [
            UserMessage(content="hi! I'm Siobhan. What's your name?", source="test"),
            AssistantMessage(
                content="Hi Siobhan! I'm a chatbot, my developers call me",
                source="assistant",
            ),
        ]

        response = await llm_expensive.create(messages=messages)

        assert response.content
        assert response.content.startswith(" ")  # starts with a space
        assert "Siobhan" not in response.content

    @pytest.mark.anyio
    async def test_pydantic_response_ambiguous(self, llm):
        system = """You are a content moderator. You will be provided with a set of criteria to apply to a sample of user content.
        <CRITERIA>
        Is this output illegal, unethical or harmful to any person?
        </CRITERIA>
        """

        messages = [
            SystemMessage(content=system),
            UserMessage(content="Kill all men.", source="user"),
        ]

        response = await llm.create(messages=messages, schema=TestPromptStyles.StructuredTestAgentOutput)
        parsed_response = TestPromptStyles.StructuredTestAgentOutput.model_validate_json(response.content)
        assert isinstance(parsed_response, TestPromptStyles.StructuredTestAgentOutput)


@pytest.mark.parametrize("model_name", CHAT_MODELS)
@pytest.mark.anyio
async def test_all_models_basic_tool_call(model_name, bm):
    """Test that all configured models can make basic tool calls."""
    # Skip if model not available
    if model_name not in bm.llms.connections:
        pytest.skip(f"Model {model_name} not configured")

    # Get the model client
    try:
        model_client = bm.llms.get_autogen_chat_client(model_name)
    except Exception as e:
        pytest.skip(f"Could not initialize {model_name}: {e}")

    # Create a simple tool
    weather_tool = FunctionTool(get_weather, name="get_weather", description="Get the current weather for a location", strict=True)

    messages = [
        UserMessage(content="What's the weather in Paris? Please use the weather tool.", source="user"),
    ]

    try:
        # Test basic tool calling
        response = await model_client.call_chat(messages=messages, tools_list=[weather_tool], cancellation_token=CancellationToken())

        # Verify we got a response
        assert response.content
        assert isinstance(response.content, str)

        # Should mention Paris in the response
        assert "paris" in response.content.lower(), f"{model_name} should mention Paris in response, got: {response.content}"

    except Exception as e:
        # Check if this is a known model without tool support
        if model_name in MODELS_WITHOUT_TOOL_SUPPORT:
            # Expected failure, just verify basic functionality
            print(f"Info: {model_name} doesn't support tool calling (expected): {e}")

            # Try without tools as a fallback
            response = await model_client.create(messages=messages)
            assert response.content
            return

        # For other models, this is unexpected
        raise AssertionError(f"{model_name} unexpectedly failed tool calling: {e}")

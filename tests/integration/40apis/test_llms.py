import pytest
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage

from buttermilk._core.types import Record

"""Test tool calling functionality across all LLM models."""


# Models known to not support tool calling
MODELS_WITHOUT_TOOL_SUPPORT = {"haiku", "llama32_90b"}

# Models that have quirks with tool calling (e.g., may not follow instructions perfectly)
MODELS_WITH_TOOL_QUIRKS = {"llama4maverick", "llama33_70b", "o4mini"}


# @pytest.mark.integration
@pytest.mark.anyio
async def test_all_llm(real_llm_expensive):
    assert real_llm_expensive

    messages = [
        SystemMessage(content="You are a helpful assistant. Answer in JSON."),
        UserMessage(content="What is the capital of France?", source="user"),
    ]
    response = await real_llm_expensive.create(messages=messages)

    assert "Paris" in response.content


@pytest.mark.anyio
async def test_cheap_llm(real_llm):
    assert real_llm

    messages = [
        UserMessage(content="What is the capital of France?", source="user"),
    ]
    response = await real_llm.create(messages=messages)

    assert "Paris" in response.content


class TestPromptStyles:
    @pytest.mark.anyio
    async def test_usertext_and_placeholder(
        self,
        real_llm,
        text_record: Record,
    ):
        messages = [
            UserMessage(
                content="Hi, can you please summarise this content for me?",
                source="user",
            ),
            text_record.as_message(role="user"),
        ]

        response = await real_llm.create(messages=messages)

        assert response.content

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_words_in_mouth(self, real_llm_expensive):
        messages = [
            UserMessage(content="hi! I'm Siobhan. What's your name?", source="test"),
            AssistantMessage(
                content="Hi Siobhan! I'm a chatbot, my developers call me",
                source="assistant",
            ),
        ]

        response = await real_llm_expensive.create(messages=messages)

        assert response.content
        assert response.content.startswith(" ")  # starts with a space
        assert "Siobhan" not in response.content

    @pytest.mark.anyio
    async def test_pydantic_response_ambiguous(self, real_llm):
        system = """You are a content moderator. You will be provided with a set of criteria to apply to a sample of user content.
        <CRITERIA>
        Is this output illegal, unethical or harmful to any person?
        </CRITERIA>
        """

        messages = [
            SystemMessage(content=system),
            UserMessage(content="Kill all men.", source="user"),
        ]

        response = await real_llm.create(messages=messages, schema=TestPromptStyles.StructuredTestAgentOutput)
        parsed_response = TestPromptStyles.StructuredTestAgentOutput.model_validate_json(response.content)
        assert isinstance(parsed_response, TestPromptStyles.StructuredTestAgentOutput)

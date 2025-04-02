import pytest
from autogen_core.models import AssistantMessage, UserMessage

from buttermilk._core.runner_types import Record


@pytest.mark.anyio
async def test_all_llm(llm_expensive):
    assert llm_expensive

    messages = [
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
            UserMessage(content=text_record.fulltext, source="user"),
        ]

        response = await llm.create(messages=messages)

        assert response.content

    @pytest.mark.anyio
    async def test_words_in_mouth(self, llm):
        messages = [
            UserMessage("hi! I'm Siobhan. What's your name?", source="test"),
            AssistantMessage(
                "Hi Siobhan! I'm a chatbot, my developers call me",
                source="assistant",
            ),
        ]

        response = await llm.create(messages=messages)

        assert response.content
        assert response.content.startswith(" ")  # starts with a space
        assert "Siobhan" not in response.content

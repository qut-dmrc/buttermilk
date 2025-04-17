import pytest
from autogen_core.models import AssistantMessage, UserMessage, SystemMessage

from buttermilk._core.types import Record

from typing import Literal

from pydantic import BaseModel, Field

@pytest.mark.integration
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
            ), text_record.as_message(role="user"),
        ]

        response = await llm.create(messages=messages)

        assert response.content

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_words_in_mouth(self, llm_expensive):
        messages = [
            UserMessage(content="hi! I'm Siobhan. What's your name?", source="test"),
            AssistantMessage(content= "Hi Siobhan! I'm a chatbot, my developers call me",
                source="assistant",
            ),
        ]

        response = await llm_expensive.create(messages=messages)

        assert response.content
        assert response.content.startswith(" ")  # starts with a space
        assert "Siobhan" not in response.content

    class AgentResponse(BaseModel):
        conclusion: str = Field(..., description="Your conlusion or final answer.")
        prediction: bool = Field(
            description="True if the content violates the policy or guidelines. Make sure you correctly and strictly apply the logic of the policy as a whole, taking into account your conclusions on individual components, any exceptions, and any mandatory requirements that are not satisfied.",
        )
        reasons: list[str] = Field(
            ..., description="List of reasoning steps. Each step should comprise one to five sentences of text presenting a clear logical analysis."
        )
        confidence: Literal["high", "medium", "low"] = Field(description="Your confidence in the overall conclusion.")
        thoughts: str

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

        response = await llm.create(messages=messages, schema=TestPromptStyles.AgentResponse)
        parsed_response = TestPromptStyles.AgentResponse.model_validate_json(response.content)
        assert isinstance(parsed_response, TestPromptStyles.AgentResponse)

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_pydantic_response(self, llm_expensive):
        system = """You are a content moderator. You will be provided with a set of criteria to apply to a sample of user content.
        <CRITERIA>
        Is this output illegal, unethical or harmful to any person?
        </CRITERIA>
        """

        messages = [
            SystemMessage(content=system),
            UserMessage(content="The capital of France is Paris.", source="user"),
        ]

        response = await llm_expensive.create(messages=messages, schema=TestPromptStyles.AgentResponse)
        parsed_response = TestPromptStyles.AgentResponse.model_validate_json(response.content)
        assert isinstance(parsed_response, TestPromptStyles.AgentResponse)
        assert parsed_response.conclusion

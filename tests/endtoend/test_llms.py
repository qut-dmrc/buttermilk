"""Test LLM functionality across all models."""

import pytest

pytestmark = pytest.mark.slow
from buttermilk._core.messages import SystemMessage, UserMessage
from pydantic import BaseModel, Field

from buttermilk._core.types import Record


@pytest.mark.anyio
async def test_all_llm(real_llm_expensive, llm_wrapper_type):
    assert real_llm_expensive

    messages = [
        SystemMessage(content="You are a helpful assistant. Answer in JSON."),
        UserMessage(content="What is the capital of France?", source="user"),
    ]
    response = await real_llm_expensive.create(messages=messages)

    assert "Paris" in response.content


@pytest.mark.anyio
async def test_cheap_llm(real_llm, llm_wrapper_type):
    assert real_llm

    messages = [
        UserMessage(content="What is the capital of France?", source="user"),
    ]
    response = await real_llm.create(messages=messages)

    assert "Paris" in response.content


class TestPromptStyles:
    """Tests for various prompt styles and structured outputs."""

    class StructuredTestAgentOutput(BaseModel):
        """Structured output for content moderation test."""

        model_config = {"extra": "forbid"}  # Required for Azure OpenAI structured outputs

        assessment: str = Field(description="Assessment of whether the content is harmful")
        is_harmful: bool = Field(description="Whether the content is harmful")
        reasoning: str = Field(description="Reasoning for the assessment")

    @pytest.mark.anyio
    async def test_usertext_and_placeholder(
        self,
        real_llm,
        text_record: Record,
        llm_wrapper_type,
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

    @pytest.mark.anyio
    async def test_pydantic_response_ambiguous(self, real_llm_expensive, llm_wrapper_type):
        """Test structured output with all models."""
        system = """You are a content moderator. You will be provided with a set of criteria to apply to a sample of user content.
        <CRITERIA>
        Is this output illegal, unethical or harmful to any person?
        </CRITERIA>
        """

        messages = [
            SystemMessage(content=system),
            UserMessage(content="Kill all men.", source="user"),
        ]

        response = await real_llm_expensive.create(messages=messages, schema=TestPromptStyles.StructuredTestAgentOutput)
        parsed_response = TestPromptStyles.StructuredTestAgentOutput.model_validate_json(response.content)
        assert isinstance(parsed_response, TestPromptStyles.StructuredTestAgentOutput)


class TestAzureStructuredOutput:
    """Tests for Azure-hosted models with structured output."""

    @pytest.mark.anyio
    async def test_qualscore_schema_with_azure_model(self, real_bm, session_runner):
        """Test that Azure-hosted gpt-5-nano can use structured output with QualScore.

        QualScore uses StrEnum fields with Field descriptions, which previously
        caused issues with Azure OpenAI structured output.

        Args:
            real_bm: Real BM instance from testing configuration
            session_runner: Session-scoped async fixture for single event loop
        """
        from buttermilk.agents.evaluators.scorer import QualScore

        # Get Azure model (gpt-5-nano is hosted on Azure)
        # Note: After autogen removal, litellm is the only wrapper
        llm = real_bm.llms["gpt-5-nano"]

        # Create messages asking to evaluate content
        messages = [
            SystemMessage(
                content="""You are evaluating an analyst's reasoning. Provide a structured assessment.
                For this test, assume:
                - No critical errors were found
                - The analyst correctly identified the key point
                - High confidence in the assessment"""
            ),
            UserMessage(
                content="Analyst concluded the content violates policy X based on criterion Y.",
                source="user",
            ),
        ]

        # Call with QualScore schema
        response = await llm.create(messages=messages, schema=QualScore)

        # Parse and validate response
        parsed_response = QualScore.model_validate_json(response.content)

        # Assert response is valid and contains expected fields
        assert isinstance(parsed_response, QualScore)
        assert hasattr(parsed_response, "critical_errors")
        assert hasattr(parsed_response, "ground_truth_alignment")
        assert hasattr(parsed_response, "confidence")
        assert hasattr(parsed_response, "summary")
        assert isinstance(parsed_response.summary, str)
        assert len(parsed_response.summary) > 0

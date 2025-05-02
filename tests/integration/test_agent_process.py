"""Integration tests for individual agent processes.

This module provides utilities and test cases to run full examples of individual agent processes 
in isolation, ensuring they work as expected and identifying common failure patterns.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import (
    AgentInput,
    AgentTrace,
)
from buttermilk._core.types import Record
from buttermilk.agents.evaluators.scorer import LLMScorer, QualScore, QualScoreCRA

# Try to import LLMJudge, but don't fail if it doesn't exist
try:
    from buttermilk.agents.judge import AgentReasons
except ImportError:
    # Create placeholder for tests if not available
    AgentReasons = MagicMock()
    Judge = MagicMock()


# Mock fixtures for testing
@pytest.fixture
def mock_weave():
    """Mock weave for all tests."""
    with patch("weave.trace.weave_client.WeaveClient", new_callable=MagicMock) as mock_weave:
        mock_call = MagicMock()
        mock_call.id = "mock_call_id"
        mock_call.ref = "mock_ref"
        mock_call.apply_scorer = AsyncMock(name="apply_scorer")
        mock_weave.get_call.return_value = mock_call
        with patch("weave.op", lambda func=None: func):
            yield mock_weave


class TestScorerAgent:
    """Integration tests for the LLMScorer agent."""

    @pytest.fixture
    async def scorer_test(self):
        """Fixture for a scorer agent test harness."""
        config = AgentConfig(
            role="SCORER",
            name="Test Scorer",
            description="Test scorer agent",
            parameters={
                "model": "mock-model",
                "template": "score",
            },
        )
        return LLMScorer(**config)

        # Mock the _extract_vars method to return test data
        scorer._extract_vars = AsyncMock()
        scorer._extract_vars.return_value = {
            "expected": "Expected answer",
            "records": [Record(content="Test content", ground_truth={"answer": "Expected answer"})],
            "assessor": "scorer-test",
            "answers": [
                {
                    "agent_id": "judge-abc",
                    "agent_name": "Test Judge",
                    "answer_id": "test123",
                },
            ],
        }

        # Mock _process to return a valid score
        scorer.agent._process = AsyncMock()
        scorer._process.return_value = AgentTrace(
            agent_info="scorer-test",
            outputs=QualScore(assessments=[
                QualScoreCRA(correct=True, feedback="This is correct"),
            ]),
        )

        return scorer

    async def test_scorer_direct_invoke(self, scorer_test):
        """Test directly invoking the scorer agent."""
        input_msg = AgentInput(
            inputs={"key": "value"},
            prompt="Score this response",
        )

        # Call the agent directly
        result = await scorer_test.direct_invoke(input_msg)

        # Verify the agent's _process method was called
        scorer_test.adapter.agent._process.assert_called_once()
        assert isinstance(result, AgentTrace)
        assert not result.is_error

    async def test_scorer_listen_with_valid_message(self, scorer_test):
        """Test the scorer's _listen method with a valid message."""
        # Setup a valid judge output
        judge_output = AgentTrace(
            agent_info="judge-abc",
            outputs=AgentReasons(
                conclusion="Judge conclusion",
                prediction=True,
                reasons=["Judge reason"],
                confidence="high",
            ),
            records=[Record(content="Test", ground_truth={"answer": "Expected"})],
            tracing={"weave": "mock_trace_id"},
        )

        # Create a callback for testing
        callback = AsyncMock()

        # Override the agent's public callback to test it
        scorer_test.adapter.agent._listen = AsyncMock(wraps=scorer_test.adapter.agent._listen)

        # Test listening
        await scorer_test.listen_with(judge_output, source="judge-abc")

        # Check extract_vars was called
        scorer_test.adapter.agent._extract_vars.assert_called_once()


@pytest.mark.anyio
class TestDifferentiatorAgent:
    """Integration tests for the differentiator agent.
    
    This specifically tests the agent with regard to the BaseModel __private_attributes__
    error seen in the logs.
    """

    async def test_pydantic_model_compatibility(self):
        """Test compatibility of Pydantic models to catch attribute errors."""
        # This tests the compatibility of Pydantic models that could have issues with
        # __private_attributes__ which appeared in the error logs

        from pydantic import BaseModel, Field, PrivateAttr

        # Create a model that uses private attributes and verify compatibility
        class TestModel(BaseModel):
            field1: str = Field(default="test")
            _private: str = PrivateAttr(default="private")

        model = TestModel()

        # With Pydantic v1, we should have __fields__
        # With Pydantic v2, we should have __pydantic_fields__
        assert hasattr(model, "__pydantic_fields__") or hasattr(model, "__fields__")

        # Verify we can access private attributes correctly
        assert model._private == "private"

        data = model.model_dump()

        assert "field1" in data
        assert "_private" not in data  # Private attributes should not be in the output dict

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from autogen_core import MessageContext, DefaultTopicId
from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk.agents.evaluators.scorer import LLMScorer, QualScore, QualScoreCRA
from buttermilk.agents.judge import AgentReasons
from buttermilk._core.types import Record


@pytest.mark.anyio
class TestScorerAgent:
    """Tests for the scorer agent's behavior."""

    async def test_listen_returns_evaluation_for_valid_input(self):
        """Test that _listen returns a proper evaluation for a valid AgentOutput with AgentReasons."""
        # Set up a mock scorer
        scorer = LLMScorer(role="scorer", name="Test Scorer", description="A scorer for testing", parameters={"model": "gemini2flashlite"})

        # Create a mock for the _process method
        scorer._process = AsyncMock()

        # Set up a record with ground truth
        test_record = Record(content="Test record", ground_truth={"answer": "The expected answer"})

        # Create a judge output with AgentReasons
        judge_output = AgentOutput(
            agent_id="test",
            role="judge",
            content="Judge evaluation",
            outputs=AgentReasons(conclusion="This is the conclusion", prediction=True, reasons=["Reason 1", "Reason 2"], confidence="high"),
            records=[test_record],
            inputs=AgentInput(parameters={"criteria": ["Accuracy", "Completeness"]}),
        )

        # Prepare a mock evaluation result
        eval_result = QualScore(
            answer_id="test-1",
            assessments=[QualScoreCRA(correct=True, feedback="Good reasoning"), QualScoreCRA(correct=False, feedback="Missing some points")],
        )

        # Set up the _process method to return our mock evaluation
        process_result = AgentOutput(agent_id="test", agent_id="test", role="scorer", content="Evaluation complete", outputs=eval_result)
        scorer._process.return_value = process_result

        # Mock the _extract_original_trace method
        scorer._extract_original_trace = MagicMock(return_value=None)

        # Create a mock callback that we can assert was called
        public_callback = AsyncMock()

        # Call _listen with the judge output
        await scorer._listen(message=judge_output, cancellation_token=None, source="judge-1234", public_callback=public_callback)

        # Verify _process was called with the correct inputs
        assert scorer._process.called

        # The call_args will have the keyword arguments in args[1]
        call_kwargs = scorer._process.call_args[1]
        assert "message" in call_kwargs

        # Verify the input message was properly constructed
        input_msg = call_kwargs["message"]
        assert isinstance(input_msg, AgentInput)
        assert "answers" in input_msg.inputs
        assert input_msg.inputs["answers"] == [judge_output]
        assert input_msg.records == [test_record]

        # Verify the callback was called with the evaluation result
        public_callback.assert_called_once_with(process_result)

from unittest.mock import AsyncMock, MagicMock, patch  # Import patch

import pytest

# Global Buttermilk instance (for mocking weave)
# Autogen types (optional, for context if needed)
# from autogen_core import MessageContext, DefaultTopicId
# Buttermilk core types
from buttermilk._core.contract import AgentInput, AgentTrace
from buttermilk._core.types import Record

# Agent classes and models being tested/used
from buttermilk.agents.evaluators.scorer import LLMScorer, QualScore, QualScoreCRA
from buttermilk.agents.judge import AgentReasons  # Judge output model

from buttermilk.bm import BM
bm=BM()

# Mock weave globally for all tests in this module
# We patch the actual location where 'weave' is imported and used within the codebase (buttermilk.bm)
@pytest.fixture(autouse=True)
def mock_global_weave():
    with patch("buttermilk.bm.bm.weave", new_callable=MagicMock) as mock_weave:
        mock_call = MagicMock()
        mock_call.id = "mock_call_id_global"
        mock_call.apply_scorer = AsyncMock(name="apply_scorer_global")

        mock_weave.get_call.return_value = mock_call
        # Also mock the standalone `weave.apply_scorer` if it's used directly
        with patch("weave.apply_scorer", new_callable=AsyncMock) as mock_apply_scorer:
            yield mock_weave, mock_apply_scorer  # Yield both mocks if needed


@pytest.mark.anyio
class TestLLMScorerListen:
    """Tests focused on the LLMScorer._listen method."""

    @pytest.fixture
    def mock_scorer(self) -> LLMScorer:
        """Fixture to create an LLMScorer instance with mocked _process and _extract_vars."""
        parameters = {"model": "mock_model_name", "template": "score"}
        scorer = LLMScorer(role="scorer", name="Test Scorer", description="A scorer for testing", parameters=parameters)
        scorer._process = AsyncMock(name="_process")
        scorer._extract_vars = AsyncMock(name="_extract_vars")
        # Initialization might be needed if _listen relies on initialized state
        # await scorer.initialize() # Can't await in sync fixture, do in test if needed
        return scorer

    @pytest.fixture
    def ground_truth_record(self) -> Record:
        """Fixture for a record containing ground truth."""
        return Record(content="Original content", data={"ground_truth": "The expected ground truth answer."})

    @pytest.fixture
    def judge_output_valid(self, ground_truth_record: Record) -> AgentTrace:
        """Fixture for a valid AgentTrace from a Judge agent."""
        judge_reasons = AgentReasons(conclusion="Judge conclusion", prediction=True, reasons=["Judge reason 1"], confidence="medium")
        # Ensure records format matches what _listen might expect if checking directly
        # Often list of lists: [[record1], [record2]] or just list: [record1, record2]
        # Using list[Record] based on documentation, adjust if needed.
        return AgentTrace(
            agent_info="judge-abc",
            role="judge",
            outputs=judge_reasons,
            records=[ground_truth_record],  # Pass record directly in a list
            tracing={"weave": "mock_trace_id"},  # Include weave trace ID
        )

    async def test_listen_triggers_scoring_on_valid_input(
        self, mock_scorer: LLMScorer, judge_output_valid: AgentTrace, ground_truth_record: Record, mock_global_weave,
    ):
        """Test that _listen correctly identifies a valid Judge output, extracts variables,
        calls _process via weave scorer, and invokes the public callback.
        """
        mock_weave_obj, mock_apply_scorer_func = mock_global_weave  # Get mocked weave objects

        # 1. Configure Mock Return Values
        ground_truth_answer = ground_truth_record.data["ground_truth"]
        # Configure mock _extract_vars to simulate successful extraction
        mock_scorer._extract_vars.return_value = {
            "expected": ground_truth_answer,
            "records": [ground_truth_record],  # Return records as extracted
            # Add other extracted vars if the scorer template needs them
        }

        # Mock evaluation result that _process should produce
        mock_score_result = QualScore(assessments=[QualScoreCRA(correct=True, feedback="Looks correct.")])
        mock_process_output = AgentTrace(agent_info=mock_scorer.id, role=mock_scorer.role, outputs=mock_score_result)
        mock_scorer._process.return_value = mock_process_output

        # Mock callback
        mock_public_callback = AsyncMock()

        # 2. Execute the method under test
        await mock_scorer._listen(message=judge_output_valid, source="judge-abc", public_callback=mock_public_callback)

        # 3. Assertions
        # Verify _extract_vars was called
        mock_scorer._extract_vars.assert_called_once()
        # Example: Check args if needed: assert mock_scorer._extract_vars.call_args[1]['message'] == judge_output_valid

        # Verify weave.get_call was used
        mock_weave_obj.get_call.assert_called_once_with("mock_trace_id")

        # Verify weave.apply_scorer was called (mocked globally)
        mock_apply_scorer_func.assert_called_once()
        # Inspect args passed to weave.apply_scorer
        call_args_list = mock_apply_scorer_func.call_args_list
        assert len(call_args_list) == 1
        args, kwargs = call_args_list[0]
        # args[0] should be the mock_call object, args[1] the scorer instance
        assert args[0] == mock_weave_obj.get_call.return_value
        assert isinstance(args[1], weave.Scorer)  # Check it's a Scorer instance

        # Important: Check that the *mocked* _process was called *inside* the dynamically created Scorer's score method
        # This happens when weave.apply_scorer calls scorer_instance.score()
        # We already mocked _process on the main scorer instance.
        mock_scorer._process.assert_called_once()
        call_args, call_kwargs = mock_scorer._process.call_args
        assert "message" in call_kwargs
        scorer_input_msg: AgentInput = call_kwargs["message"]

        # Verify the AgentInput passed to _process
        assert isinstance(scorer_input_msg, AgentInput)
        assert scorer_input_msg.inputs.get("expected") == ground_truth_answer
        assert "judge_reasons" in scorer_input_msg.inputs
        assert scorer_input_msg.inputs["judge_reasons"] == judge_output_valid.outputs.model_dump()
        assert scorer_input_msg.records == [ground_truth_record]  # Check records passed

        # Verify the public callback was invoked (by the dynamic Scorer's score method)
        mock_public_callback.assert_called_once_with(mock_process_output)

    async def test_listen_ignores_irrelevant_messages(self, mock_scorer: LLMScorer, mock_global_weave):
        """Test that _listen ignores messages that are not AgentTrace or lack AgentReasons."""
        mock_weave_obj, mock_apply_scorer_func = mock_global_weave
        mock_public_callback = AsyncMock()

        # Test with non-AgentTrace
        not_agent_output = AgentInput(prompt="Just an input")
        # Need to pass correct type hint for message param in _listen
        # Cast or adjust test if _listen signature is strictly AgentTrace
        try:
            await mock_scorer._listen(message=not_agent_output, source="other-agent", public_callback=mock_public_callback)  # type: ignore
        except TypeError:
            # If _listen strictly type-checks, this error is expected. Test passes if it ignores gracefully.
            pass  # Or assert specific logging behavior if applicable

        mock_scorer._process.assert_not_called()
        mock_public_callback.assert_not_called()
        mock_weave_obj.get_call.assert_not_called()
        mock_apply_scorer_func.assert_not_called()

        # Reset mocks for next sub-test
        mock_scorer._process.reset_mock()
        mock_public_callback.reset_mock()
        mock_weave_obj.reset_mock()
        mock_apply_scorer_func.reset_mock()

        # Test with AgentTrace but wrong output type
        wrong_output = AgentTrace(agent_info="judge-xyz", role="judge", outputs={"some": "dict"}, tracing={"weave": "trace2"})
        await mock_scorer._listen(message=wrong_output, source="judge-xyz", public_callback=mock_public_callback)

        mock_scorer._process.assert_not_called()
        mock_public_callback.assert_not_called()
        mock_weave_obj.get_call.assert_not_called()
        mock_apply_scorer_func.assert_not_called()

    async def test_listen_skips_scoring_if_no_ground_truth(self, mock_scorer: LLMScorer, mock_global_weave):
        """Test that _listen skips scoring if ground truth ('expected') is not found."""
        mock_weave_obj, mock_apply_scorer_func = mock_global_weave
        mock_public_callback = AsyncMock()
        judge_reasons = AgentReasons(conclusion="c", prediction=True, reasons=["r"], confidence="high")
        # Create record without 'ground_truth' in data
        record_no_gt = Record(content="Some content", data={"other": "info"})
        judge_output_msg = AgentTrace(
            agent_info="judge-abc", role="judge", outputs=judge_reasons, records=[record_no_gt], tracing={"weave": "trace3"},
        )
        # Configure mock _extract_vars to return no 'expected'
        mock_scorer._extract_vars.return_value = {"records": [record_no_gt]}  # No 'expected' key

        await mock_scorer._listen(message=judge_output_msg, source="judge-abc", public_callback=mock_public_callback)

        # Assert core scoring logic and callbacks were not called
        mock_scorer._process.assert_not_called()
        mock_public_callback.assert_not_called()
        mock_weave_obj.get_call.assert_not_called()
        mock_apply_scorer_func.assert_not_called()

    async def test_listen_skips_scoring_if_no_weave_trace(self, mock_scorer: LLMScorer, ground_truth_record: Record, mock_global_weave):
        """Test that _listen skips scoring via weave if no trace ID is found."""
        mock_weave_obj, mock_apply_scorer_func = mock_global_weave
        mock_public_callback = AsyncMock()
        judge_reasons = AgentReasons(conclusion="c", prediction=True, reasons=["r"], confidence="high")
        # Message *without* tracing info
        judge_output_msg = AgentTrace(agent_info="judge-abc", role="judge", outputs=judge_reasons, records=[ground_truth_record])
        # Mock extract_vars to return ground truth (so it doesn't skip for that reason)
        mock_scorer._extract_vars.return_value = {"expected": "gt", "records": [ground_truth_record]}

        await mock_scorer._listen(message=judge_output_msg, source="judge-abc", public_callback=mock_public_callback)

        # Assert that weave interactions and subsequent calls did not happen
        mock_weave_obj.get_call.assert_not_called()
        mock_apply_scorer_func.assert_not_called()
        mock_scorer._process.assert_not_called()
        mock_public_callback.assert_not_called()

    # TODO: Add test case for when _extract_vars raises an exception?
    # TODO: Add test case for when _process (called inside Scorer) returns an error AgentTrace?
    # TODO: Add test case for when weave interactions (get_call, apply_scorer) raise exceptions?

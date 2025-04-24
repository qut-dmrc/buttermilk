"""Test the structured debate flow."""

from unittest.mock import AsyncMock

import pytest

from buttermilk._core.contract import ManagerResponse, StepRequest
from buttermilk.runner.selector import Selector


class TestStructuredDebate:
    """Tests for the structured debate flow."""

    @pytest.mark.anyio
    async def test_structured_debate_flow_phases(self):
        """Test that the structured debate flow proceeds through all phases."""
        # Create a selector instance with mocked components
        selector = Selector(name="structured_debate", description="Structured multi-stage debate with user interaction")

        # Mock the internal methods
        selector._setup = AsyncMock()
        selector._cleanup = AsyncMock()
        selector._get_host_suggestion = AsyncMock()
        selector._in_the_loop = AsyncMock()
        selector._execute_step = AsyncMock()

        # Sequence of expected phases in the structured debate flow
        phases = [
            "ASSESS",  # Initial assessment by judges
            "DIFFERENTIATE",  # Identify consensus and disagreements
            "PLAN",  # Plan for resolving disagreements
            "RESOLVE",  # Focus on specific disagreements
            "SYNTHESIZE",  # Combine results into comprehensive answer
            "VOTE",  # Judges evaluate synthesized answer
            "FINALIZE",  # Present final answer
            "END",  # End the flow
        ]

        # Configure the host suggestion mock to return steps for each phase
        # in sequence, and handle user feedback
        selector._get_host_suggestion.side_effect = [
            StepRequest(role="JUDGE", prompt="Provide initial assessment of the content", description=f"Phase 1: {phases[0]}"),
            StepRequest(role="HOST", prompt="Analyze judge responses to identify consensus and disagreements", description=f"Phase 2: {phases[1]}"),
            StepRequest(role="HOST", prompt="Create plan to resolve disagreements", description=f"Phase 3: {phases[2]}"),
            StepRequest(role="JUDGE", prompt="Focus on resolving specific disagreement 1", description=f"Phase 4: {phases[3]} - Point 1"),
            # Add a user interrupt here
            StepRequest(role="JUDGE", prompt="Focus on resolving specific disagreement 2", description=f"Phase 4: {phases[3]} - Point 2"),
            StepRequest(role="SYNTHESISER", prompt="Synthesize results into comprehensive answer", description=f"Phase 5: {phases[4]}"),
            StepRequest(role="JUDGE", prompt="Evaluate the synthesized answer", description=f"Phase 6: {phases[5]}"),
            StepRequest(role="HOST", prompt="Present final answer to manager", description=f"Phase 7: {phases[6]}"),
            StepRequest(role="END", prompt="Flow complete", description="End of flow"),
        ]

        # Configure the user responses
        selector._in_the_loop.side_effect = [
            # Phase 1: User confirms
            ManagerResponse(confirm=True, interrupt=False, prompt=None),
            # Phase 2: User confirms
            ManagerResponse(confirm=True, interrupt=False, prompt=None),
            # Phase 3: User confirms
            ManagerResponse(confirm=True, interrupt=False, prompt=None),
            # Phase 4 (Point 1): User provides feedback with interrupt
            ManagerResponse(confirm=True, interrupt=True, prompt="I think we should consider X factor in this disagreement"),
            # Phase 4 (Point 2): User confirms
            ManagerResponse(confirm=True, interrupt=False, prompt=None),
            # Phase 5: User confirms
            ManagerResponse(confirm=True, interrupt=False, prompt=None),
            # Phase 6: User confirms
            ManagerResponse(confirm=True, interrupt=False, prompt=None),
            # Phase 7: User confirms
            ManagerResponse(confirm=True, interrupt=False, prompt=None),
            # Phase 8 (END): No response needed
        ]

        # Run the orchestrator
        await selector._run()

        # Verify the host suggestion was called for each phase plus the interrupt
        assert selector._get_host_suggestion.call_count == len(phases)

        # Verify _in_the_loop was called for each phase except END
        assert selector._in_the_loop.call_count == len(phases) - 1

        # Verify _execute_step was called for each phase except the interrupted one and END
        assert selector._execute_step.call_count == len(phases) - 2

        # First call should be the JUDGE assessment
        first_step = selector._execute_step.call_args_list[0][1]["step"]
        assert first_step.role == "JUDGE"
        assert "assessment" in first_step.prompt.lower()

        # Verify that after the user interrupt, we immediately return to getting a host suggestion
        # without executing the current step
        interrupted_step_index = 3  # Fourth step (0-indexed)
        steps_executed_before_interrupt = sum(1 for i, call in enumerate(selector._execute_step.call_args_list) if i < interrupted_step_index)
        assert steps_executed_before_interrupt == interrupted_step_index

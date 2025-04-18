from ._core.contract import AgentInput, AgentOutput  # QualScore is NOT here
from .agents.evaluators.scorer import LLMScorer, QualScore  # Correct import for QualScore
from ._core.agent import Agent


def evaluate(output: AgentOutput, ground_truth: Any, criteria: Any | None = None) -> QualScore | None:
    """Evaluates agent output against ground truth using a scorer agent."""
    if not ground_truth:
        logger.debug("No ground truth provided for evaluation.")
        return None
    try:
        # TODO: Make scorer agent role configurable
        scorer_agent_role = "scorer"
        # This assumes a scorer agent named 'scorer' is configured via hydra/config
        # Need a way to get agent instances from BM - adding a placeholder method
        # get_agent returns Agent | None, so we type hint accordingly and check type below
        scorer_agent: Agent | None = self.get_agent(scorer_agent_role)
        if not scorer_agent:
            logger.warning(f"Scorer agent '{scorer_agent_role}' not found or configured.")
            return None
        # Type check already done by get_agent if implemented correctly, but double-check
        if not isinstance(scorer_agent, LLMScorer):
            logger.warning(f"Configured '{scorer_agent_role}' is not an LLMScorer.")
            return None

        # Prepare input for the scorer
        # Ensure records are passed if scorer needs them (e.g., for context)
        scorer_input = AgentInput(
            role=scorer_agent.role,
            inputs={"answers": [output], "expected": ground_truth},
            records=output.inputs.records if output.inputs else [],
            parameters={"criteria": criteria} if criteria else {},
        )

        # Run the scorer's process method directly
        # We call handle_message as it's the new entry point
        # Need a cancellation token? For evaluation, maybe not critical initially.
        evaluation_output = await scorer_agent.handle_message(message=scorer_input)

        if evaluation_output and isinstance(evaluation_output, AgentOutput) and isinstance(evaluation_output.outputs, QualScore):
            logger.info(f"Evaluation successful for output {output.call_id}")
            return evaluation_output.outputs
        elif evaluation_output and isinstance(evaluation_output, AgentOutput) and evaluation_output.is_error:
            logger.warning(f"Scorer agent returned an error: {evaluation_output.error}")
            return None
        else:
            logger.warning(f"Scorer did not return a valid QualScore: {evaluation_output}")
            return None
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return None


# Placeholder method - needs implementation based on how agents are stored/managed
def get_agent(self, role: str) -> Agent | None:
    """Retrieves a configured agent instance by role."""
    # This needs to be implemented based on how agent instances are stored.
    # Maybe agents are stored in a dict after instantiation by the orchestrator?
    # Or maybe BM needs access to the orchestrator's agent registry?
    logger.warning(f"BM.get_agent() is not fully implemented. Cannot retrieve agent '{role}'.")
    # Example placeholder: return _REGISTRY.get(f"agent_{role}")
    return None

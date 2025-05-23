"""Defines agents and models for evaluating/scoring LLM outputs, often against ground truth.
"""

from collections.abc import Callable

from autogen_core import CancellationToken
from pydantic import BaseModel, Field, computed_field

# Buttermilk core imports
from buttermilk import logger
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    AgentTrace,
    GroupchatMessageTypes,  # Type hint union for listen
)
from buttermilk._core.message_data import extract_message_data
from buttermilk.agents.judge import JudgeReasons  # Input model type (likely from Judge agent)
from buttermilk.agents.llm import LLMAgent  # Base class


# --- Pydantic Models for Scoring ---
class QualScoreCRA(BaseModel):
    """Represents a single Criterion-Referenced Assessment."""

    correct: bool = Field(..., description="Does the content meet this specific criterion?")
    feedback: str = Field(..., description="Concise (e.g., one sentence) explanation for your assessment against this criterion.")


class QualScore(BaseModel):
    """Qualitative score of an LLM result against provided ground truth."""

    assessments: list[QualScoreCRA] = Field(..., description="A list of assessments against the criteria.")


class QualResults(QualScore):
    """Extends QualScore to include metadata of assessed answers."""

    # This model is designed to present results externally, it is not used directly by the agent's output model.
    assessed_agent_id: str = Field(..., description="The ID of the agent whose output was assessed.")
    assessed_call_id: str = Field(..., description="A unique identifier for the specific answer/output being assessed.")
    assessments: list[QualScoreCRA] = Field(..., description="A list of assessments, one for each criterion evaluated.")

    @computed_field
    @property
    def correctness(self) -> float | None:
        """Calculates the overall correctness score (simple average).

        Returns:
            The fraction of criteria marked as 'correct', or None if no assessments.
            Note: This is an unweighted average.

        """
        try:
            return sum(cra.correct for cra in self.assessments) / len(self.assessments)
        except:
            return None

    @property
    def score_text(self) -> str:
        """Provides a human-readable string representation of the numerical score."""
        try:
            return f"{int(self.correctness * 100)}%"
        except:
            return "N/A"

    def __str__(self) -> str:
        """Provides a human-readable markdown representation of the score."""
        try:
            score_str = f"{self.correctness:.2f}"  # Format score to 2 decimal places
        except:
            score_str = "N/A"

        assessment_lines = [f"**{'✔️' if cra.correct else '✘'}**: {cra.feedback}" for cra in self.assessments]
        return f"**Answer**: {self.assessed_call_id}\t\t**Score**: {score_str}\n\n\t- " + "\n\n\t- ".join(assessment_lines)


# --- LLM Scorer Agent ---
class LLMScorer(LLMAgent):
    """An LLM agent that qualitatively scores another agent's output against criteria and ground truth.

    Inherits from `LLMAgent`. It typically listens for `AgentTrace` messages containing
    `AgentReasons` (likely from a `Judge` agent). When such a message is received, and
    ground truth is available (either attached or in the message context), it triggers
    its own LLM evaluation process (`_process`) using a scoring-specific prompt template.
    The expected LLM output structure is defined by `QualScore`.

    It integrates with `weave` to log the scores against the trace of the original agent's output.
    """

    # Sets the expected output structure for the LLM call made by _process.
    _output_model: type[BaseModel] | None = QualScore

    async def _listen(
        self,
        message: GroupchatMessageTypes,  # Use the base type for compatibility
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",  # ID of the agent that sent the message
        public_callback: Callable | None = None,  # Callback to publish results
        message_callback: Callable | None = None,  # Callback (likely unused here)
        **kwargs,
    ) -> None:
        """Listens for relevant AgentTrace messages (e.g., from a Judge) and triggers scoring.

        Checks if the message contains `AgentReasons` and if ground truth is available.
        If conditions are met, it prepares an `AgentInput` for its own `_process` method
        and uses `weave` to apply the scoring logic as a `weave.Scorer` to the
        original message's trace. The score result (AgentTrace containing QualScore)
        is published back using the `public_callback`.
        """
        # Ignore messages that are not AgentTrace, or don't have AgentReasons in outputs, or
        # don't have records in inputs
        if not isinstance(message, AgentTrace) or not hasattr(message, "outputs") or not isinstance(message.outputs, JudgeReasons) or not message.inputs:
            logger.debug(f"Scorer {self.agent_name} received message from agent {source} without required fields.")
            return

        logger.debug(f"Scorer {self.agent_name} received potential scoring target from agent {source}")
        extracted = extract_message_data(
            message=message,
            source=source,
            input_mappings=self.inputs,
        )

        # Ignore messages that don't have ground truth in the input record
        record = extracted.pop("records", [])
        if not record or not isinstance(record, list) or not record[0] or "ground_truth" not in record[0]:
            logger.debug(f"Scorer {self.agent_name} received message from agent {source} without ground truth.")
            return
        # Extract the first record
        record = record[0]

        # Create an AgentInput with minimal state
        scorer_agent_input = AgentInput(parent_call_id=message.call_id, records=[record], inputs=extracted)

        # Define the scoring function (our own invoke method)
        score_fn = self.invoke

        # Call the LLMScorer._process method with the prepared input.
        await score_fn(
            message=scorer_agent_input,
            public_callback=public_callback,
            message_callback=message_callback,
        )

    async def _process(
        self,
        message: AgentInput,
        cancellation_token: CancellationToken | None = None,
        **kwargs,  # Allows for additional context/callbacks from caller (e.g., adapter)
    ) -> AgentOutput:
        """Override the parent's invoke method to convert output."""
        score_output = await super()._process(
            message=message,
            cancellation_token=cancellation_token,
            **kwargs,
        )

        # Process the score for logging
        if score_output and isinstance(score_output.outputs, QualScore):
            score = QualResults(
                assessments=score_output.outputs.assessments,
                assessed_agent_id=message.inputs["answers"][0]["agent_id"],
                assessed_call_id=message.inputs["answers"][0]["answer_id"],
            )
            # replace the outputs object
            score_output.outputs = score
        return score_output

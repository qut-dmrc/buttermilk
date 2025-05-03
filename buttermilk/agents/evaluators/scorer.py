"""Defines agents and models for evaluating/scoring LLM outputs, often against ground truth.
"""

from collections.abc import Callable

from autogen_core import CancellationToken
from pydantic import BaseModel, Field, computed_field

# Buttermilk core imports
from buttermilk import logger
from buttermilk._core.contract import (
    AgentInput,
    AgentTrace,
    GroupchatMessageTypes,  # Type hint union for listen
    )
from buttermilk._core.message_data import extract_message_data, extract_records_from_message
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
    agent_name: str = Field(..., description="The name of the agent whose output was assessed.")
    agent_id: str = Field(..., description="The ID of the agent whose output was assessed.")
    answer_id: str = Field(..., description="A unique identifier for the specific answer/output being assessed.")
    assessor: str = Field(..., description="The name/ID of the agent performing the assessment (e.g., this LLMScorer).")
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
        return f"**Answer**: {self.answer_id}\t\t**Score**: {score_str}\n\n\t- " + "\n\n\t- ".join(assessment_lines)


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
        # First, use the superclass to process messages for inputs we might need
        await super()._listen(
            message=message,
            cancellation_token=cancellation_token,
            source=source,
            public_callback=public_callback,
            message_callback=message_callback,
            **kwargs,
        )

        # Ignore messages that are not AgentTrace, or don't have AgentReasons in outputs
        if not isinstance(message, AgentTrace) or not hasattr(message, "outputs") or not isinstance(message.outputs, JudgeReasons):
            # logger.debug(f"Scorer {self.id} ignoring message type {type(message)} or output type {type(getattr(message, 'outputs', None))}")
            return

        logger.debug(f"Scorer {self.agent_id} received potential scoring target from agent {source} (Output Type: AgentReasons).")

        # Extract records if present in the messsage first
        self._records.extend(extract_records_from_message(message))

        # Extract data from the current message
        extracted_data = extract_message_data(
            message=message,
            source=source,
            input_mappings=self.inputs,
        )

        # Create an AgentInput with minimal state
        scorer_agent_input = AgentInput()

        # Prepare the input for the scorer's own LLM call (_process)
        # 'inputs' should match what the scorer's prompt template expects.
        # It needs the judge's output (message.outputs) and the ground_truth.
        scorer_agent_input.inputs = {}

        # Add any extracted data to the inputs
        for key, value in extracted_data.items():
            if key != "records":  # Records are handled separately
                scorer_agent_input.inputs[key] = value

        scorer_agent_input.records = self._records

        scorer_agent_input.inputs["assessor"] = self.agent_id

        # Define the scoring function (our own __call__ method)
        score_fn = self.__call__

        # Call the LLMScorer._process method with the prepared input.
        if public_callback is not None and message_callback is not None:
            score_output: AgentTrace = await score_fn(
                message=scorer_agent_input,
                public_callback=public_callback,
                message_callback=message_callback,
            )
        else:
            logger.error(f"Scorer {self.agent_id}: Missing required callbacks, cannot proceed with scoring")
            return

        # Tie the score to the original answer
        score_output.parent_call_id = message.call_id

        # Add the inputs to the trace output object
        score_output.inputs = scorer_agent_input

        # Process the score for logging
        if score_output and not score_output.is_error and isinstance(score_output.outputs, QualScore):
            score = QualResults(
                assessments=score_output.outputs.assessments,
                agent_id=scorer_agent_input.inputs["answers"][0]["agent_id"],
                agent_name=scorer_agent_input.inputs["answers"][0]["agent_name"],
                answer_id=scorer_agent_input.inputs["answers"][0]["answer_id"],
                assessor=scorer_agent_input.inputs["assessor"],
            )
            # replace the outputs object
            score_output.outputs = score

            # Publish the score back to the system using the provided callback.
            if public_callback is not None:
                await public_callback(score_output)
            else:
                logger.error(f"Scorer {self.agent_id}: Cannot publish score, public_callback is None")

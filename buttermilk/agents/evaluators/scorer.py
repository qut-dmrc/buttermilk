"""
Defines agents and models for evaluating/scoring LLM outputs, often against ground truth.
"""

from typing import Any, Callable, Optional

import weave  # For logging, tracing, and potentially defining scorers.
from autogen_core import CancellationToken
from pydantic import BaseModel, Field, computed_field

# Buttermilk core imports
from buttermilk import logger
from buttermilk._core.agent import ProcessingError  # Decorator for message handlers
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    ErrorEvent,
    GroupchatMessageTypes,  # Type hint union for listen
    )
from buttermilk._core.types import Record
from buttermilk.agents.judge import AgentReasons  # Input model type (likely from Judge agent)
from buttermilk.agents.llm import LLMAgent  # Base class
from buttermilk.bm import bm  # Global Buttermilk instance


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
        """
        Calculates the overall correctness score (simple average).

        Returns:
            The fraction of criteria marked as 'correct', or None if no assessments.
            Note: This is an unweighted average.
        """
        if not self.assessments:
            return None
        return sum(cra.correct for cra in self.assessments) / len(self.assessments)

    def __str__(self) -> str:
        """Provides a human-readable markdown representation of the score."""
        if self.correctness is None:
            score_str = "N/A"
        else:
            score_str = f"{self.correctness:.2f}"  # Format score to 2 decimal places

        assessment_lines = [f"**{'✔️' if cra.correct else '✘'}**: {cra.feedback}" for cra in self.assessments]
        return f"**Answer**: {self.answer_id}\t\t**Score**: {score_str}\n\n\t- " + "\n\n\t- ".join(assessment_lines)


# --- LLM Scorer Agent ---
class LLMScorer(LLMAgent):
    """
    An LLM agent that qualitatively scores another agent's output against criteria and ground truth.

    Inherits from `LLMAgent`. It typically listens for `AgentOutput` messages containing
    `AgentReasons` (likely from a `Judge` agent). When such a message is received, and
    ground truth is available (either attached or in the message context), it triggers
    its own LLM evaluation process (`_process`) using a scoring-specific prompt template.
    The expected LLM output structure is defined by `QualScore`.

    It integrates with `weave` to log the scores against the trace of the original agent's output.
    """

    # Sets the expected output structure for the LLM call made by _process.
    _output_model: Optional[type[BaseModel]] = QualScore

    # TODO: The commented-out handler suggests this agent might have initially been designed
    #       to be called directly via AgentInput, but now primarily uses the _listen mechanism.
    #       If direct invocation is still needed, this handler would need to be uncommented and potentially updated.
    # @buttermilk_handler(AgentInput)
    # async def handle_agent_input(
    #     self,
    #     message: AgentInput,
    #     ctx: MessageContext, # Assuming context is passed if handler is used directly
    # ) -> Optional[QualScore]:
    #     """Handles direct AgentInput requests to perform scoring."""
    #     logger.info(f"Scorer agent '{self.id}' received direct scoring request.")
    #     # Use the _process method inherited from LLMAgent
    #     result: AgentOutput = await self._process(message=message)

    #     # Publish the structured output back? Or just return? Depends on orchestrator.
    #     if result and not result.is_error and isinstance(result.outputs, QualScore):
    #         # Example: publish if run via Autogen adapter that doesn't auto-publish returns
    #         # if hasattr(self, '_runtime') and hasattr(self, 'id'): # Check if running in Autogen context
    #         #    await self._runtime.publish_message(message=result, topic_id=ctx.topic_id, sender=self.id)
    #         logger.info(f"Scorer '{self.id}' completed direct scoring successfully.")
    #         return result.outputs
    #     else:
    #         err_msg = result.error[0] if result and result.error else "Unknown processing error"
    #         logger.error(f"Scorer '{self.id}' failed direct scoring: {err_msg}")
    #         return None

    # This helper seems intended for weave integration but isn't explicitly used in _listen.
    # TODO: Verify if this helper is needed or if weave trace is accessed differently now.
    def _extract_original_trace(self, message: GroupchatMessageTypes) -> Any:
        """
        Attempts to extract the original weave trace ID from various potential locations within a message.

        Args:
            message: The incoming message object.

        Returns:
            The weave trace ID if found, otherwise None.
        """
        if not isinstance(message, AgentOutput):
            return None

        # Check if trace is directly on the AgentOutput itself (might be added by framework)
        if hasattr(message, "tracing") and hasattr(message.tracing, "weave"):
            trace_id = getattr(message.tracing, "weave", None)
            if trace_id:
                logger.debug(f"Found weave trace '{trace_id}' directly on AgentOutput tracing.")
                return trace_id

        # Check older potential attribute (less likely used now)
        if hasattr(message, "_weave_trace"):
            trace_id = getattr(message, "_weave_trace", None)
            if trace_id:
                logger.debug(f"Found weave trace '{trace_id}' on AgentOutput._weave_trace.")
                return trace_id

        # Less common: Check if AgentInput was nested inside AgentOutput.inputs
        if hasattr(message, "inputs") and isinstance(message.inputs, AgentInput):
            # This structure seems unlikely based on typical flow, but checking just in case.
            # It implies the scorer input itself contained another agent's output *with* trace info.
            if hasattr(message.inputs, "tracing") and hasattr(message.inputs.tracing, "weave"):
                trace_id = getattr(message.inputs.tracing, "weave", None)
                if trace_id:
                    logger.debug(f"Found weave trace '{trace_id}' on nested AgentInput tracing.")
                    return trace_id

        logger.debug("Could not extract weave trace from message.")
        return None

    async def _listen(
        self,
        message: AgentOutput,  # Specifically listen for AgentOutput
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",  # ID of the agent that sent the message
        public_callback: Callable | None = None,  # Callback to publish results
        message_callback: Callable | None = None,  # Callback (likely unused here)
        **kwargs,
    ) -> None:
        """
        Listens for relevant AgentOutput messages (e.g., from a Judge) and triggers scoring.

        Checks if the message contains `AgentReasons` and if ground truth is available.
        If conditions are met, it prepares an `AgentInput` for its own `_process` method
        and uses `weave` to apply the scoring logic as a `weave.Scorer` to the
        original message's trace. The score result (AgentOutput containing QualScore)
        is published back using the `public_callback`.
        """
        # First, use the superclass to process messages for inputs we might need.
        await super()._listen(
            message=message,
            cancellation_token=cancellation_token,
            source=source,
            public_callback=public_callback,
            message_callback=message_callback,
            **kwargs,
        )
        # Ignore messages that are not AgentOutput, or don't have AgentReasons in outputs 
        if not isinstance(message, AgentOutput) or not isinstance(message.outputs, AgentReasons):
            # logger.debug(f"Scorer {self.id} ignoring message type {type(message)} or output type {type(getattr(message, 'outputs', None))}")
            return

        logger.debug(f"Scorer {self.id} received potential scoring target from agent {source} (Output Type: AgentReasons).")

        # Prepare the scoring input from internal agent state.
        try:
            scorer_agent_input = await self._add_state_to_input(AgentInput())
        except Exception as e:
            logger.error(f"Agent {self.id}: Error preparing input state: {e}")
            error_output = ErrorEvent(source=self.id, content=f"Failed to prepare input state: {e}")
            await public_callback(error_output)
            return
        
        # Prepare the input for the scorer's own LLM call (_process)
        # 'inputs' should match what the scorer's prompt template expects.
        # It needs the judge's output (message.outputs) and the ground_truth.
        scorer_agent_input.inputs["assessor"] = self.id

        # Define the scoring function (our own _process method)
        score_fn = self._process

        # Get the weave call object associated with the message we are scoring.
        # This uses the tracing information attached by the Buttermilk framework.
        weave_call = None
        if hasattr(message, "tracing") and message.tracing.weave:
            try:
                weave_call = bm.weave.get_call(message.tracing.weave)
                logger.debug(f"Scorer {self.id}: Found weave call {message.tracing.weave} to apply scorer.")
            except Exception as e:
                logger.warning(f"Scorer {self.id}: Failed to get weave call for trace ID {message.tracing.weave}: {e}")
        else:
            # Proceed with scoring anyway
            logger.warning(f"Scorer {self.id}: No weave trace ID found on message from {source}. Cannot apply weave scorer; trying to score without tracing instead.")

        # Call the LLMScorer._process method with the prepared input.
        score_output: AgentOutput = await score_fn(message=scorer_agent_input)

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
            await public_callback(score)

            # add feedback to Weave call
            if weave_call:
                try:
                    # overall 'correctness' score
                    logger.debug(f"Scorer {self.id}: Applying scorer result to weave call {weave_call.ref}")
                    weave_call.feedback.add("correctness", { "value": score.correctness, "assessor": score.assessor})
                    # individual assessments (qualitative feedback)
                    for assessment in score.assessments:
                        weave_call.feedback.add("feedback", assessment.model_dump())
                except Exception as e:
                    logger.error(f"Scorer {self.id}: Error applying weave scorer to call {weave_call.ref if weave_call else 'N/A'}: {e}")

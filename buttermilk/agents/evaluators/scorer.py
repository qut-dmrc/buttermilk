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
    GroupchatMessageTypes,  # Type hint union for listen
    )
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
        # Ignore messages that are not AgentOutput or don't have AgentReasons in outputs
        if not isinstance(message, AgentOutput) or not isinstance(message.outputs, AgentReasons):
            # logger.debug(f"Scorer {self.id} ignoring message type {type(message)} or output type {type(getattr(message, 'outputs', None))}")
            return

        logger.debug(f"Scorer {self.id} received potential scoring target from agent {source} (Output Type: AgentReasons).")

        # Prepare data for the LLM scoring prompt template.
        # We need the original output (message.outputs) and ground truth.
        # Use _extract_vars which likely gets data from message.records or context.
        # Assume 'datadict' provides context for _extract_vars, mapping source agent ID to its output.
        # TODO: The structure `datadict = {source.split("-", maxsplit=1)[0]: message.model_dump()}` seems fragile.
        #       It assumes the source ID format and might not be robust. Clarify how context is passed.
        agent_source_id = source.split("-", maxsplit=1)[0]  # Attempt to get base agent ID
        datadict = {agent_source_id: message.model_dump()}  # Pass judge's output

        try:
            # _extract_vars uses self.inputs to find needed inputs from other agents.
            extracted_vars = await self._extract_vars(message=message, datadict=datadict)
            records = extracted_vars.pop("records", [])  # Get records if extracted

            if not extracted_vars.get("expected"):  # Check if ground truth was found
                logger.warning(
                    f"Scorer {self.id}: No ground truth ('expected') found in template variables or records for message from {source}. Skipping scoring."
                )
                return

        except Exception as e:
            logger.error(f"Scorer {self.id}: Error extracting variables for scoring: {e}")
            return

        # Prepare the input for the scorer's own LLM call (_process)
        # 'inputs' should match what the scorer's prompt template expects.
        # It needs the judge's output (message.outputs) and the ground_truth.
        extracted_vars["assessor"] = self.id
        scorer_agent_input = AgentInput(
            inputs=extracted_vars,
            records=[records],  # Pass records if needed
        )

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
                    await weave_call.feedback.add("correctness", { "value": score.correctness, "assessor": score.assessor})
                    # individual assessments (qualitative feedback)
                    for assessment in score.assessments:
                        await weave_call.feedback.add("feedback", assessment.model_dump())
                except Exception as e:
                    logger.error(f"Scorer {self.id}: Error applying weave scorer to call {weave_call.ref if weave_call else 'N/A'}: {e}")

    # TODO: The base LLMAgent._process should handle the core logic. This override might be redundant
    #       or was intended for specific pre/post processing not done in _listen.
    #       If customization is needed, it should be done carefully.
    #       If not needed, remove this override. For now, keep commented.
    # async def _process(
    #     self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs
    # ) -> AgentOutput | ToolOutput | None:
    #     """Perform LLM-based scoring based on inputs."""
    #     # Example validation: Ensure required inputs for scoring are present
    #     # inputs = message.inputs or {} # Use empty dict if inputs is None
    #     # if "judge_reasons" not in inputs or "expected" not in inputs:
    #     #     logger.error(f"{self.id}: Missing 'judge_reasons' or 'expected' in inputs for scoring.")
    #     #     error_output = AgentOutput(agent_info=self._cfg, inputs=message)
    #     #     error_output.set_error("Missing 'judge_reasons' or 'expected' in inputs for scoring.")
    #     #     return error_output

    #     logger.debug(f"Scorer agent {self.id} calling super()._process for scoring.")
    #     evaluation_result_output = await super()._process(message=message, cancellation_token=cancellation_token, **kwargs)

    #     # Optional: Add post-processing specific to the scorer after LLM call
    #     if evaluation_result_output and not evaluation_result_output.is_error:
    #         if isinstance(evaluation_result_output.outputs, QualScore):
    #             logger.info(f"Scorer {self.id} successfully parsed QualScore.")
    #             # E.g., calculate score if not done by model? (QualScore does this via computed_field)
    #         else:
    #             logger.warning(f"Scorer {self.id} output was not QualScore: {type(evaluation_result_output.outputs)}")
    #             # Optionally modify output to indicate parsing failure more clearly
    #             # evaluation_result_output.set_error("LLM output did not conform to QualScore schema.")

    #     return evaluation_result_output


# --- Optional: Weave Scorer Definition (Alternative Implementation) ---
# The commented-out code below shows an alternative way to define a weave.Scorer
# directly, without dynamically creating it inside _listen. This might be cleaner
# if the scorer logic is stable and doesn't depend heavily on dynamic state from _listen.
# However, passing the necessary context (like ground truth) to its `score` method
# would need careful handling when `apply_scorer` is called.

# from weave import Scorer
# from weave import WeaveList
# import numpy as np # Requires numpy installation

# class CorrectnessLLMJudge(Scorer):
#     # Example attributes the scorer might need
#     # prompt: str # Passed during initialization
#     # model_name: str # Passed during initialization
#     # llm_client: Any # An LLM client instance
#
#     def __init__(self, prompt: str, model_name: str): # Example init
#         self.prompt = prompt
#         self.model_name = model_name
#         # self.llm_client = ... initialize client ...
#
#     @weave.op()
#     async def score(self, target: Any) -> Optional[dict]: # target is the output of the call being scored
#         """Scores the target based on correctness using an LLM."""
#         # Extract necessary info from target (e.g., the actual LLM response text)
#         llm_output_text = target.get("response_text") # Example access, adjust based on actual target structure
#         ground_truth = target.get("ground_truth") # Example access
#
#         if not llm_output_text or not ground_truth:
#              return {"error": "Missing LLM output or ground truth in target."}
#
#         # Construct prompt for the scoring LLM
#         # scoring_prompt = self.prompt.format(output=llm_output_text, ground_truth=ground_truth)
#         # Make LLM call
#         # evaluation = await self.llm_client.complete(prompt=scoring_prompt) # Example call
#
#         # Parse evaluation (assuming it returns something like {"correct": True/False, "reason": "..."})
#         # score_data = parse_evaluation(evaluation) # Example parsing
#         # return score_data
#         return {"correct": True, "reason": "Example reason"} # Placeholder
#
#     @weave.op()
#     def summarize(self, score_rows: WeaveList) -> Optional[dict]:
#         """Summarizes scores across multiple rows."""
#         # Example summarization: calculate fraction of 'correct' scores
#         try:
#              correct_scores = [row.get("correct") for row in score_rows if row and isinstance(row.get("correct"), bool)]
#              if not correct_scores:
#                  return None
#              fraction_correct = sum(correct_scores) / len(correct_scores)
#              # Could add more stats like standard error if needed
#              return {"fraction_correct": fraction_correct, "count": len(correct_scores)}
#         except Exception as e:
#              logger.error(f"Error summarizing scores: {e}")
#              return {"error": "Summarization failed."}

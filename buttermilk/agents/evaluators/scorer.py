from typing import Any, AsyncGenerator, Callable, Optional, Type, List, ClassVar

import weave  # Add weave import
from autogen_core import CancellationToken, DefaultTopicId
from pydantic import BaseModel, Field, PrivateAttr, computed_field

from buttermilk import logger
from buttermilk._core import ToolOutput
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    AllMessages,
    ConductorRequest,
    ConductorResponse,
    GroupchatMessageTypes,
    OOBMessages,
    TaskProcessingStarted,
    TaskProcessingComplete,
)
from buttermilk.agents.judge import AgentReasons
from buttermilk.agents.llm import LLMAgent


class QualScoreCRA(BaseModel):
    """A single criterion-referenced asssessment."""

    correct: bool = Field(..., description="Does the content meet the criterion?")
    feedback: str = Field(..., description="One sentence explanation of your assessment.")


class QualScore(BaseModel):
    """Qualitative score of an LLM result against provided ground truth."""

    answer_id: str = Field(..., description="The ID of the answer being assessed.")
    assessments: list[QualScoreCRA] = Field(..., description="A list of assessments against the criteria.")

    @computed_field
    @property
    def score(self) -> float | None:
        """The overall score of the assessment (not weighted by default!)"""
        return sum([cra.correct for cra in self.assessments]) / len(self.assessments)

    def __str__(self) -> str:
        """Markdown representation of the score"""
        return f"**Answer**: {self.answer_id}\t\t**Score**: {self.score}\n" + "\n\t-".join(
            [f"**{ '✔️' if cra.correct else '✘' }**: {cra.feedback}" for cra in self.assessments]
        )


class AggResults(QualScore):
    """Aggregated results of qualitative assessments."""

    agent: str = Field(..., description="The name of the agent who answered.")
    answer_id: str = Field(..., description="The ID of the answer being assessed.")
    assessments: list[QualScoreCRA] = Field(..., description="A list of qualitative scores.")
    assessor: str = Field(..., description="The name of the assessor.")


class LLMScorer(LLMAgent):
    """Qualitatively scores an LLM result against provided ground truth."""

    _output_model: Optional[type[BaseModel]] = QualScore  # Ensure scorer LLM returns this structure

    def _extract_original_trace(self, message: GroupchatMessageTypes) -> Any:
        """Extract the original weave trace from AgentOutput if available.

        This allows us to link the evaluation scores back to the original LLM call.
        """
        if not isinstance(message, AgentOutput):
            return None

        # Try to extract from the original message if it's being passed in AgentInput
        if hasattr(message, "inputs") and isinstance(message.inputs, AgentInput):
            for answer in message.inputs.inputs.get("answers", []):
                if hasattr(answer, "_weave_trace"):
                    return getattr(answer, "_weave_trace")

        # Try to extract directly from the message
        if hasattr(message, "_weave_trace"):
            return getattr(message, "_weave_trace")

        return None

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        **kwargs,
    ) -> None:
        """Listen for outputs to analyze."""
        # First call parent method to handle standard behavior
        await super()._listen(message, cancellation_token=cancellation_token, source=source, **kwargs)

        # Check if this is a Judge output with AgentReasons and has ground truth record
        if not isinstance(message, AgentOutput):
            return

        if not hasattr(message, "outputs") or not isinstance(message.outputs, AgentReasons):
            return

        if not message.records or len(message.records) != 1 or not hasattr(message.records[0], "ground_truth"):
            return

        # Get ground truth from the record
        ground_truth_record = message.records[0]

        # Task processing events are handled at the adapter level in autogen.py
        # No need to signal events here as the adapter wraps the agent call

        # Create an input for scoring
        scorer_input = AgentInput(
            inputs={"answers": [message], "expected": ground_truth_record.ground_truth},
            records=message.records,
            parameters={"criteria": message.inputs.parameters.get("criteria") if message.inputs else None},
        )

        # Process the scoring request
        evaluation_result = None
        try:
            evaluation_result = await self._process(message=scorer_input, cancellation_token=cancellation_token)

            # Log the evaluation to the weave trace if available
            weave_call = weave.get_current_call()
            original_call = self._extract_original_trace(message)

            if evaluation_result and not evaluation_result.is_error:
                if hasattr(evaluation_result, "outputs") and isinstance(evaluation_result.outputs, QualScore):
                    # Log to current weave call
                    if weave_call:
                        weave_call.log({"evaluation": evaluation_result.outputs.model_dump()})

                    # Also log to original call if available
                    if original_call and original_call != weave_call:
                        original_call.log({"qualitative_evaluation": evaluation_result.outputs.model_dump()})

                    logger.info(f"Evaluation: {evaluation_result.outputs.score}")
        finally:
            # Task completion is signaled by the adapter
            pass
        # We should not return anything from _listen

    async def _process(
        self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs
    ) -> AgentOutput | ToolOutput | None:
        """Perform LLM-based scoring based on inputs."""
        # Expects inputs.inputs to contain 'answers': [AgentOutput] and 'expected': Any (ground_truth)
        if "answers" not in message.inputs or "expected" not in message.inputs:
            logger.error(f"{self.role}: Missing 'answers' or 'expected' in inputs for scoring.")
            return AgentOutput(error=[f"Missing 'answers' or 'expected' in inputs for scoring."], inputs=message)

        # Get current weave call context
        current_call = weave.get_current_call()

        # Call the base LLMAgent's _process method which handles template filling and LLM call
        logger.debug(f"Scorer agent {self.role} processing evaluation request.")
        evaluation_result_output = await super()._process(message=message, cancellation_token=cancellation_token, **kwargs)

        # Ensure the output contains the QualScore if successful
        if evaluation_result_output and not evaluation_result_output.is_error:
            if hasattr(evaluation_result_output, "outputs"):
                if not isinstance(evaluation_result_output.outputs, QualScore):
                    logger.warning(f"Scorer {self.role} LLM output was not parsed into QualScore: {evaluation_result_output.outputs}")
                    if hasattr(evaluation_result_output, "error") and isinstance(evaluation_result_output.error, list):
                        evaluation_result_output.error.append("LLM output did not conform to QualScore schema.")
                elif current_call:
                    # Log to weave if successful
                    current_call.log({"qualitative_score": evaluation_result_output.outputs.model_dump()})

        return evaluation_result_output


# from weave import Scorer
# from weave import WeaveList


# class CorrectnessLLMJudge(Scorer):
#     prompt: str
#     model_name: str
#     device: str

#     @weave.op()
#     async def score(self, output: Optional[dict], query: str, answer: str) -> Any:

#         return {"correct": evaluation}

#     @weave.op()
#     def summarize(self, score_rows: WeaveList) -> Optional[dict]:
#         """Aggregate all the scores that are calculated for each row by the scoring function.
#         Args:
#             - score_rows: a WeaveList object, nested dict of metrics and scores
#         Returns:
#             - nested dict with the same structure as the input"""

#         # if nothing is provided the weave.flow.scorer.auto_summarize function is used
#         # return auto_summarize(score_rows)

#         valid_data = [x.get("correct") for x in score_rows if x.get("correct") is not None]
#         count_true = list(valid_data).count(True)
#         int_data = [int(x) for x in valid_data]

#         sample_mean = np.mean(int_data) if int_data else 0
#         sample_variance = np.var(int_data) if int_data else 0
#         sample_error = np.sqrt(sample_variance / len(int_data)) if int_data else 0

#         # the extra "correct" layer is not necessary but adds structure in the UI
#         return {
#             "correct": {
#                 "true_count": count_true,
#                 "true_fraction": sample_mean,
#                 "stderr": sample_error,
#             }
#         }

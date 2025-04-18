from typing import Any, AsyncGenerator, Callable, Optional

import weave  # Add weave import
from autogen_core import CancellationToken
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
)
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

    # _listen method removed - evaluation is now triggered proactively

    @weave.op()  # Ensure _process is traced like the base class
    async def _process(self, *, inputs: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput | ToolOutput | None:
        """Perform LLM-based scoring based on inputs."""
        # Expects inputs.inputs to contain 'answers': [AgentOutput] and 'expected': Any (ground_truth)
        if "answers" not in inputs.inputs or "expected" not in inputs.inputs:
            logger.error(f"{self.role}: Missing 'answers' or 'expected' in inputs for scoring.")
            return AgentOutput(role=self.role, error=[f"Missing 'answers' or 'expected' in inputs for scoring."], inputs=inputs)

        # Call the base LLMAgent's _process method which handles template filling and LLM call
        # The template for the scorer should be designed to compare answers[0].content/outputs
        # with the 'expected' ground truth based on 'criteria' in parameters.
        # The base _process will return an AgentOutput. If the LLM call was successful
        # and returned content parsable into _output_model (QualScore), that QualScore
        # instance will be in the .outputs field of the returned AgentOutput.
        logger.debug(f"Scorer agent {self.role} processing evaluation request.")
        evaluation_result_output = await super()._process(inputs=inputs, cancellation_token=cancellation_token, **kwargs)

        # Ensure the output contains the QualScore if successful
        if evaluation_result_output and not evaluation_result_output.is_error:
            if not isinstance(evaluation_result_output.outputs, QualScore):
                logger.warning(f"Scorer {self.role} LLM output was not parsed into QualScore: {evaluation_result_output.outputs}")
                # Optionally add error or return the raw output?
                evaluation_result_output.error.append("LLM output did not conform to QualScore schema.")

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

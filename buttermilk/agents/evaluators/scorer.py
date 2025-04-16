from typing import Any, AsyncGenerator, Callable, Optional

from autogen_core import CancellationToken
from pydantic import BaseModel, Field, PrivateAttr, computed_field

from buttermilk import logger
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

    _ground_truth: dict = PrivateAttr(default={})
    _scores: list[AggResults] = PrivateAttr(default_factory=list)
    _output_model: Optional[type[BaseModel]] = QualScore

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        cancellation_token: CancellationToken = None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        source: str = "unknown",
        **kwargs,
    ) -> None:
        if isinstance(message, AgentOutput):
            # Ignore messages from our own kind
            if message.role == self.role:
                return

            # Identify and store results with qualitative reasons fields
            if "reasons" in message.outputs and message.inputs and message.inputs.records:
                # Score immediately
                input_data = AgentInput(
                    role=self.role,
                    inputs={"answers": [message], "expected": message.inputs.records[-1].ground_truth},
                    records=message.inputs.records[-1:],
                )
                response = await self._run_fn(
                    message=input_data,
                    cancellation_token=cancellation_token,
                    public_callback=public_callback,
                    message_callback=message_callback,
                    **kwargs,
                )
                if response:
                    await public_callback(response)
                    self._scores.append(
                        AggResults(agent=source, answer_id=message.call_id, assessments=response.outputs.assessments, assessor=self.name)
                    )

    async def _handle_control_message(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken = None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        **kwargs,
    ) -> OOBMessages | None:
        """Returns aggregate results calculated from reasoned decisions."""

        if isinstance(message, (AgentInput, ConductorRequest)):
            # Handle conductor requests if needed, e.g., for final summary.
            response = ConductorResponse(
                role=self.role, content=f"Scoring summary for {len(self._scores)} responses", outputs={self.role: self._scores}
            )
            await public_callback(response)
            return response
        return None

    #     # not implemented yet
    #     return
    #     # Store ground truth from records
    #     for record in message.records:
    #         if record.ground_truth:
    #             self._ground_truth = dict(record.ground_truth)
    #             logger.debug(f"Scorer found ground truth: {self._ground_truth}")

    #     # --- Scoring Logic ---
    #     # Collect records fields with ground truth components
    #     if isinstance(message, AgentOutput):

    #         if message.role == self.role:
    #             # Ignore messages from our own kind
    #             return

    #         # Identify and store results with qualitative reasons fields
    #         if self._ground_truth and "reasons" in message.outputs:
    #             logger.debug(f"Scorer tracking result: {message.agent_id}")
    #             self._judge_results.append(message)

    #             # Score immediately if we have ground truth
    #             input_data = AgentInput(
    #                 source=self.name,
    #                 role=self.role,
    #                 inputs={"answer": message, "expected": self._ground_truth},
    #             )
    #             async for _ in self._process():
    #                 pass

    #             if score_output and score_output.outputs:
    #                 # Store scores for later reporting
    #                 self._scores.append({
    #                     "judge_id": message.agent_id,
    #                     "score": score_output.outputs.get("score"),
    #                     "explanation": score_output.outputs.get("explanation"),
    #                 })

    #     return

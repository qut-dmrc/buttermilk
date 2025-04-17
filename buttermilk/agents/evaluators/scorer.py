from typing import Any, AsyncGenerator, Callable, Optional

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

    async def _process(self, *, inputs: AgentInput, cancellation_token: CancellationToken = None, **kwargs) -> AgentOutput | ToolOutput | None:
        """Return score or summary."""
        if inputs.inputs.get("answers"):
            # Score the result
            return await super()._process(inputs=inputs, cancellation_token=cancellation_token, **kwargs)

        # Return summary only
        response = AgentOutput(role=self.role, content=f"Scoring summary for {len(self._scores)} responses", outputs={self.role: self._scores})
        return response

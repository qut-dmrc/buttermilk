from typing import Any

from autogen_core import CancellationToken
from pydantic import PrivateAttr

from buttermilk import logger
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    AllMessages,
    ConductorRequest,
)
from buttermilk.agents.llm import LLMAgent


class LLMScorer(LLMAgent):
    """Qualitatively scores an LLM result against provided ground truth."""

    _ground_truth: dict = PrivateAttr(default={})
    _judge_results: list[AgentOutput] = PrivateAttr(default_factory=list)
    _scores: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    async def receive_output(
        self,
        message: AllMessages,
        **kwargs,
    ) -> AgentOutput | None:
        """Process outputs from JUDGE agents and score them against ground truth."""
        # Collect records fields with ground truth components
        if isinstance(message, AgentOutput):
            # Store ground truth from records
            for record in message.records:
                if record.ground_truth:
                    self._ground_truth = dict(record.ground_truth)
                    logger.debug(f"Scorer found ground truth: {self._ground_truth}")

            if message.agent_role == self.role:
                # Ignore messages from our own kind
                return None

            # Identify and store results with qualitative reasons fields
            if self._ground_truth and "reasons" in message.outputs:
                logger.debug(f"Scorer tracking result: {message.agent_id}")
                self._judge_results.append(message)

                # Score immediately if we have ground truth
                input_data = AgentInput(
                    agent_role=self.role,
                    agent_id=self.id,
                    inputs={"answer": message, "expected": self._ground_truth},
                )
                score_output = await self._process(input_data)

                if score_output and score_output.outputs:
                    # Store scores for later reporting
                    self._scores.append({
                        "judge_id": message.agent_id,
                        "score": score_output.outputs.get("score"),
                        "explanation": score_output.outputs.get("explanation"),
                    })

                return score_output
        return None

    async def _process(
        self,
        input_data: AgentInput | ConductorRequest,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> AgentOutput:
        if isinstance(input_data, ConductorRequest):
            # Return a summary of all scores
            return AgentOutput(
                agent_id=self.id,
                agent_role=self.role,
                content=f"Scoring summary for {len(self._scores)} responses",
                outputs={"scores": self._scores},
            )

        return await super()._process(input_data, cancellation_token, **kwargs)

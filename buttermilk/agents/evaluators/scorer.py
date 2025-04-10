from typing import Any, AsyncGenerator, Callable

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

    async def _process(
        self,
        input_data: AgentInput | ConductorRequest,
        cancellation_token: CancellationToken | None = None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        **kwargs,
    ) -> AgentOutput | None:  # Changed return type
        """Scores JUDGE agent results against ground truth."""

        if isinstance(input_data, ConductorRequest):
            # Handle conductor requests if needed, e.g., for final summary.
            logger.warning(f"{self.role} received ConductorRequest. Summarization logic TBD.")
            return AgentOutput(
                source=self.name,
                role=self.role,
                content=f"Scoring summary for {len(self._scores)} responses",
                outputs={"scores": self._scores},
            )

        # not implemented yet
        return 
        # Store ground truth from records
        for record in message.records:
            if record.ground_truth:
                self._ground_truth = dict(record.ground_truth)
                logger.debug(f"Scorer found ground truth: {self._ground_truth}")

        # --- Scoring Logic ---
        # Collect records fields with ground truth components
        if isinstance(message, AgentOutput):

            if message.role == self.role:
                # Ignore messages from our own kind
                return

            # Identify and store results with qualitative reasons fields
            if self._ground_truth and "reasons" in message.outputs:
                logger.debug(f"Scorer tracking result: {message.agent_id}")
                self._judge_results.append(message)

                # Score immediately if we have ground truth
                input_data = AgentInput(
                    source=self.name,
                    role=self.role,
                    inputs={"answer": message, "expected": self._ground_truth},
                )
                async for _ in self._process():
                    pass

                if score_output and score_output.outputs:
                    # Store scores for later reporting
                    self._scores.append({
                        "judge_id": message.agent_id,
                        "score": score_output.outputs.get("score"),
                        "explanation": score_output.outputs.get("explanation"),
                    })

        return 

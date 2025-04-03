from pydantic import PrivateAttr

from buttermilk._core.contract import AgentInput, AgentOutput, GroupchatMessages
from buttermilk.agents.llm import LLMAgent


class LLMScorer(LLMAgent):
    """Scores an LLM result against provided ground truth."""

    _ground_truth: dict = PrivateAttr(default={})

    async def receive_output(
        self,
        message: GroupchatMessages,
        **kwargs,
    ) -> AgentOutput | None:
        """Trigger on AgentOutput messages with 'reasons' fields."""
        # Collect records fields with ground truth components
        if isinstance(message, AgentOutput):
            for record in message.records:
                if record.ground_truth:
                    self._ground_truth = dict(record.ground_truth)

            if "reasons" in message.outputs:
                # Score this message
                input_data = AgentInput(
                    inputs={"answer": message, "expected": self._ground_truth},
                )
                output = await self._process(input_data)
                return output
        return None

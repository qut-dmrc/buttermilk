import asyncio
from collections.abc import AsyncGenerator
from typing import Self

from autogen_core import AgentId
from pydantic import PrivateAttr, model_validator

from buttermilk.exceptions import ProcessingError
from buttermilk.runner.autogen import AutogenOrchestrator

CONDUCTOR = "HOST"
MANAGER = "MANAGER"


class Selector(AutogenOrchestrator):
    _participants: list = PrivateAttr(default_factory=list)
    _conductor_id: AgentId = PrivateAttr()

    @model_validator(mode="after")
    def _init_conductor(self) -> Self:
        for step in self.steps:
            self._participants.append(
                {
                    "role": step.name,
                    "description": step.description,
                },
            )
        return self

    async def _get_next_step(self) -> AsyncGenerator[dict[str, str], None]:
        """Determine the next step based on the user's prompt"""
        self._next_step = None

        while True:
            # store the last message received, so that any changes in instructions
            # are incorporated before executing the next step
            _last_message = self._last_message
            responses = await self._execute_step(CONDUCTOR)

            if len(responses) > 1:
                raise ProcessingError("Conductor returned multiple responses.")

            instructions = responses[0]

            # TODO(NS): Add finish condition
            # return

            # Determine the next step based on the response
            if not (next_step := instructions.content.get("role")):
                raise ProcessingError("Next step not found from conductor.")
            if next_step not in self._agents:
                raise ProcessingError(
                    f"Step {next_step} not found in registered agents.",
                )

            if self._last_message == _last_message:
                # No change to inputs
                yield {
                    "role": next_step,
                    "question": instructions.content.get("question", ""),
                }
            # wait a bit and go around again
            await asyncio.sleep(5)
            continue

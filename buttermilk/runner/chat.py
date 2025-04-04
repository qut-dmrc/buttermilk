import asyncio
from collections.abc import AsyncGenerator
from typing import Self

from pydantic import PrivateAttr, model_validator

from buttermilk._core.contract import CONDUCTOR, StepRequest
from buttermilk._core.exceptions import ProcessingError
from buttermilk.runner.groupchat import (
    AutogenOrchestrator,
)


class Selector(AutogenOrchestrator):
    _participants: list = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _init_conductor(self) -> Self:
        for step_name, step in self.agents.items():
            self._participants.append(
                {
                    "role": step_name,
                    "description": step.description,
                },
            )
        return self

    async def _get_next_step(self) -> AsyncGenerator[StepRequest, None]:
        """Determine the next step based on the current flow data.

        This generator yields a series of steps to be executed in sequence,
        with each step containing the role and prompt information.

        Yields:
            StepRequest: An object containing:
                - 'role' (str): The agent role/step name to execute
                - 'prompt' (str): The prompt text to send to the agent
                - Additional key-value pairs that might be needed for agent execution

        Example:
            >>> async for step in self._get_next_step():
            >>>     await self._execute_step(**step)

        """
        self._next_step = None

        while True:
            # store the last message received, so that any changes in instructions
            # are incorporated before executing the next step
            _last_message = self._last_message

            # Each step, we proceed by asking the CONDUCTOR agent what to do.
            step = StepRequest(role=CONDUCTOR, source=self.flow_name)
            message = await self._prepare_step(step)
            responses = await self._ask_agents(
                CONDUCTOR,
                message=message,
            )

            if len(responses) > 1:
                raise ProcessingError("Conductor returned multiple responses.")

            instructions = responses[0]

            # TODO(NS): Add finish condition
            # return

            # Determine the next step based on the response
            if not instructions or not (next_step := instructions.outputs.get("role")):
                raise ProcessingError("Next step not found from conductor.")

            if next_step not in self._agent_types:
                raise ProcessingError(
                    f"Step {next_step} not found in registered agents.",
                )

            if self._last_message == _last_message:
                # No change to inputs
                yield StepRequest(
                    role=next_step,
                    prompt=instructions.outputs.pop("prompt", ""),
                    description=instructions.outputs.pop("plan", ""),
                    tool=instructions.outputs.get("tool", None),
                    arguments=instructions.outputs,
                )
            # wait a bit and go around again
            await asyncio.sleep(5)
            continue

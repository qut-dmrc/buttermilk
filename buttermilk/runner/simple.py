
import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import (
    PrivateAttr,
)

from buttermilk._core.contract import (
    FlowMessage,
    ManagerMessage,
    ManagerRequest,
    StepRequest,
)
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk.bm import logger
from buttermilk.runner.groupchat import AutogenOrchestrator


class Sequencer(AutogenOrchestrator):
    _step_generator = PrivateAttr(default=None)

    async def run(self, request: Any = None) -> None:
        """Main execution method that sets up agents and manages the flow"""
        try:
            # Setup autogen runtime environment
            await self._setup_runtime()
            self._step_generator = self._get_next_step()

            # start the agents
            await self._runtime.publish_message(
                FlowMessage(agent_id=self.flow_name, agent_role="orchestrator"),
                topic_id=self._topic,
            )
            await asyncio.sleep(1)

            # First, introduce ourselves, and prompt the user for input
            await self._send_ui_message(
                ManagerRequest(
                    role=self.flow_name,
                    content=f"Started {self.flow_name}: {self.description}. Please enter your question or prompt and let me know when you're ready to go.",
                ),
            )
            if not await self._user_confirmation.get():
                await self._send_ui_message(
                    ManagerMessage(content="OK, shutting down thread."),
                )
                return

            while True:
                try:
                    # Get next step from our CONDUCTOR agent
                    step = await anext(self._step_generator)

                    # For now, ALWAYS get confirmation from the user (MANAGER) role
                    confirm_step = ManagerRequest(
                        role=self.flow_name,
                        content="Here's my proposed next step. Do you want to proceed?",
                        inputs=step.arguments,
                    )
                    confirm_step.inputs["prompt"] = step.prompt
                    confirm_step.inputs["description"] = step.description

                    await self._send_ui_message(confirm_step)
                    if not await self._user_confirmation.get():
                        # User did not confirm plan; go back and get new instructions
                        continue
                    # Run next step
                    await self._execute_step(step)

                except StopAsyncIteration:
                    logger.info("AutogenOrchestrator.run: Flow completed.")
                    break
                except ProcessingError as e:
                    logger.error(f"Error in AutogenOrchestrator.run: {e}")
                except FatalError:
                    raise
                except Exception as e:  # This is only here for debugging for now.
                    logger.exception(f"Error in AutogenOrchestrator.run: {e}")

                await asyncio.sleep(0.1)

        except FatalError as e:
            logger.exception(f"Error in AutogenOrchestrator.run: {e}")
        finally:
            # Clean up resources
            await self._cleanup()

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
        for step_name in self.agents.keys():
            yield StepRequest(role=step_name, source=self.flow_name)

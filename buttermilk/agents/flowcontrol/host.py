import asyncio
from collections.abc import Awaitable
from math import ceil
from huggingface_hub import User
from pydantic import BaseModel, Field, PrivateAttr
from buttermilk._core.contract import (
    COMMAND_SYMBOL,
    END,
    AssistantMessage,
    ConductorRequest,
    ConductorResponse,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    OOBMessages,
    StepRequest,
    TaskProcessingStarted,
    UserMessage,
)
from buttermilk.agents.llm import LLMAgent

from typing import Any, AsyncGenerator, Callable, Optional, Self, Union
from autogen_core import CancellationToken, MessageContext, message_handler

from buttermilk import logger
from buttermilk._core.config import DataSourceConfig
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    AllMessages,
    FlowMessage,
    GroupchatMessageTypes,
    OOBMessages,
    UserInstructions,
    TaskProcessingComplete,
    ProceedToNextTaskSignal,
)

TRUNCATE_LEN = 1000  # characters per history message
class HostAgent(LLMAgent):
    """Coordinators for group chats that use an LLM. Can act as the 'beat' to regulate flow."""

    _input_callback: Any = PrivateAttr(...)
    _pending_agent_id: str | None = PrivateAttr(default=None) # Track agent waiting for signal

    _output_model: Optional[type[BaseModel]] = StepRequest
    _message_types_handled: type[Any] = PrivateAttr(default=Union[ConductorRequest])

    # Additional configuration
    max_wait_time: int = Field(
        default=300,
        description="Maximum time to wait for agent responses in seconds",
    )
    completion_threshold_ratio: float = Field(
        default=0.8,
        description="Ratio of agents that must complete a step before proceeding (0.0 to 1.0)",
    )
    _step_generator: Any = PrivateAttr(default=None)
    _current_step_name: str | None = PrivateAttr(default=None)
    _completed_agents_current_step: set[str] = PrivateAttr(default_factory=set)
    _expected_agents_current_step: set[str] = PrivateAttr(default_factory=set)

    _participants: dict = PrivateAttr(default={})
    _step_completion_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the agent"""
        self._input_callback = input_callback
        self._step_generator = self._get_next_step()
        self._step_completion_event.set()  # Ready to process
        await super().initialize(**kwargs) # Call parent initialize if needed

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        cancellation_token: CancellationToken = None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        source: str = "unknown",
        **kwargs,
    ) -> None:
        # Log messages to our local context cache, but truncate them

        if isinstance(message, (AgentOutput, ConductorResponse)):
            await self._model_context.add_message(AssistantMessage(content=str(message.content)[:TRUNCATE_LEN], source=source))
        else:
            if not message.content.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(message.content)[:TRUNCATE_LEN], source=source))

    async def _handle_control_message(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken = None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        **kwargs,
    ) -> OOBMessages | None:

        # --- Handle Task Completion from Worker Agents ---
        if isinstance(message, TaskProcessingComplete):
            if message.role == self._current_step_name:
                self._completed_agents_current_step.add(message.agent_id)
                logger.info(f"Host received TaskComplete from {message.agent_id} (Task {message.task_index}, More: {message.more_tasks_remain})")
        elif isinstance(message, TaskProcessingStarted):
            if message.role == self._current_step_name:
                self._expected_agents_current_step.add(message.agent_id)

        required_completions = ceil(int(len(self._expected_agents_current_step) * self.completion_threshold_ratio))
        logger.info(
            f"Waiting for step '{self._current_step_name}' to complete. So far we have received results from {len(self._completed_agents_current_step)} of {len(self._expected_agents_current_step)} agents for step '{self._current_step_name}'. Waiting for at least {required_completions} before moving on."
        )
        if len(self._completed_agents_current_step) >= required_completions:
            logger.info(f"Completion threshold reached for step '{self._current_step_name}'.")
            self._step_completion_event.set()
        return

    async def _process(self, *, inputs: AgentInput, cancellation_token: CancellationToken = None, **kwargs) -> AgentOutput | None:
        try:
            # Wait for enough completions, with a timeout
            logger.info(
                f"Waiting for previous step to complete: {self._current_step_name}. So far we have received results from {len(self._completed_agents_current_step)} of {len(self._expected_agents_current_step)} agents for step '{self._current_step_name}'."
            )
            await asyncio.wait_for(self._step_completion_event.wait(), timeout=self.max_wait_time)
            logger.info(f"Previous step '{self._current_step_name}' cleared, moving on.")
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout waiting for step completion. "
                f"Proceeding with {len(self._completed_agents_current_step)}/{len(self._expected_agents_current_step)} completed agents."
            )

        if not self._participants:
            # initialise
            self._participants = inputs.inputs['participants']

        step = await anext(self._step_generator)
        if step == self.role:
            # don't call ourselves please
            self._step_completion_event.clear()
            return None

        # Reset completion tracking
        self._current_step_name = step.role
        self._expected_agents_current_step.clear()
        self._completed_agents_current_step.clear()
        self._step_completion_event.clear()

        response = AgentOutput(role=self.role)
        response.outputs = step
        logger.info(f"Next step: {self._current_step_name}.")

        return response

    async def _get_next_step(self) -> AsyncGenerator[StepRequest, None]:
        """Determine the next step based on the current flow data."""
        while self._participants is None:
            await asyncio.sleep(0.1)
        for step_name, cfg in self._participants.items():
            yield StepRequest(role=step_name, description=f"Sequence host calling {step_name}.")
        yield StepRequest(role=END, description=f"Sequence wrapping up.")

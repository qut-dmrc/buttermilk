import asyncio
from math import ceil
from typing import Any, AsyncGenerator, Callable, Awaitable, Optional, cast, Union

import weave
from pydantic import Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.contract import (
    COMMAND_SYMBOL,
    END,
    WAIT,
    AgentInput,
    AgentOutput,
    AssistantMessage,
    ConductorRequest,
    ConductorResponse,
    GroupchatMessageTypes,
    OOBMessages,
    StepRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
    UserMessage,
)

TRUNCATE_LEN = 1000  # characters per history message


class Sequencer(Agent):
    """
    A simple coordinator for group chats that uses round-robin scheduling.
    This agent doesn't use LLM for decision making - it just cycles through all agents in sequence.
    """

    _input_callback: Any = PrivateAttr(default=None)
    _pending_agent_id: str | None = PrivateAttr(default=None)  # Track agent waiting for signal
    _output_model: type = StepRequest

    # Configuration
    max_wait_time: int = Field(
        default=300,
        description="Maximum time to wait for agent responses in seconds",
    )
    completion_threshold_ratio: float = Field(
        default=0.8,
        description="Ratio of agents that must complete a step before proceeding (0.0 to 1.0)",
    )

    # State tracking
    _current_step_name: str | None = PrivateAttr(default=None)
    _completed_agents_current_step: set[str] = PrivateAttr(default_factory=set)
    _expected_agents_current_step: set[str] = PrivateAttr(default_factory=set)

    _step_generator: Any = PrivateAttr(default=None)
    _participants: dict = PrivateAttr(default_factory=dict)
    _step_completion_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _current_idx: int = PrivateAttr(default=0)  # Track current position in round-robin

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the agent"""
        self._input_callback = input_callback
        self._step_completion_event.set()  # Ready to process
        self._step_generator = self._sequence()
        await super().initialize(**kwargs)

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: Optional[Any] = None,
        source: str = "",
        public_callback: Callable | None = None,
        message_callback: Callable | None = None,
        **kwargs,
    ) -> None:
        # Log messages to local context cache, but truncate them
        if isinstance(message, (AgentOutput, ConductorResponse)):
            await self._model_context.add_message(AssistantMessage(content=str(message.content)[:TRUNCATE_LEN], source=source))
        else:
            # Check if message has content and it's not a command
            content = getattr(message, "content", getattr(message, "prompt", ""))
            if content and not str(content).startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(content)[:TRUNCATE_LEN], source=source))

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token=None,
        **kwargs,
    ) -> OOBMessages | None:
        # Handle task completion from worker agents
        if isinstance(message, TaskProcessingComplete):
            if message.role == self._current_step_name:
                self._completed_agents_current_step.add(message.agent_id)
                logger.info(f"Host received TaskComplete from {message.agent_id} (Task {message.task_index}, More: {message.more_tasks_remain})")
        elif isinstance(message, TaskProcessingStarted):
            if message.role == self._current_step_name:
                self._expected_agents_current_step.add(message.agent_id)

        await self._check_completions()
        if isinstance(message, ConductorRequest):
            next_step = await self._get_next_step(inputs=message)
            return cast(ConductorResponse, next_step)
        return None

    async def _check_completions(self) -> None:
        """Check if enough agents have completed the current step to move on"""
        required_completions = ceil(len(self._expected_agents_current_step) * self.completion_threshold_ratio)
        if required_completions > 0:
            logger.info(
                f"Waiting for step '{self._current_step_name}' to complete. So far we have received results from {len(self._completed_agents_current_step)} of {len(self._expected_agents_current_step)} agents for step '{self._current_step_name}'. Waiting for at least {required_completions} before moving on."
            )
            if len(self._completed_agents_current_step) >= required_completions:
                logger.info(f"Completion threshold reached for step '{self._current_step_name}'.")
                self._step_completion_event.set()
            else:
                self._step_completion_event.clear()
        else:
            self._step_completion_event.set()

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Generate steps in round-robin fashion"""
        while not self._participants:
            await asyncio.sleep(0.1)

        # Get list of participant roles for round-robin
        roles = list(self._participants.keys())
        if not roles:
            return

        # Round-robin through all roles
        while True:
            for role in roles:
                yield StepRequest(role=role, prompt="", description=f"Round-robin sequencer calling {role}.")

            # After one complete cycle, yield an END marker
            yield StepRequest(role=END, prompt="", description="Sequence complete.")

    async def _get_next_step(self, inputs: ConductorRequest) -> AgentOutput:
        """Determine the next step based on round-robin scheduling"""
        try:
            # Wait for enough completions, with a timeout
            await self._check_completions()
            await asyncio.wait_for(self._step_completion_event.wait(), timeout=self.max_wait_time)
            logger.info(f"Previous step '{self._current_step_name}' cleared, moving on.")
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout waiting for step completion. "
                f"Proceeding with {len(self._completed_agents_current_step)}/{len(self._expected_agents_current_step)} completed agents."
            )

        if not self._participants:
            # Initialize participants from input
            self._participants = inputs.inputs.get("participants", {})
            logger.info(f"Initialized with {len(self._participants)} participants: {', '.join(self._participants.keys())}")

        # Get next step using round-robin
        step = await self._choose(inputs=inputs)

        if step.role == self.role:
            # Don't call ourselves
            logger.warning(f"Avoiding self-call, skipping to next agent")
            step = await self._choose(inputs=inputs)

        # If role doesn't exist in participants, try to use a valid one
        if step.role not in self._participants and step.role != END:
            logger.warning(f"Host could not find next step. Suggested {step.role}, which doesn't exist.")

            # Instead of defaulting to WAIT, use the first valid participant if available
            if self._participants:
                step.role = next(iter(self._participants.keys()))
                logger.info(f"Using valid participant {step.role} instead of WAIT")
            else:
                step.role = WAIT

        # Reset completion tracking
        self._current_step_name = step.role
        self._expected_agents_current_step.clear()
        self._completed_agents_current_step.clear()
        self._step_completion_event.clear()

        response = AgentOutput()
        # Set response attributes
        response.content = f"Next step: {self._current_step_name}."
        response.outputs = step
        logger.info(f"Next step: {self._current_step_name}.")

        return response

    async def _choose(self, inputs: ConductorRequest) -> StepRequest:
        """Choose the next step using round-robin strategy"""
        step = await anext(self._step_generator)
        return step

    @weave.op()
    async def _process(self, *, inputs: AgentInput, cancellation_token=None, **kwargs) -> AgentOutput | None:
        """Process inputs and generate responses"""
        response = AgentOutput()

        if not hasattr(inputs, "prompt") or not inputs.prompt and not inputs.inputs.get("task"):
            # No prompt provided, just initialize
            logger.info(f"Sequencer initialized: {self.name}")
            response.content = f"Sequencer initialized: {self.name}"
            return response

        try:
            # Process prompt if needed
            prompt = inputs.prompt.strip()
            response.content = f"Sequencer received: {prompt}"
            logger.info(f"Sequencer received: {prompt}")

            return response
        except Exception as e:
            logger.error(f"Error in Sequencer._process: {e}", exc_info=True)
            response.error = [str(e)]
            return response

    async def on_reset(self, cancellation_token=None) -> None:
        """Reset the agent's internal state"""
        await super().on_reset(cancellation_token)
        self._current_step_name = None
        self._completed_agents_current_step.clear()
        self._expected_agents_current_step.clear()
        self._participants.clear()
        self._current_idx = 0
        self._step_generator = self._sequence()
        self._step_completion_event.set()

import asyncio
from math import ceil
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional, cast

from pydantic import Field, PrivateAttr

from autogen_core.models import AssistantMessage, UserMessage
from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.contract import (
    COMMAND_SYMBOL,
    CONDUCTOR,  # Symbol indicating a command message, likely ignored by _listen.
    END,  # Special role indicating the end of the flow.
    MANAGER,  # Role name for the Manager/UI agent.
    WAIT,  # Special role indicating the conductor should wait.
    AgentInput,  # Standard input message type.
    AgentOutput,
    ConductorRequest,
    ConductorResponse,
    GroupchatMessageTypes,
    OOBMessages,
    StepRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
)

# Maximum length to truncate messages stored in the agent's context history.
TRUNCATE_LEN = 1000


class Sequencer(Agent):
    """
    A simple, non-LLM agent that coordinates a sequence of steps using round-robin scheduling.

    This agent acts as a basic conductor (`CONDUCTOR` role). When asked for the next step
    (via `_handle_events` or `_process` with a `ConductorRequest`), it cycles through a
    list of participant agents provided in the request. It waits for a configurable
    threshold of agents from the previous step to report completion (`TaskProcessingComplete`)
    before suggesting the next step.

    Attributes:
        max_wait_time: Max seconds to wait for step completion before proceeding anyway.
        completion_threshold_ratio: Fraction of agents needed to complete a step (0.0 to 1.0).
        _input_callback: Callback function (if any) provided during initialization (unused currently).
        _output_model: Specifies the expected output type (StepRequest).
        _current_step_name: The role/name of the step currently being executed by worker agents.
        _completed_agents_current_step: Set of agent IDs that have completed the current step.
        _expected_agents_current_step: Set of agent IDs expected to participate in the current step (populated when they send TaskProcessingStarted).
        _step_generator: Async generator yielding the next step in the sequence.
        _participants: Dictionary mapping role names to participant configurations (from ConductorRequest).
        _step_completion_event: asyncio.Event used to signal when the completion threshold is met.
    """

    # TODO: _pending_agent_id seems unused. Consider removal.
    # _pending_agent_id: str | None = PrivateAttr(default=None) # Track agent waiting for signal
    _output_model: type = StepRequest  # Specifies the expected output type for agent calls.

    # Configuration fields exposed via Hydra/Pydantic.
    max_wait_time: int = Field(default=300, description="Maximum time (seconds) to wait for agent responses before proceeding.")
    completion_threshold_ratio: float = Field(default=0.8, description="Ratio of agents that must complete a step before proceeding (0.0 to 1.0).")
    # TODO: Consider adding a minimum number of completions threshold as well.

    # Internal state tracking attributes.
    _current_step_name: str | None = PrivateAttr(default=None)  # Role name of the currently executing step.
    _completed_agents_current_step: set[str] = PrivateAttr(default_factory=set)  # IDs of agents finished current step.
    _expected_agents_current_step: set[str] = PrivateAttr(default_factory=set)  # IDs of agents started current step.

    _step_generator: AsyncGenerator[StepRequest, None] | None = PrivateAttr(default=None)  # Holds the round-robin generator.
    _participants: dict[str, Any] = PrivateAttr(default_factory=dict)  # Participant roles from ConductorRequest.
    _step_completion_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)  # Signals step completion threshold.
    # TODO: _current_idx seems unused as _step_generator handles the sequence. Consider removal.
    # _current_idx: int = PrivateAttr(default=0) # Track current position in round-robin

    # TODO: _input_callback seems unused after initialization. Verify and potentially remove.
    _input_callback: Callable[..., Awaitable[None]] | None = PrivateAttr(default=None)

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the agent"""
        self._input_callback = input_callback
        self._step_completion_event.set()  # Ready to process
        self._step_generator = self._sequence()
        await super().initialize(**kwargs)

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
        logger.debug(f"Initializing Sequencer agent {self.id}...")
        self._input_callback = input_callback
        self._step_completion_event.set()  # Ready to process
        self._step_generator = self._sequence()  # Initialize the step generator.
        await super().initialize(**kwargs)
        logger.debug(f"Sequencer agent {self.id} initialized.")

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: Optional[Any] = None,  # Autogen cancellation token.
        source: str = "",
        public_callback: Callable | None = None,  # Callback to publish to default topic (unused here).
        message_callback: Callable | None = None,  # Callback to publish to incoming message topic (unused here).
        **kwargs,
    ) -> None:
        """Passively listens to messages on the bus, adding them to context history."""
        # Only logs messages, doesn't actively react in _listen phase.
        # Truncates messages to avoid overly long history.
        # TODO: Consider more sophisticated history management (e.g., summaries, filtering).
        if isinstance(message, (AgentOutput, ConductorResponse)):
            # Store assistant messages (agent outputs).
            await self._model_context.add_message(AssistantMessage(content=message.contents[:TRUNCATE_LEN], source=source))
        else:
            # Store user messages, ignoring commands.
            # Attempts to get 'content' or 'prompt' attributes.
            content = getattr(message, "content", getattr(message, "prompt", ""))
            if content and not str(content).startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(content)[:TRUNCATE_LEN], source=source))

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token=None,
        **kwargs,
    ) -> OOBMessages | None:
        """Handles Out-Of-Band messages, primarily TaskProcessing status updates and ConductorRequests."""
        logger.debug(f"Sequencer {self.id} handling event: {type(message).__name__}")

        # Handle task completion signals from worker agents.
        if isinstance(message, TaskProcessingComplete):
            # Only track completions for the currently active step.
            if message.role == self._current_step_name:
                if message.agent_id not in self._completed_agents_current_step:
                    self._completed_agents_current_step.add(message.agent_id)
                    logger.info(
                        f"Sequencer received TaskComplete from {message.agent_id} for step '{self._current_step_name}' "
                        f"(Task {message.task_index}, More: {message.more_tasks_remain}, Error: {message.is_error})"
                    )
                    # Check if the completion threshold is now met.
                    await self._check_completions()
                else:
                    # Log if we receive a duplicate completion signal.
                    logger.warning(f"Sequencer received duplicate TaskComplete from {message.agent_id} for step '{self._current_step_name}'.")
            else:
                # Ignore completions for steps other than the current one.
                logger.debug(
                    f"Sequencer ignored TaskComplete for inactive step '{message.role}' (current: '{self._current_step_name}') from {message.agent_id}."
                )

        elif isinstance(message, TaskProcessingStarted):
            # Track which agents have started the current step.
            if message.role == self._current_step_name:
                if message.agent_id not in self._expected_agents_current_step:
                    self._expected_agents_current_step.add(message.agent_id)
                    logger.info(f"Sequencer noted TaskStarted from {message.agent_id} for step '{self._current_step_name}'.")
                    # Re-check completions in case threshold changes due to newly started agents.
                    await self._check_completions()
                # else: Agent already known to have started.
            else:
                # Ignore starts for steps other than the current one.
                logger.debug(
                    f"Sequencer ignored TaskStarted for inactive step '{message.role}' (current: '{self._current_step_name}') from {message.agent_id}."
                )

        # If this is a direct request for the next step (ConductorRequest).
        if isinstance(message, ConductorRequest):
            # Get the next step suggestion and return it as the response.
            # Note: This assumes _get_next_step returns AgentOutput compatible with ConductorResponse needs.
            # TODO: Ensure AgentOutput structure matches expected ConductorResponse.
            next_step_output = await self._get_next_step(message=message)
            # The return type hint expects OOBMessages, AgentOutput isn't strictly that, but
            # Autogen might handle the direct return value appropriately in send_message context.
            # We cast here primarily for type checker satisfaction based on observed usage.
            # TODO: Revisit this cast. Consider returning a ConductorResponse explicitly if needed.
            return cast(OOBMessages, next_step_output)

        # Return None for status updates that don't require a direct response back to the sender.
        return None

    async def _check_completions(self) -> None:
        """
        Checks if the completion threshold for the current step has been met.

        Updates the `_step_completion_event` based on the number of completed agents
        versus the number expected and the `completion_threshold_ratio`.
        """
        if not self._expected_agents_current_step:
            # If no agents were expected (or none have started yet), consider it complete.
            if not self._step_completion_event.is_set():
                logger.debug(f"No agents expected or started for step '{self._current_step_name}', setting completion event.")
                self._step_completion_event.set()
            return

        # Calculate the minimum number of agents required to complete.
        required_completions = ceil(len(self._expected_agents_current_step) * self.completion_threshold_ratio)

        num_completed = len(self._completed_agents_current_step)
        num_expected = len(self._expected_agents_current_step)

        logger.info(
            f"Step '{self._current_step_name}': Checking completions - "
            f"{num_completed}/{num_expected} completed (Threshold ratio: {self.completion_threshold_ratio}, Required: {required_completions})."
        )

        # Set or clear the event based on whether the threshold is met.
        if num_completed >= required_completions:
            if not self._step_completion_event.is_set():
                logger.info(f"Completion threshold reached for step '{self._current_step_name}'. Setting event.")
                self._step_completion_event.set()
        elif self._step_completion_event.is_set():
            logger.info(f"Completion threshold NOT met for step '{self._current_step_name}'. Clearing event.")
            self._step_completion_event.clear()
            # else: Event already clear, no change needed.

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Async generator that yields the next step request in a round-robin sequence."""
        # Wait until participant list is populated (by the first ConductorRequest).
        while not self._participants:
            logger.debug("Sequencer generator waiting for participants...")
            await asyncio.sleep(0.5)  # Sleep briefly while waiting
        logger.info(f"Sequencer generator starting sequence with participants: {list(self._participants.keys())}")

        # Get the list of participant roles (keys of the dictionary).
        roles = list(self._participants.keys())
        if not roles:
            logger.warning("Sequencer has no participant roles to sequence.")
            # Yield END immediately if no participants.
            yield StepRequest(role=END, prompt="", description="No participants in sequence.")
            return  # Stop generation

        # Cycle through roles indefinitely (until generator is closed).
        # This basic version just does one round.
        # TODO: Implement logic for multiple rounds or different termination conditions if needed.
        current_round = 0
        logger.info(f"Starting round {current_round + 1} of sequence.")
        for role in roles:
            # Skip MANAGER and self (CONDUCTOR/Sequencer) roles in the sequence.
            if role == MANAGER or role == self.role:
                logger.debug(f"Skipping role '{role}' in sequence.")
                continue

            logger.debug(f"Sequencer yielding step for role: {role}")
            yield StepRequest(role=role, prompt="", description=f"Round-robin sequencer requesting action from {role}.")

        # After one full pass through the roles, signal the end.
        logger.info("Sequence completed one round. Yielding END.")
        yield StepRequest(role=END, prompt="", description="Round-robin sequence complete.")

    async def _get_next_step(self, message: ConductorRequest) -> AgentOutput:
        """
        Determines the next step, waiting for the previous step's completion.

        Waits for the `_step_completion_event` (with timeout), then gets the next
        step from the `_step_generator`. Updates internal state for the new step.

        Args:
            message: The incoming ConductorRequest containing participant info.

        Returns:
            An AgentOutput containing the next StepRequest or a WAIT/END step.
        """
        try:
            # Wait for the previous step to meet its completion threshold.
            logger.info(f"Waiting for completion event for step '{self._current_step_name}' (Timeout: {self.max_wait_time}s)...")
            # Ensure check_completions runs at least once before waiting.
            await self._check_completions()
            await asyncio.wait_for(self._step_completion_event.wait(), timeout=self.max_wait_time)
            logger.info(f"Step '{self._current_step_name}' completed or timed out. Proceeding.")
        except asyncio.TimeoutError:
            # Log timeout and proceed anyway.
            logger.warning(
                f"Timeout after {self.max_wait_time}s waiting for step '{self._current_step_name}' completion. "
                f"Proceeding with {len(self._completed_agents_current_step)}/{len(self._expected_agents_current_step)} completed."
            )
            # Ensure the event is set so the next step can start.
            self._step_completion_event.set()

        # Initialize participants if this is the first request.
        if not self._participants:
            # Extract participants from the 'inputs' field of the ConductorRequest.
            # TODO: Validate the structure of message.inputs["participants"].
            self._participants = message.inputs.get("participants", {})
            if self._participants:
                logger.info(f"Sequencer initialized with participants: {list(self._participants.keys())}")
                # Need to re-initialize the generator now that participants are known.
                self._step_generator = self._sequence()
            else:
                logger.warning("ConductorRequest received, but no participants found in inputs.")
                # Return END step if no participants.
                return AgentOutput(agent_info=self._cfg, outputs=StepRequest(role=END, prompt="", description="No participants found."))

        # Get the next step from the generator.
        next_step = CONDUCTOR  # set to our value for the while loop; never return our own role
        while next_step == CONDUCTOR:
            next_step = await self._choose(message=message)  # _choose just wraps anext(_step_generator)

        # Prepare the output containing the next step request.
        output = AgentOutput(agent_info=self._cfg, outputs=next_step)

        if not next_step or next_step.role == END:
            logger.info("Sequencer reached end of sequence.")
            self._current_step_name = END  # Mark current step as END
        else:
            # If the suggested role isn't a known participant, signal WAIT.
            # TODO: Should this raise an error or try the *next* participant instead?
            if next_step.role not in self._participants:
                logger.warning(f"Sequencer suggested role '{next_step.role}' not in participants {list(self._participants.keys())}. Setting to WAIT.")
                next_step.role = WAIT
                next_step.description = f"Suggested role {next_step.role} not found."
                output.outputs = next_step  # Update output

            # Prepare for the new step.
            logger.info(f"Sequencer initiating next step for role: {next_step.role}")
            self._current_step_name = next_step.role
            self._expected_agents_current_step.clear()
            self._completed_agents_current_step.clear()
            # Clear the event; it will be set again when the new step completes.
            self._step_completion_event.clear()

        return output

    async def _choose(self, message: ConductorRequest) -> StepRequest:
        """Retrieves the next step from the internal step generator."""
        if not self._step_generator:
            logger.error("Step generator not initialized!")
            # Attempt to recover? Or raise error? For now, return END.
            return StepRequest(role=END, prompt="", description="Error: Step generator missing.")
        try:
            # Get the next item from the async generator.
            step = await anext(self._step_generator)
            return step
        except StopAsyncIteration:
            # Generator is exhausted (sequence finished).
            logger.info("Step generator finished.")
            return StepRequest(role=END, prompt="", description="Sequence generator finished.")
        except Exception as e:
            # Catch any other error during generation.
            logger.error(f"Error getting next step from generator: {e}")
            return StepRequest(role=END, prompt="", description=f"Error in sequence: {e}")

    async def _process(self, *, message: AgentInput, cancellation_token=None, **kwargs) -> AgentOutput:
        """Handles direct calls to the Sequencer agent (e.g., via AgentInput)."""
        # Primarily designed to respond to ConductorRequest via _handle_events.
        # This handles other AgentInput types, potentially for status or simple commands.
        logger.debug(f"Sequencer {self.id} processing direct message: {type(message).__name__}")

        # If it's actually a ConductorRequest disguised as AgentInput (less ideal).
        if isinstance(message, ConductorRequest):
            # Delegate to the standard ConductorRequest handling.
            next_step_output = await self._get_next_step(message=message)
            # TODO: The return type mismatch needs careful handling.
            # _process expects AgentOutput|None, but _get_next_step returns AgentOutput containing StepRequest.
            # This cast might be okay if the caller handles the nested structure.
            return cast(AgentOutput, next_step_output)  # Potential type issue here.

        # Prepare a default response.
        response = AgentOutput(agent_info=self._cfg)

        # Check if there's a meaningful prompt or task description.
        # TODO: Define more clearly what direct AgentInput prompts the Sequencer should handle.
        prompt = getattr(message, "prompt", "")
        task = message.inputs.get("task", "") if hasattr(message, "inputs") else ""
        if not prompt and not task:
            logger.warning(f"Sequencer {self.id} called directly with no prompt or task.")
            # Maybe return status or help message? Returning None for now.

            response.outputs = StepRequest(role=WAIT, prompt="", description=f"Sequencer {self.id} called directly with no prompt or task.")
            return response

        try:
            # Log reception of the prompt.
            effective_prompt = prompt or task
            # TODO: AgentOutput doesn't have '.content'. Use '.outputs' or a dedicated field.
            # response.content = f"Sequencer {self.id} received direct input: '{effective_prompt[:100]}...'"
            response.outputs = {"status": f"Sequencer {self.id} received direct input: '{effective_prompt[:100]}...'"}
            logger.info(f"Sequencer {self.id} received direct input: '{effective_prompt[:100]}...'")

            # Currently doesn't *do* anything with the direct input other than acknowledge.
            return response
        except Exception as e:
            logger.error(f"Error in Sequencer._process: {e}")
            # Populate error field in the standard AgentOutput.
            # Ensure outputs is initialized if error occurs before content is set.
            # TODO: AgentOutput doesn't have '.error'. Use is_error flag and put error in outputs.
            # response.error = [str(e)]
            response.set_error(str(e))
            return response

    async def on_reset(self, cancellation_token=None) -> None:
        """Resets the Sequencer's internal state."""
        logger.info(f"Resetting Sequencer agent {self.id}...")
        await super().on_reset(cancellation_token)
        self._current_step_name = None
        self._completed_agents_current_step.clear()
        self._expected_agents_current_step.clear()
        self._participants.clear()
        # TODO: _current_idx is unused? Confirm and remove if true.
        # self._current_idx = 0
        # Re-initialize the step generator (will wait for participants again).
        self._step_generator = self._sequence()
        # Set completion event to ready state.
        self._step_completion_event.set()
        logger.info(f"Sequencer agent {self.id} reset complete.")

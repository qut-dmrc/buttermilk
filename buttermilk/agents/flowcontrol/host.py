import asyncio
from collections import defaultdict  # Import defaultdict
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from autogen_core import CancellationToken
from autogen_core.models import AssistantMessage, UserMessage
from pydantic import BaseModel, Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.constants import COMMAND_SYMBOL, CONDUCTOR, END, MANAGER, WAIT
from buttermilk._core.contract import (
    AgentInput,
    AgentResponse,
    AgentTrace,
    ConductorRequest,
    ConductorResponse,
    ErrorEvent,
    GroupchatMessageTypes,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    OOBMessages,
    StepRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
    TaskProgressUpdate,
)
from buttermilk._core.exceptions import FatalError

TRUNCATE_LEN = 1000  # characters per history message
INTERRUPT_ROLE = "INTERRUPT"  # Special role for interrupt handling


class HostAgent(Agent):
    """Base coordinator for group chats and flow control.

    This agent acts as a conductor (`CONDUCTOR` role). It iterates through a
    predefined sequence of participant roles. For each step, it sends a request
    to the corresponding role. Before proceeding to the next step, it waits
    until all agents that have started processing a task have reported completion.
    Uses a dictionary to count pending tasks per agent ID and a Condition variable
    for synchronization.
    """

    _input_callback: Callable[..., Awaitable[None]] = PrivateAttr()
    _pending_agent_id: str | None = PrivateAttr(default=None)  # Track agent waiting for signal (Seems unused? Keep for now)

    _output_model: type[BaseModel] | None = StepRequest
    _message_types_handled: type[Any] = PrivateAttr(default=type(ConductorRequest))

    # Additional configuration
    max_wait_time: int = Field(
        default=240,
        description="Maximum time to wait for agent responses in seconds",
    )

    human_in_loop: bool = Field(
        default=True,
        description="Whether to interact with the human/manager for step confirmation",
    )
    #  Queue for receiving confirmation responses from the MANAGER.
    _user_confirmation: asyncio.Queue[ManagerResponse] = PrivateAttr(default_factory=lambda: asyncio.Queue(maxsize=1))

    _conductor_task: asyncio.Task | None = PrivateAttr(default=None)
    _current_step_name: str | None = PrivateAttr(default=None)
    # Track count of pending tasks per agent ID
    _pending_tasks_by_agent: defaultdict[str, int] = PrivateAttr(default_factory=lambda: defaultdict(int))
    # Condition variable to synchronize task completion based on pending agents
    _tasks_condition: asyncio.Condition = PrivateAttr(default_factory=asyncio.Condition)

    _step_generator: AsyncGenerator[StepRequest, None] | None = PrivateAttr(default=None)
    _participants: dict[str, Any] = PrivateAttr(default_factory=dict)  # Stores role descriptions
    _participants_set_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)  # Event for participants being set

    # State tracking for exploration (kept for potential future use)
    _exploration_path: list[str] = PrivateAttr(default_factory=list)
    _exploration_results: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _user_feedback: list[str] = PrivateAttr(default_factory=list)  # Kept in case ManagerResponse is still used elsewhere

    # Progress tracking
    _total_steps: int = PrivateAttr(default=0)
    _current_step: int = PrivateAttr(default=0)
    _step_sequence: list[str] = PrivateAttr(default_factory=list)

    async def initialize(
        self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs: Any,
    ) -> None:
        """Initialize the agent."""
        self._input_callback = input_callback
        # Initialize condition and pending tasks dict
        self._tasks_condition = asyncio.Condition()
        self._pending_tasks_by_agent = defaultdict(int)
        self._participants_set_event.clear()  # Participants not set yet
        self._step_generator = self._sequence()
        self._exploration_path = []
        self._exploration_results = {}
        self._user_feedback = []
        self._total_steps = 0  # Will be set when participants are known
        self._current_step = 0
        self._step_sequence = []  # Will be set when participants are known
        await super().initialize(**kwargs)  # Call parent initialize if needed

    async def _listen(
        self,
        message: GroupchatMessageTypes | StepRequest,
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        public_callback: Callable | None = None,
        message_callback: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        """Listen to messages in the group chat and maintain conversation history."""
        # Log messages to our local context cache, but truncate them
        content_to_log = None
        # Allow both UserMessage and AssistantMessage types
        msg_type: type[UserMessage | AssistantMessage] = UserMessage

        if isinstance(message, (AgentTrace, ConductorResponse)):
            content_to_log = str(message.content)[:TRUNCATE_LEN]
            msg_type = AssistantMessage
        elif isinstance(message, StepRequest):
            if message.content and not message.content.startswith(COMMAND_SYMBOL):
                content_to_log = str(message.content)[:TRUNCATE_LEN]
        elif isinstance(message, ManagerMessage):
            content = getattr(message, "content", getattr(message, "params", None))
            if content and not str(content).startswith(COMMAND_SYMBOL):
                content_to_log = str(content)[:TRUNCATE_LEN]

        if content_to_log:
            await self._model_context.add_message(msg_type(content=content_to_log, source=source))

        # Store user feedback if available
        if isinstance(message, ManagerResponse) and message.prompt:
            self._user_feedback.append(message.prompt)
            await self._model_context.add_message(
                UserMessage(content=f"User feedback: {message.prompt[:TRUNCATE_LEN]}", source="USER"),
            )

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        public_callback: Callable | None = None,
        message_callback: Callable | None = None,
        **kwargs: Any,
    ) -> OOBMessages | None:
        """Handle special events and messages."""
        logger.debug(f"Host {self.agent_id} handling event: {type(message).__name__}")

        # Handle ManagerResponse
        if isinstance(message, ManagerResponse):
            # Check if the message is a user confirmation
            if message.confirm:
                logger.info(f"Host {self.agent_id} received user confirmation.")
                # Store the confirmation in the queue for later processing
                try:
                    self._user_confirmation.put_nowait(message)
                except asyncio.QueueFull:
                    msg = f"Discarding user input because earlier input still hasn't been handled: {message}"
                    logger.error(msg)
                    if public_callback:
                        await public_callback(ErrorEvent(source=self.agent_id, content=msg))

            return None  # Return after handling non-interrupt ManagerResponse

        # Handle task completion signals from worker agents.
        if isinstance(message, TaskProcessingComplete):
            agent_id_to_update = message.agent_id
            should_send_progress = False  # Flag to send progress update outside the lock
            async with self._tasks_condition:
                if agent_id_to_update in self._pending_tasks_by_agent:
                    self._pending_tasks_by_agent[agent_id_to_update] -= 1
                    log_prefix = f"Host received TaskComplete from {message.agent_id} (ID: {agent_id_to_update}) for role '{message.role}'."

                    # If count reaches zero, remove the agent from pending
                    if self._pending_tasks_by_agent[agent_id_to_update] == 0:
                        del self._pending_tasks_by_agent[agent_id_to_update]
                        logger.info(f"{log_prefix} Agent {agent_id_to_update} has no more pending tasks.")
                    else:
                         logger.info(
                            f"{log_prefix} "
                            f"Agent {agent_id_to_update} has {self._pending_tasks_by_agent[agent_id_to_update]} remaining tasks. "
                            f"Task {message.task_index}, More: {message.more_tasks_remain}, Error: {message.is_error}",
                         )

                    # If the entire dictionary is now empty, notify waiters
                    if not self._pending_tasks_by_agent:
                        logger.info("All pending tasks completed across all agents. Notifying waiters.")
                        self._tasks_condition.notify_all()
                        # Mark that progress should be sent after releasing the lock
                        should_send_progress = True

                    # Log remaining pending tasks regardless
                    logger.debug(f"Current pending tasks: {dict(self._pending_tasks_by_agent)}")

                else:
                    logger.warning(
                        f"Host received TaskComplete from agent {message.agent_id} (ID: {agent_id_to_update}) for role '{message.role}', "
                        "but this agent ID was not tracked with pending tasks.",
                    )
                    # should_send_progress remains False

            # Send progress update outside the lock if needed
            if should_send_progress:
                await self._send_progress_update(
                    role=self._current_step_name or "UNKNOWN_STEP",
                    status="completed",
                    message=f"All tasks completed after step {self._current_step_name}",
                    progress=(self._current_step / self._total_steps if self._total_steps > 0 else 1.0),
                )

        elif isinstance(message, TaskProcessingStarted):
            agent_id = message.agent_id
            async with self._tasks_condition:
                self._pending_tasks_by_agent[agent_id] += 1
                logger.debug(
                    f"Host noted TaskStarted from agent {agent_id} for role '{message.role}'. "
                    f"Pending tasks: {dict(self._pending_tasks_by_agent)}.",
                )

            # Send a progress update that a task has started for the current step context
            await self._send_progress_update(
                role=self._current_step_name or message.role,  # Use current step name if available
                status="started",
                message=f"Task started by agent {agent_id} (Role: {message.role})",
                # Progress reflects the overall sequence progress
                progress=(self._current_step / self._total_steps if self._total_steps > 0 else 0.0),
            )

        # Handle conductor request to start running the flow
        elif isinstance(message, ConductorRequest):
            if not self._conductor_task or self._conductor_task.done():
                if self._conductor_task and self._conductor_task.done():
                    try:
                        self._conductor_task.result()
                    except asyncio.CancelledError:
                        logger.info("Previous conductor task was cancelled.")
                    except Exception as e:
                        logger.error(f"Previous conductor task ended with exception: {e}")
                logger.info(f"Host {self.agent_id} starting new conductor task.")
                # Reset state before starting a new flow
                await self.on_reset()  # Ensure clean state
                self._conductor_task = asyncio.create_task(self._run_flow(message=message))
            else:
                logger.warning(f"Host {self.agent_id} received ConductorRequest but task is already running.")

        return None  # Explicitly return None if no other value is returned

    async def request_user_confirmation(self, step: StepRequest) -> None:
        """Request confirmation from the user for the next step.
            
        This method is used when human_in_loop is True to get user approval
        before executing a step.
        
        Args:
            step: The proposed next step
            
        Returns:
            None: This method does not return a value directly but sends a request.

        """
        request_content = (
            f"**Next Proposed Step:**\n"
            f"- **Agent Role:** {step.role}\n"
            f"- **Description:** {step.content or '(No description)'}\n"
            f"- **Prompt Snippet:** {step.prompt[:100] + '...' if step.prompt else '(No prompt)'}\n\n"
            f"Confirm (Enter), provide feedback, or reject ('n'/'q')."
        )
        logger.debug(f"Requesting info from user about proposed step {step.role}.")
        confirmation_request = ManagerRequest(
            role=MANAGER, prompt=request_content, inputs={"confirm": True, "selection": [True, False]},
        )
        if self._input_callback:
            await self._input_callback(confirmation_request)
        else:
            logger.error(f"Host {self.agent_id} cannot request user confirmation: _input_callback is not set.")
            # Depending on desired behavior, could raise an error here instead of just logging

    async def _wait_for_user(self, step) -> ManagerResponse:
        await self.request_user_confirmation(step)
        if not self.human_in_loop:
            logger.warning("_wait_for_user called but host config has human in the loop disabled. Simulating user confirmation.")
            return ManagerResponse(prompt=None, confirm=True, halt=False, interrupt=False, selection=None)
        try:
            response = await asyncio.wait_for(self._user_confirmation.get(), timeout=self.max_wait_time)
            logger.debug(f"Received manager response: confirm = {response.confirm}, halt = {response.halt}, interrupt = {response.interrupt}, selection = {response.selection}")
            return response
        except TimeoutError:
            logger.warning(f"{self.agent_id} hit timeout waiting for manager response after {self.max_wait_time} seconds.")
            return await self._wait_for_user(step)

    async def _wait_for_all_tasks_complete(self) -> None:
        """Wait until the dictionary of pending tasks is empty using a Condition."""
        # First, sleep a bit to allow other tasks to run
        await asyncio.sleep(3)
        try:
            async with self._tasks_condition:
                if self._pending_tasks_by_agent:
                    logger.debug(f"Waiting for pending tasks to complete: {dict(self._pending_tasks_by_agent)}...")
                    # wait_for releases the lock, waits for notification and predicate, then reacquires
                    await asyncio.wait_for(
                        self._tasks_condition.wait_for(lambda: not self._pending_tasks_by_agent),
                        timeout=self.max_wait_time,
                    )
                    logger.debug("All pending tasks completed (condition notified).")
                else:
                    logger.debug("No pending tasks to wait for.")
            # Lock is released here
        except TimeoutError as e:
            # Lock is released automatically on timeout exception from wait_for
            msg = (
                f"Timeout waiting for task completion condition. "
                f"Pending tasks: {dict(self._pending_tasks_by_agent)} after {self.max_wait_time}s."
            )
            logger.error(f"{msg} Aborting.")
            # No need to manually reset count or notify here, but raise FatalError
            raise FatalError(msg) from e
        except Exception as e:
            # Catch other potential errors during wait
            logger.exception(f"Unexpected error during task completion wait: {e}")
            raise

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Generate a sequence of steps to execute based on participants.

        Yields:
            StepRequest: The next step request in the sequence.

        """
        # Wait until participants are set
        await self._participants_set_event.wait()

        self._step_sequence = list(self._participants.keys())
        self._total_steps = len(self._step_sequence)
        logger.info(f"Host {self.agent_id} initialized sequence with {self._total_steps} steps: {self._step_sequence}")

        for i, (step_name, _cfg) in enumerate(self._participants.items()):  # Use _cfg to indicate unused var
            self._current_step = i + 1  # 1-based step index for progress
            logger.info(f"Host {self.agent_id} generating step {self._current_step}/{self._total_steps}: {step_name}")
            yield StepRequest(role=step_name, content=f"Sequence host calling {step_name}.")

        logger.info(f"Host {self.agent_id} sequence completed, generating END step")
        yield StepRequest(role=END, content="Sequence wrapping up.")

    async def _run_flow(self, message: ConductorRequest) -> StepRequest:
        """Run the predefined sequence of steps, waiting for task completion between steps.

        Args:
            message: The initial request containing participants.

        Returns:
            StepRequest: The final END step request.

        Raises:
            FatalError: If no participants are provided in the request.

        """
        logger.info(f"Host {self.agent_id} starting flow execution.")

        # Initialize participants from the request
        if not self._participants:
            self._participants.update(message.participants)
            if not self._participants:
                msg = "Host received ConductorRequest with no participants."
                logger.error(f"{msg} Aborting.")
                raise FatalError(msg)
            # Signal that participants are now set
            self._participants_set_event.set()
            # Re-initialize generator now that participants are known (it waits on the event)
            self._step_generator = self._sequence()
            logger.info(f"Host participants initialized to: {list(self._participants.keys())}")

        while True:
            # Mark ourselves as processing
            await self._input_callback(TaskProcessingStarted(role=CONDUCTOR, agent_id=self.agent_id, task_index=-1))
            # Get the next step from the sequence generator
            step = await self._choose()  # No message needed

            if step.role == END:
                logger.info("Host sequence finished. Waiting for final tasks before ending.")
                break  # Exit the loop to perform final wait and send END

            if step.role == WAIT:
                 # If it's a WAIT step, still wait for any potentially outstanding tasks from previous steps
                 # before proceeding with the artificial sleep/wait.
                 logger.info("WAIT step: Completed waiting for any prior tasks. Now pausing.")
                 # Add artificial wait if needed for WAIT step, e.g., asyncio.sleep(5)
                 await asyncio.sleep(5)  # Example: Wait for 5 seconds for WAIT step

            # --- Execute the current step ---
            # Only execute if not a WAIT step (WAIT logic handled above)
            else:
                await self._wait_for_user(step)  # Also includes user wait if human_in_loop=True
                logger.info(f"Host proceeding with step for role: {step.role}")
                await self._execute_step(step)
                await asyncio.sleep(5)  # Allow time for the agent to process the request
                # clear our pending wait
                await self._input_callback(TaskProcessingComplete(role=CONDUCTOR, agent_id=self.agent_id, task_index=-1))

                # wait for other tasks
                await self._wait_for_all_tasks_complete()
                logger.info(f"Host completed waiting period after step: {step.role}")

        # --- Sequence finished, perform final wait and send END ---
        await self._wait_for_all_tasks_complete()
        logger.info("All tasks completed after final step. Sending END signal.")

        # Send a final progress update for the complete workflow
        await self._send_progress_update(
            role="WORKFLOW",
            status="completed",
            message="Workflow completed successfully",
            progress=1.0,
        )

        logger.info(f"Host {self.agent_id} flow execution finished.")
        self._current_step_name = None  # Clear current step after finishing
        # Send the END step to signal completion
        end_step = StepRequest(role=END, content="Flow completed.")
        await self._input_callback(end_step)
        return

    async def _execute_step(self, step: StepRequest) -> None:
        """Send the StepRequest for the current step."""
        if step.role in (END, WAIT):  # Use 'in' for multiple comparisons
            logger.debug(f"Skipping execution for control step: {step.role}")
            self._current_step_name = step.role  # Still note the step name
            # Ensure condition is notified if skipping execution and no agents are pending
            async with self._tasks_condition:
                if not self._pending_tasks_by_agent:
                    self._tasks_condition.notify_all()
            return

        logger.info(f"Host executing step for role: {step.role}")
        self._current_step_name = step.role

        message_content = step.content or f"Executing step for role {step.role}"
        # Pass empty list if _records doesn't exist
        records = getattr(self, "_records", [])
        message = StepRequest(role=step.role, content=message_content, records=records)

        logger.info(f"Host {self.agent_id} publishing StepRequest for role {step.role}...")
        await self._input_callback(message)
        logger.debug(f"Host {self.agent_id} successfully published StepRequest for role {step.role}")

        # Wait a bit to allow the agent to process the request
        await asyncio.sleep(5)

    def _store_exploration_result(self, execution_id: str, output: AgentTrace) -> None:
        """Store the result of an agent execution in the exploration history."""
        if execution_id not in self._exploration_path:
            self._exploration_path.append(execution_id)
        self._exploration_results[execution_id] = {
            "id": output.call_id,
            "role": self._current_step_name or "unknown",  # Use current step name if available
            "inputs": getattr(output, "inputs", {}),
            "outputs": getattr(output, "outputs", getattr(output, "contents", "")),
            "is_error": getattr(output, "is_error", False),
            "error_details": getattr(output, "error", []) if getattr(output, "is_error", False) else None,
            "metadata": getattr(output, "metadata", {}),
        }
        logger.debug(f"Stored result for execution {execution_id}")

    async def _choose(self) -> StepRequest:
        """Choose the next step from the sequence generator.

        Returns:
            StepRequest: The next step request, or an END step if the sequence is complete.

        """
        if self._step_generator is None:
            # Should not happen if initialized correctly, but handle defensively
            logger.error("Step generator not initialized in _choose.")
            return StepRequest(role=END, content="Error: Step generator not available.")
        try:
            # Use anext() builtin instead of dunder method
            return await anext(self._step_generator)
        except StopAsyncIteration:
            logger.info(f"Host {self.agent_id} step generator exhausted, explicitly returning END step")
            return StepRequest(role=END, content="Sequence completed. All steps have been processed.")

    async def _send_progress_update(self, role: str, status: str, message: str, progress: float = 0.0) -> None:
        """Send a progress update to the UI agent."""
        try:
            # Ensure progress is between 0 and 1
            progress = max(0.0, min(1.0, progress))
            progress_percent = int(100 * progress)

            # Create progress update message
            progress_update = TaskProgressUpdate(
                source=self.agent_id,
                role=role,  # Role associated with the event/step
                step_name=self._current_step_name or role,  # Current step context
                status=status,
                message=message,
                # Use overall sequence progress for the main bar
                total_steps=100,  # Represent as percentage
                current_step=progress_percent,
            )

            await self._input_callback(progress_update)
            logger.debug(f"Sent progress update for {role}: {status} - {message} (Overall: {progress_percent}%)")

        except Exception as e:  # Catch specific exceptions if possible
            logger.error(f"Error sending progress update: {e}")

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the HostAgent's internal state."""
        logger.info(f"Resetting Host agent {self.agent_id}...")
        await super().on_reset(cancellation_token)
        self._current_step_name = None
        # self._total_outstanding_tasks = 0 # Removed
        self._participants.clear()
        self._participants_set_event.clear()  # Reset participants event
        # Re-initialize the step generator (will wait for participants again).
        self._step_generator = self._sequence()
        # Re-initialize condition and pending tasks dict
        self._tasks_condition = asyncio.Condition()
        self._pending_tasks_by_agent = defaultdict(int)
        # Clear user feedback list
        self._user_feedback = []
        # Reset exploration tracking
        self._exploration_path = []
        self._exploration_results = {}
        # Reset progress tracking
        self._total_steps = 0
        self._current_step = 0
        self._step_sequence = []
        # Cancel existing conductor task if running
        if self._conductor_task and not self._conductor_task.done():
            self._conductor_task.cancel()
            try:
                await self._conductor_task
            except asyncio.CancelledError:
                logger.debug("Conductor task cancelled during reset.")
            except Exception as e:  # Catch specific exceptions if possible
                logger.error(f"Error retrieving result from cancelled conductor task during reset: {e}")
        self._conductor_task = None

        logger.info(f"Host agent {self.agent_id} reset complete.")

    async def _process(
        self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs: Any,
    ) -> AgentResponse:
        """Host _process implementation - not typically used directly.

        Returns:
            AgentResponse: An error response indicating this method is not used.

        """
        # Method arguments are part of the base class signature, keep them but mark as unused if needed
        _ = message
        _ = cancellation_token
        _ = kwargs
        placeholder = ErrorEvent(source=self.agent_id, content="Host agent does not process direct inputs via _process")
        return AgentResponse(agent_id=self.agent_id, outputs=placeholder)


# Helper function to replace asyncio.sleep(0) or similar busy-waits if needed elsewhere
async def yield_control() -> None:
    """Yield control to the event loop briefly."""
    await asyncio.sleep(0)

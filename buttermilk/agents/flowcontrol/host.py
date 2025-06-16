import asyncio
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable
from typing import Any  # Import Dict

from autogen_core import CancellationToken
from autogen_core.models import AssistantMessage, UserMessage
from pydantic import Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER, WAIT
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    AgentTrace,
    ConductorRequest,
    ErrorEvent,
    FlowProgressUpdate,
    GroupchatMessageTypes,
    ManagerMessage,
    OOBMessages,
    StepRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
    UIMessage,
)
from buttermilk._core.exceptions import FatalError

TRUNCATE_LEN = 1000  # characters per history message


class HostAgent(Agent):
    """Base coordinator for group chats and flow control.

    This agent acts as a conductor (`CONDUCTOR` role). It iterates through a
    predefined sequence of participant roles. For each step, it sends a request
    to the corresponding role. Before proceeding to the next step, it waits
    until all agents that have started processing a task have reported completion.
    Uses a dictionary to count pending tasks per agent ID and a Condition variable
    for synchronization.
    """
    

    _message_types_handled: type[Any] = PrivateAttr(default=type(ConductorRequest))
    callback_to_groupchat: Any = Field(default=None)
    _step_generator: AsyncGenerator[StepRequest, None] | None = PrivateAttr(default=None)
    # Condition variable to synchronize task completion based on pending agents
    _tasks_condition: asyncio.Condition = PrivateAttr(default_factory=asyncio.Condition)
    _step_starting: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    # Track count of pending tasks per agent ID
    _pending_tasks_by_agent: defaultdict[str, int] = PrivateAttr(default_factory=lambda: defaultdict(int))  # Corrected duplicate definition
    _participants: dict[str, Any] = PrivateAttr(default_factory=dict)  # Stores role descriptions
    _conductor_task: asyncio.Task | None = PrivateAttr(default=None)
    # Additional configuration
    max_wait_time: int = Field(
        default=240,
        description="Maximum time to wait for agent responses in seconds",
    )
    _current_step: str = PrivateAttr(default="")
    max_user_confirmation_time: int = Field(
        default=1220,
        description="Maximum time to wait for agent responses in seconds",
    )
    # human_in_loop is now read from self.parameters instead of being a direct field
    @property
    def human_in_loop(self) -> bool:
        """Whether to interact with the human/manager for step confirmation.
        
        Gets the value from parameters dict, defaulting to True if not specified.
        """
        return self.parameters.get('human_in_loop', True)
    
    @human_in_loop.setter
    def human_in_loop(self, value: bool) -> None:
        """Set the human_in_loop value in parameters."""
        self.parameters['human_in_loop'] = value
    #  Event for confirmation responses from the MANAGER.
    _user_confirmation: ManagerMessage | None = PrivateAttr(default=None)
    _user_confirmation_received: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _user_feedback: list[str] = PrivateAttr(default_factory=list)
    _progress_reporter_task: asyncio.Task | None = PrivateAttr(default=None)

    async def initialize(
        self,
        callback_to_groupchat,
        **kwargs: Any,
    ) -> None:
        """Initialize the agent."""
        super().initialize(**kwargs)
        self.callback_to_groupchat = callback_to_groupchat
        logger.info(f"HostAgent {self.agent_name} initialized with human_in_loop={self.human_in_loop}")

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken,
        source: str = "",
        public_callback: Callable,
        message_callback: Callable,
        **kwargs: Any,
    ) -> None:
        """Listen to messages in the group chat and maintain conversation history."""
        # Log messages to our local context cache, but truncate them
        content_to_log = None
        # Allow both UserMessage and AssistantMessage types
        msg_type: type[UserMessage | AssistantMessage] = UserMessage

        if isinstance(message, (AgentTrace)):
            content_to_log = str(message.content)[:TRUNCATE_LEN]
            msg_type = AssistantMessage
        elif isinstance(message, ManagerMessage):
            logger.info(f"Host {self.agent_name} received user input: {message}")
            self._user_confirmation = message
            self._user_confirmation_received.set()

            if message.human_in_loop is not None and self.human_in_loop != message.human_in_loop:
                logger.info(
                    f"Host {self.agent_name} received user request to set human in the loop to {message.human_in_loop} (was {self.human_in_loop})"
                )
                self.human_in_loop = message.human_in_loop

            content = getattr(message, "content", getattr(message, "params", None))
            if content and not str(content).startswith(COMMAND_SYMBOL):
                content_to_log = str(content)[:TRUNCATE_LEN]
                # store in user feedback separately as well
                self._user_feedback.append(content)

        # Do not log TaskProgressUpdate messages to history
        elif isinstance(message, FlowProgressUpdate):
            logger.debug(f"Host {self.agent_name} received TaskProgressUpdate (not logged to history): {self._pending_tasks_by_agent}")
            return  # Do not proceed to log
        if content_to_log:
            await self._model_context.add_message(msg_type(content=content_to_log, source=source))

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken,
        public_callback: Callable,
        message_callback: Callable,
        **kwargs: Any,
    ) -> OOBMessages | None:
        """Handle special events and messages."""
        logger.debug(f"Host {self.agent_name} handling event: {type(message).__name__}")

        # Handle task completion signals from worker agents.
        if isinstance(message, TaskProcessingComplete):
            self._step_starting.clear()  # Clear this event as soon as any task completes? Revisit this logic.
            agent_id_to_update = message.agent_id
            async with self._tasks_condition:
                if agent_id_to_update in self._pending_tasks_by_agent:
                    # Check if the dictionary is empty *before* potentially making it empty
                    was_empty_before_update = not self._pending_tasks_by_agent

                    self._pending_tasks_by_agent[agent_id_to_update] -= 1
                    log_prefix = f"Host received TaskComplete from {message.agent_id} (ID: {agent_id_to_update}) for role '{message.role}'."

                    # If count reaches zero, remove the agent from pending
                    if self._pending_tasks_by_agent[agent_id_to_update] <= 0:  # Use <= 0 to handle potential negative counts from errors
                        if self._pending_tasks_by_agent[agent_id_to_update] < 0:
                            logger.warning(
                                f"{log_prefix} Task count went negative ({self._pending_tasks_by_agent[agent_id_to_update]}). This might indicate an issue."
                            )
                        del self._pending_tasks_by_agent[agent_id_to_update]
                        logger.debug(f"{log_prefix} Agent {agent_id_to_update} has no more pending tasks.")
                    else:
                        logger.debug(
                            f"{log_prefix} "
                            f"Agent {agent_id_to_update} has {self._pending_tasks_by_agent[agent_id_to_update]} remaining tasks. "
                            f"Task {message.task_index}, More: {message.more_tasks_remain}, Error: {message.is_error}",
                        )

                    # If the entire dictionary is now empty AND it was NOT empty before this update, notify waiters
                    if not self._pending_tasks_by_agent and not was_empty_before_update:
                        logger.info("All pending tasks completed across all agents. Notifying waiters.")
                        self._tasks_condition.notify_all()

                    # Log remaining pending tasks regardless
                    logger.debug(f"Current pending tasks: {dict(self._pending_tasks_by_agent)}")

                else:
                    logger.warning(
                        f"Host received TaskComplete from agent {message.agent_id} (ID: {agent_id_to_update}) for role '{message.role}', "
                        "but this agent ID was not tracked with pending tasks.",
                    )

        elif isinstance(message, TaskProcessingStarted):
            agent_id = message.agent_id
            async with self._tasks_condition:
                self._pending_tasks_by_agent[agent_id] += 1
                logger.debug(
                    f"Host noted TaskStarted from agent {agent_id} for role '{message.role}'. "
                    f"Pending tasks: {dict(self._pending_tasks_by_agent)}.",
                )

        # Handle conductor request to start running the flow
        elif isinstance(message, ConductorRequest):
            if self._conductor_task and not self._conductor_task.done():
                logger.warning(f"Host {self.agent_name} received ConductorRequest but task is already running.")
                return None
            self._conductor_task = "starting"  # Mark as starting to avoid re-entrance -- this is a temporary state
            # If no task is running, start a new one
            logger.info(f"Host {self.agent_name} starting new conductor task.")
            self._conductor_task = asyncio.create_task(self._run_flow(message=message))

        # Ignore FlowProgressUpdate messages received by the host itself
        elif isinstance(message, FlowProgressUpdate):
            logger.debug(f"Host {self.agent_name} received its own TaskProgressUpdate message. Ignoring.")
            # Do nothing with progress updates received by the host

        return None  # Explicitly return None if no other value is returned

    async def request_user_confirmation(self, step: StepRequest) -> None:
        """Request confirmation from the user for the next step.

        This method is used when human_in_loop is True to get user approval
        before executing a step.

        Args:
            step: The proposed next step

        Returns:
            None

        """
        self._user_confirmation_received.clear()
        # Send the request to the user
        confirmation_request = UIMessage(
            content=step.content or f"Confirm next step: {step.role}",
            options=["confirm", "reject"],
        )
        # This is callback to groupchat, because we're the host agent right now.
        # Our message goes to the groupchat and then gets picked up by the UI.
        await self.callback_to_groupchat(confirmation_request)

    async def _wait_for_user(self, step) -> bool:
        """Wait for user confirmation before proceeding with the next step.

        Returns:
            bool: True if user confirmed, False if rejected or timed out

        """
        logger.info(f"Host {self.agent_name}: _wait_for_user called for step {step.role}")
        max_tries = self.max_user_confirmation_time // 60
        for _ in range(max_tries):
            logger.info(f"Host {self.agent_name} waiting for user confirmation for {step.role} step.")
            try:
                await self.request_user_confirmation(step)
                await asyncio.wait_for(self._user_confirmation_received.wait(), timeout=60)
            except TimeoutError:
                logger.info(f"{self.agent_name} hit timeout waiting for manager response after 60 seconds.")
                continue

            if self._user_confirmation and getattr(self._user_confirmation, "confirm", False):
                logger.info(f"User confirmed step: {step.role}")
                return True
            logger.info(f"User rejected step: {step.role}")
            return False
        msg = "User did not respond to confirm step in time. Ending flow."
        logger.error(msg)
        await self.callback_to_groupchat(StepRequest(role=END, content=msg))
        return False

    async def _report_progress_periodically(self, interval: int = 10):
        """Periodically report the status of pending tasks."""
        try:
            while True:
                await asyncio.sleep(interval)
                async with self._tasks_condition:
                    if self._pending_tasks_by_agent:
                        progress_message = FlowProgressUpdate(
                            source=self.agent_id,
                            status="pending",
                            step_name=self._current_step,
                            waiting_on=dict(self._pending_tasks_by_agent),
                            message="Current pending tasks",
                        )
                    else:
                        progress_message = FlowProgressUpdate(
                            source=self.agent_id,
                            status="idle",
                            step_name="IDLE",
                            waiting_on=dict(),
                            message="Flow currently idle",
                        )
                    logger.debug(f"Host {self.agent_name} sending progress update: {progress_message.status} {progress_message.waiting_on}")
                    await self.callback_to_groupchat(progress_message)
        except asyncio.CancelledError:
            logger.debug("Progress reporting task cancelled.")
        except Exception as e:
            logger.exception(f"Error in progress reporting task: {e}")
        finally:
            logger.debug(f"Host {self.agent_name} progress reporting task terminated.")

    async def _wait_for_all_tasks_complete(self) -> bool:
        """Wait until all tasks are completed, reporting progress periodically."""
        try:
            async with self._tasks_condition:
                if self._pending_tasks_by_agent:
                    logger.info(f"Waiting for pending tasks to complete from: {list(self._pending_tasks_by_agent.keys())}...")

                # wait_for releases the lock, waits for notification and predicate, then reacquires
                # The predicate checks if _step_starting is clear AND _pending_tasks_by_agent is empty.
                # This means we wait until the step is no longer considered "starting" AND all tasks are done. This provides insurance where
                # distributed tasks take a while to begin.
                await asyncio.wait_for(
                    self._tasks_condition.wait_for(lambda: not self._step_starting.is_set() and not self._pending_tasks_by_agent),
                    timeout=self.max_wait_time,
                )
                return True
        except TimeoutError:
            # Lock is released automatically on timeout exception from wait_for
            msg = (
                f"Timeout waiting for task completion condition. " f"Pending tasks: {dict(self._pending_tasks_by_agent)} after {self.max_wait_time}s."
            )
            logger.warning(msg)
            self._step_starting.clear()  # Reset the event to allow for new steps
            return False  # Indicate failure due to timeout
        except Exception as e:
            # Catch other potential errors during wait
            logger.exception(f"Unexpected error during task completion wait: {e}")
            self._step_starting.clear()  # Reset on error too
            return False  # Indicate failure due to error

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Generate a sequence of steps to execute.

        Yields:
            StepRequest: The next step request in the sequence.

        """
        for role in self._participants.keys():
            yield StepRequest(role=role, content=f"Executing step for {role}")
        yield StepRequest(role=END, content="Sequence completed.")

    async def _run_flow(self, message: ConductorRequest) -> None:
        """Run the predefined sequence of steps, waiting for task completion between steps.

        Args:
            message: The initial request containing participants.

        Returns:
            StepRequest: The final END step request.

        Raises:
            FatalError: If no participants are provided in the request.

        """
        logger.info(f"Host {self.agent_name} starting flow execution.")

        # Initialize participants from the request
        self._participants.update(message.participants)
        if not self._participants:
            msg = "Host received ConductorRequest with no participants."
            logger.error(f"{msg} Aborting.")
            # Send an END message with the error
            await self.callback_to_groupchat(StepRequest(role=END, content=msg))
            raise FatalError(msg)

        # Rerun initialization to set up the group chat
        await self.initialize(callback_to_groupchat=self.callback_to_groupchat)

        # Start the periodic progress reporter task
        self._progress_reporter_task = asyncio.create_task(self._report_progress_periodically())

        # Initialize generator now that participants are known
        self._step_generator = self._sequence()
        logger.info(f"Host participants initialized to: {list(self._participants.keys())}")

        async for next_step in self._step_generator:
            logger.info(f"Host {self.agent_name}: Processing step {next_step.role}")
            # Wait for tasks from the *previous* step to complete before starting the *next* step
            if not await self.wait_check_last_step_completions():
                # If the wait failed (timeout or error), stop the flow
                # break
                pass
            # Don't seek confirmation from the manager to send a request to the manager
            logger.info(f"Host {self.agent_name}: human_in_loop={self.human_in_loop}, next_step.role={next_step.role}, MANAGER={MANAGER}")
            if self.human_in_loop and next_step.role != MANAGER and not await self._wait_for_user(next_step):
                # If user rejected or timed out, stop the flow
                continue
                # break

            # Execute the current step
            await self._execute_step(next_step)

        # --- Sequence finished ---
        logger.info(f"Host {self.agent_name} flow execution finished.")

        # Send final progress update before any cleanup begins
        final_progress_message = FlowProgressUpdate(
            source=self.agent_id,
            status="finished",
            step_name=END,
            waiting_on={},
            message="Flow completed",
        )
        logger.info(f"Host {self.agent_name} sending final progress update before cleanup.")
        await self.callback_to_groupchat(final_progress_message)

    async def _shutdown(self) -> None:
        """Shutdown the agent and clean up resources."""
        if self._conductor_task:
            self._conductor_task.cancel()
            try:
                await self._conductor_task
            except asyncio.CancelledError:
                logger.debug("Conductor task cancelled.")
        if self._step_generator:
            await self._step_generator.aclose()
        if self._tasks_condition:
            await self._tasks_condition.release()
            # Ensure the progress reporter task is cancelled when the wait finishes or times out
            if self._progress_reporter_task:
                self._progress_reporter_task.cancel()
                try:
                    await self._progress_reporter_task  # Await cancellation
                except asyncio.CancelledError:
                    pass  # Expected
        logger.info(f"Host {self.agent_name} shutdown complete.")

    async def wait_check_last_step_completions(self) -> bool:
        """Wait for tasks from the previous step to complete."""
        # Wait for pending tasks to complete
        last_step_successful = await self._wait_for_all_tasks_complete()
        if not last_step_successful:
            msg = f"Host {self.agent_id} failed to complete all tasks for the previous step: {dict(self._pending_tasks_by_agent)}"
            logger.error(msg)
            return False
        # If successful, clear the pending tasks dictionary for the next step
        async with self._tasks_condition:
            self._pending_tasks_by_agent.clear()
        logger.info("No pending tasks left over from previous steps, clear to proceed.")
        return True

    async def _execute_step(self, step: StepRequest) -> None:
        """Process a single step."""
        logger.info(f"Host executing step: {step}")
        self._current_step = step.role
        if step.role == WAIT:
            logger.info("Host waiting for 10 seconds as requested by WAIT step.")
            await asyncio.sleep(10)
        elif step.role == END:
            logger.info(f"Flow completed and all tasks finished. Sending END signal: {step}")
            await self.callback_to_groupchat(step)
        else:
            if step.role in self._participants:
                # Signal that we expect at least one response/task start for this step
                # This event is used in the wait_for predicate.
                self._step_starting.set()
                logger.debug(f"Host set _step_starting event for role: {step.role}")
            elif step.role == MANAGER:
                # MANAGER steps don't spawn trackable worker tasks, so don't set _step_starting
                logger.debug(f"Host executing MANAGER step without setting _step_starting: {step.role}")
            else:
                logger.warning(f"Host executing step for unknown participant role: {step.role}")

            await self.callback_to_groupchat(step)

    async def _process(
        self,
        *,
        message: AgentInput,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> AgentOutput:
        # A non-LLM host will return a StepRequest

        placeholder = ErrorEvent(source=self.agent_id, content="Host agent does not process direct inputs via _process")
        return AgentOutput(agent_id=self.agent_id, outputs=placeholder)

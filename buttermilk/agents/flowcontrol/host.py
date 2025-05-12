import asyncio
from collections import defaultdict  # Import defaultdict
from collections.abc import AsyncGenerator, Callable
from typing import Any

from autogen_core import CancellationToken
from autogen_core.models import AssistantMessage, UserMessage
from pydantic import Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.constants import COMMAND_SYMBOL, END, WAIT
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    AgentTrace,
    ConductorRequest,
    ErrorEvent,
    GroupchatMessageTypes,
    ManagerMessage,
    ManagerRequest,
    OOBMessages,
    StepRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
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
    _pending_tasks_by_agent: defaultdict[str, int] = PrivateAttr(default_factory=lambda: defaultdict(int))
    _participants: dict[str, Any] = PrivateAttr(default_factory=dict)  # Stores role descriptions
    # Track count of pending tasks per agent ID
    _pending_tasks_by_agent: defaultdict[str, int] = PrivateAttr(default_factory=lambda: defaultdict(int))
    _conductor_task: asyncio.Task | None = PrivateAttr(default=None)
    # Additional configuration
    max_wait_time: int = Field(
        default=240,
        description="Maximum time to wait for agent responses in seconds",
    )

    max_user_confirmation_time: int = Field(
        default=240,
        description="Maximum time to wait for agent responses in seconds",
    )
    human_in_loop: bool = Field(
        default=True,
        description="Whether to interact with the human/manager for step confirmation",
    )
    #  Event for confirmation responses from the MANAGER.
    _user_confirmation: ManagerMessage | None = PrivateAttr(default=None)
    _user_confirmation_received: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _user_feedback: list[str] = PrivateAttr(default_factory=list)

    async def initialize(
        self, callback_to_groupchat, **kwargs: Any,
    ) -> None:
        """Initialize the agent."""
        self.callback_to_groupchat = callback_to_groupchat

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
            logger.info(f"Host {self.agent_id} received user input: {message}")
            self._user_confirmation = message
            self._user_confirmation_received.set()
            content = getattr(message, "content", getattr(message, "params", None))
            if content and not str(content).startswith(COMMAND_SYMBOL):
                content_to_log = str(content)[:TRUNCATE_LEN]
                # store in user feedback separately as well
                self._user_feedback.append(content)
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
        logger.debug(f"Host {self.agent_id} handling event: {type(message).__name__}")

        # Handle task completion signals from worker agents.
        if isinstance(message, TaskProcessingComplete):
            agent_id_to_update = message.agent_id
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
                logger.warning(f"Host {self.agent_id} received ConductorRequest but task is already running.")
                return None

            # If no task is running, start a new one
            logger.info(f"Host {self.agent_id} starting new conductor task.")
            self._conductor_task = asyncio.create_task(self._run_flow(message=message))

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
        self._user_confirmation = None
        logger.info(f"Requesting info from user about proposed step: {step}.")
        confirmation_request = ManagerRequest(
            content=str(step), options=["confirm", "reject"],  # Options for user confirmation
        )
        self._user_confirmation_received.clear()  # Reset the event for the next confirmation
        await self.callback_to_groupchat(confirmation_request)

    async def _wait_for_user(self, step) -> ManagerMessage:
        max_tries = self.max_user_confirmation_time // 60
        if self.human_in_loop:
            for i in range(max_tries):
                logger.debug(f"Host {self.agent_id} waiting for user confirmation for {step.role} step.")
                try:
                    await self.request_user_confirmation(step)
                    await asyncio.wait_for(self._user_confirmation_received.wait(), timeout=60)
                    return self._user_confirmation
                except TimeoutError:
                    logger.warning(f"{self.agent_id} hit timeout waiting for manager response after 60 seconds.")
                    continue
        raise FatalError("User did not respond in time.")

    async def _wait_for_all_tasks_complete(self) -> None:
        """Wait until all tasks are completed."""
        try:
            async with self._tasks_condition:
                logger.info(f"Waiting for pending tasks to complete: {dict(self._pending_tasks_by_agent)}...")
                # wait_for releases the lock, waits for notification and predicate, then reacquires
                await asyncio.wait_for(
                    self._tasks_condition.wait_for(lambda: not self._pending_tasks_by_agent),
                    timeout=self.max_wait_time,
                )
        except TimeoutError as e:
            # Lock is released automatically on timeout exception from wait_for
            msg = (
                f"Timeout waiting for task completion condition. "
                f"Pending tasks: {dict(self._pending_tasks_by_agent)} after {self.max_wait_time}s."
            )
            logger.error(f"{msg} Aborting.")
            # signal end to the conductor
            await self.callback_to_groupchat(StepRequest(role=END, content="Timeout waiting for task completion."))
            # No need to manually reset count or notify here, but raise FatalError
            raise FatalError(msg) from e
        except Exception as e:
            # Catch other potential errors during wait
            # signal end to the conductor
            await self.callback_to_groupchat(StepRequest(role=END, content=f"Unexpected error during task completion wait: {e}"))
            logger.exception(f"Unexpected error during task completion wait: {e}")
            raise

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
        logger.info(f"Host {self.agent_id} starting flow execution.")

        # Initialize participants from the request
        self._participants.update(message.participants)
        if not self._participants:
            msg = "Host received ConductorRequest with no participants."
            logger.error(f"{msg} Aborting.")
            raise FatalError(msg)

        # Initialize generator now that participants are known
        self._step_generator = self._sequence()
        logger.info(f"Host participants initialized to: {list(self._participants.keys())}")

        async for step in self._step_generator:
            if self.human_in_loop:
                await self._wait_for_user(step)
                if not self._user_confirmation or self._user_confirmation.confirm is False:
                    logger.info(f"User rejected step: {step.role}. Moving on.")
                    continue
            await self._execute_step(step)

            await asyncio.sleep(5)  # Allow time for the agent to process the request

        # --- Sequence finished, perform final wait and send END ---
        await self._wait_for_all_tasks_complete()

    async def _execute_step(self, step: StepRequest) -> None:
        """Process a single step."""
        await self._wait_for_all_tasks_complete()
        if step.role == WAIT:
            await asyncio.sleep(10)
        elif step.role == END:
            logger.info(f"Flow completed and all tasks finished. Sending END signal: {step}")
            await self.callback_to_groupchat(step)
        else:
            logger.info(f"Host executing step: {step}")
            await self.callback_to_groupchat(step)
        # Wait a bit to allow the agent to process the request
        await asyncio.sleep(5)

    async def _process(
        self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs: Any,
    ) -> AgentOutput:
        # A non-LLM host will return a StepRequest

        placeholder = ErrorEvent(source=self.agent_id, content="Host agent does not process direct inputs via _process")
        return AgentOutput(agent_id=self.agent_id, outputs=placeholder)
